"""Batched ("parallel-rotation") cyclic-Jacobi ``eigh`` lowering.

The sequential cyclic-Jacobi composition in ``writer.py``
(``_emit_eigh_jacobi``) performs ONE (p, q) rotation per ``stablehlo.while``
iteration, so a sweep of ``n(n-1)/2`` rotations costs that many iterations —
each one a handful of dispatches, index gathers and selects over the WHOLE
matrix. This module implements the batched variant
(``eigh_algorithm = "jacobi_batched"``, the default): one while iteration
applies ALL ``m`` mutually-DISJOINT rotations of a round at once, so a sweep
costs ``L`` iterations (``L = n-1`` for even ``n``, ``L = n`` for odd ``n``)
instead of ``n(n-1)/2`` — a ~``n/2``-fold reduction in loop iterations and in
the number of matrix-wide passes.

Design (mirrors the numpy prototype that pinned the formulas):

- **Round robin schedule.** For an even ``n`` the vertices ``0..n-2`` rotate
  around a FIXED vertex ``f = n-1``; for an odd ``n`` there is no fixed vertex
  and slot 0 of every round is a BYE (a forced-identity pseudo-rotation
  ``p == q == r``). With ``L`` rounds per sweep this produces every unordered
  pair exactly once per sweep.
- **Arithmetic index maps, no scatters.** The partner map and the angle
  assignment are computed per INDEX from hoisted iotas (``slot``,
  ``partner``, ``is_p``) and per SLOT from the round counter ``r`` (``p``,
  ``q``), so the rotation is two elementwise half-steps
  ``X' = c*X + s*take(X, partner, axis)`` — the SAME textbook formulas as the
  sequential body, applied to every rotation of the round simultaneously. The
  loop body therefore contains NO per-element masks, NO scatter regions and
  NO O(n) op count: its op count is CONSTANT in ``n`` (only the tensor types
  in the emitted text mention ``n``).
- **Layout / orientation.** The carried matrices live in the SAME transposed
  ``(n, n, batch...)`` layout as the sequential composition (so the shared
  post-loop tail ``_emit_eigh_finish`` works unchanged): A gets a dim-1
  (column) half-step followed by a dim-0 (row) half-step, V gets a single
  dim-0 (row) half-step (the carried V is ``V^T``). Rounding in ``numpy``
  pinned this orientation — a dim-1 V step breaks the eigenvector pairing.
- **Same loop shape as the sequential path.** One ``stablehlo.while``
  carrying ``(a, v, k)`` (plus the i1 ``done`` flag under
  ``eigh_early_exit``), cond region EXACTLY ``k < iters`` with
  ``iters = ceil(sweeps * L / U)``, and every constant/iota hoisted OUTSIDE
  the loop (iree 3.11 crashes on constants materialized inside while bodies
  — ``util.global.load`` undefined ``@__hoisted_*``).
- **Round unrolling (``U`` = ``eigh_unroll``, default
  ``writer._EIGH_UNROLL_DEFAULT``).** One while iteration applies ``U``
  consecutive schedule rounds — round ``k*U + u`` for ``u in range(U)`` —
  so the loop body costs ~``U``× the arithmetic but the loop runs
  ``total/U`` times. The composition is overhead-bound (every iteration is
  a handful of HAL dispatches + barriers, measured ~105-110 µs per
  iteration CONSTANT in ``n`` on iree/rocm), so unrolling is a near-linear
  win until the per-round matrix work starts to dominate. ``total`` is
  padded up to a multiple of ``U`` by REPEATING the tail rounds of the
  schedule: a repeated (p, q) rotation is an ordinary similarity transform
  (never a no-op guarantee failure — the angle formula simply re-zeroes an
  already-small off-diagonal), so the applied sequence stays the same
  cyclic-Jacobi schedule over the same fixed sweep count and the output
  contract (ascending ``w``, ``V``, stable pair-sort + gather column
  reorder) is unchanged.

This module imports only ``etl.core`` (+ numpy). It deliberately does NOT
import ``writer`` — ``Writer`` inherits :class:`EighBatchedMixin`, so every
helper below reaches the emission primitives through ``self``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from etl.core import BackendError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from etl.ir import Op


def eigh_batched_schedule(n: int) -> tuple[int, int, int | None]:
    """Static round-robin schedule for the batched cyclic-Jacobi rotation.

    Returns ``(L, m, fixed)``:

    - ``L`` — rounds per sweep (and the modulus of the round counter).
    - ``m`` — slots per round (``(L + 1) // 2``).
    - ``fixed`` — the FIXED vertex for an even ``n`` (``n - 1``); ``None``
      for an odd ``n`` (no fixed vertex — slot 0 of each round is a bye).

    Even ``n`` (``L = n-1``, ``m = n//2``): slot 0 pairs the fixed vertex
    ``f`` with ``r``; slots ``i >= 1`` pair ``(r+i, r-i)``. Odd ``n``
    (``L = n``, ``m = (n+1)//2``): slot 0 is the bye ``(r, r)``; slots
    ``i >= 1`` pair ``(r+i, r-i)``. Either way ``L`` rounds cover every
    unordered pair ``{a, b}``, ``a < b``, exactly once per sweep
    (``L*m = n(n-1)/2`` real rotations for even ``n``, ``L*(m-1)`` for odd).
    """
    if n < 2:
        raise BackendError(
            "stablehlo export: the batched Jacobi eigh schedule requires "
            f"n >= 2, got {n}"
        )
    if n % 2 == 0:
        L = n - 1
        fixed: int | None = n - 1
    else:
        L = n
        fixed = None
    m = (L + 1) // 2
    return L, m, fixed


class _RoundPrep:
    """The HOISTED (outside-the-loop) values and static facts ONE batched
    round emission needs.

    Built once per ``eigh`` op so the unrolled while body can emit ``U``
    rounds — each with a different runtime round index — without
    re-deriving the SSA names of the loop-invariant scalar constants,
    their broadcasts and the hoisted iotas (all referenced lexically from
    inside the body region; a value materialized INSIDE a while body
    breaks iree's constant hoisting, see the module docstring).
    """

    __slots__ = (
        "i64", "dtype", "n", "m", "batch", "fixed",
        "pre_shape", "vec_shape",
        "i_l", "i_two",
        "ti", "jj", "l_m", "l_n", "two_n", "two_l_n", "zero_n", "zero_m",
        "f_m", "f_n", "b_true_n", "b_false_n",
        "two_p", "one_p", "neg_one_p", "zero_p", "is_bye_b",
    )

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class EighBatchedMixin:
    """Mixin for ``Writer``: the batched cyclic-Jacobi ``eigh`` composition
    plus the pieces SHARED with the sequential composition (the while cond
    region and the post-loop tail)."""

    # ------------------------------------------------------------------
    # Shared pieces (used by BOTH eigh compositions)
    # ------------------------------------------------------------------

    def _emit_eigh_cond_region(self, arg_types, i_total, k_index: int) -> str:
        """The eigh while-loop COND region — ALWAYS ``k < total``.

        ``k_index`` is the position of the rotation counter among the
        carries (4 in the sequential composition, 2 in the batched one).
        The early-exit ``done`` flag never appears here: a loop-carried i1
        through ``not``/``and`` in a cond region crashes iree-compile
        3.11's AffinityAnalysis (measured 0/10 SIGSEGV); the body instead
        clamps ``k`` to ``total`` when ``done`` fires. No closing brace —
        the caller's ``_emit_while``-style assembly supplies it.
        """
        i64 = np.dtype("int64")
        cond_args = [self._new_name() for _ in range(len(arg_types))]
        cond_lines: list[str] = []
        cnd = self._cmp(cond_args[k_index], i_total, "LT", i64, (), cond_lines)
        return (
            "  ^bb0("
            + ", ".join(f"{nm} : {tp}" for nm, tp in zip(cond_args, arg_types))
            + "):\n"
            + "\n".join("    " + line for line in cond_lines)
            + f"\n    stablehlo.return {cnd} : tensor<i1>"
        )

    def _emit_eigh_finish(self, op: Op, a, v, dtype, x_shape, n, lines) -> None:
        """The post-loop tail shared by BOTH ``eigh`` compositions.

        The carried matrices arrive in the transposed ``(n, n, batch...)``
        layout; transpose back to ``(batch..., n, n)`` for the diagonal
        extract and the column reorder (the inverse of the init perm
        ``[rank-1, rank-2] + [0..rank-3]`` is ``[2..rank-1] + [1, 0]`` —
        batch dims in order, MATRIX DIMS REVERSED; both are involutions on
        the matrix block). A's back-transpose is only needed for batch dims
        (rank 2: an A-derived matrix stayed symmetric), V's is real at every
        rank (rank 2: ``[1, 0]``). Then: diagonal extract -> ``w``, a stable
        pair-sort for the ASCENDING eigenvalue order -> permutation, and a
        gather reorder of the V columns -> ``v``. Binds
        ``op.results[0]``/``[1]`` and appends every line to ``lines``.
        """
        rank = len(x_shape)
        batch = tuple(x_shape[:-2])
        t_shape = (n, n) + batch
        if batch:
            perm = list(range(2, rank)) + [1, 0]
            a2 = self._new_name()
            lines.append(
                f'{a2} = "stablehlo.transpose"({a}) '
                f"{{permutation = {self._i64_array(perm)}}} : "
                f"({self._type_str(dtype, t_shape)}) -> "
                f"{self._type_str(dtype, x_shape)}"
            )
            a = a2
        v_perm = list(range(2, rank)) + [1, 0]
        v2 = self._new_name()
        lines.append(
            f'{v2} = "stablehlo.transpose"({v}) '
            f"{{permutation = {self._i64_array(v_perm)}}} : "
            f"({self._type_str(dtype, t_shape)}) -> "
            f"{self._type_str(dtype, x_shape)}"
        )
        v = v2
        # The diagonal now holds the (unsorted) eigenvalues.
        w = self._emit_diagonal_extract(a, dtype, x_shape, n, lines)
        # Ascending order + the permutation, then the V column reorder.
        w_shape = batch + (n,)
        sv, si = self._emit_stable_argsort(
            w, dtype, w_shape, len(w_shape) - 1, lines
        )
        vs = self._emit_reorder_columns(v, si, dtype, x_shape, n, lines)
        self._names[id(op.results[0])] = sv
        self._names[id(op.results[1])] = vs

    # ------------------------------------------------------------------
    # Batched composition
    # ------------------------------------------------------------------

    def _emit_eigh_batched(self, op: Op, x, x_shape, n, batch, in_dtype,
                           dtype) -> str:
        """``eigh`` -> ONE ``stablehlo.while`` applying a whole ROUND of
        mutually-disjoint cyclic-Jacobi rotations per iteration.

        Carries ``(a, v, k)`` (+ i1 ``done`` under ``eigh_early_exit``);
        ``total = sweeps * L`` iterations (``L`` rounds per sweep — see
        :func:`eigh_batched_schedule`), cond region ``k < total``. ``r = k
        mod L`` selects the round; the per-slot pair ``(p, q)`` and the
        per-index (``slot``, ``partner``, ``is_p``) maps are derived
        ARITHMETICALLY from hoisted iotas, so no scatter/mask pass over the
        matrix is emitted. The angle per pair is the same textbook
        ``tau = (a_qq - a_pp)/(2 a_pq)`` formula as the sequential body
        (guarded so ``a_pq == 0`` -> ``(c, s) = (1, 0)`` AFTER the formulas,
        and — odd ``n`` only — the bye slot forced to identity), then each
        matrix takes the half-steps ``B = c*A + s*take(A, partner, 1)``,
        ``A' = c*B + s*take(B, partner, 0)``, with V getting the single
        dim-0 step (the carried V is ``V^T``). Every op count is CONSTANT in
        ``n``. Validation (dynamic dims, dtype rules, square check) is
        SHARED with the sequential path and has already run in
        ``_emit_eigh``.
        """
        i64 = np.dtype("int64")
        i1 = np.dtype("bool")
        rank = len(x_shape)
        kb = len(batch)
        L, m, fixed = eigh_batched_schedule(n)
        even = fixed is not None
        sweeps = self._eigh_sweeps(n, dtype)
        total = sweeps * L
        #: Round unrolling: ``U`` schedule rounds per while iteration, so
        #: the loop runs ``ceil(total / U)`` iterations. The tail is padded
        #: by REPEATING rounds of the schedule (see the module docstring) —
        #: same rotation sequence, same fixed sweep count.
        U = self._eigh_unroll
        n_iters = -(-total // U)
        t_shape = (n, n) + batch
        pre_shape = (m,) + batch      # per-SLOT values
        vec_shape = (n,) + batch      # per-INDEX values
        #: (m,)/(n,) values keep their source axis at position 0 and the
        #: batch dims trailing (dims 1..kb of the pre/vec shapes).
        batch_dims = list(range(1, kb + 1))
        lines: list[str] = []
        # --- carry inits: A (converted + transposed layout) and V = eye ---
        a = self._name(x)
        if dtype != in_dtype:
            t = self._new_name()
            lines.append(
                f"{t} = stablehlo.convert {a} : "
                f"({self._type_str(in_dtype, x_shape)}) -> "
                f"{self._type_str(dtype, x_shape)}"
            )
            a = t
        if batch:
            perm = [rank - 1, rank - 2] + list(range(rank - 2))
            a_t = self._new_name()
            lines.append(
                f'{a_t} = "stablehlo.transpose"({a}) '
                f"{{permutation = {self._i64_array(perm)}}} : "
                f"({self._type_str(dtype, x_shape)}) -> "
                f"{self._type_str(dtype, t_shape)}"
            )
            a = a_t
        eye = np.eye(n, dtype=dtype)
        v = self._new_name()
        lines.append(
            f"{v} = stablehlo.constant {self._constant_text(eye)} : "
            f"{self._type_str(dtype, (n, n))}"
        )
        if batch:
            v = self._emit_broadcast_static(v, dtype, (n, n), x_shape, lines, op)
        v_perm = [rank - 1, rank - 2] + list(range(rank - 2))
        v_t = self._new_name()
        lines.append(
            f'{v_t} = "stablehlo.transpose"({v}) '
            f"{{permutation = {self._i64_array(v_perm)}}} : "
            f"({self._type_str(dtype, x_shape)}) -> "
            f"{self._type_str(dtype, t_shape)}"
        )
        v = v_t
        # --- hoisted iotas: the slot index (m,) and the full index (n,) ---
        ti = self._emit_iota((m,), 0, lines)
        jj = self._emit_iota((n,), 0, lines)
        # --- hoisted i64 scalars (schedule arithmetic) --------------------
        i_zero, extra = self._scalar_constant_for(i64, 0, ())
        lines.extend(extra)
        i_one, extra = self._scalar_constant_for(i64, 1, ())
        lines.extend(extra)
        i_two, extra = self._scalar_constant_for(i64, 2, ())
        lines.extend(extra)
        i_l, extra = self._scalar_constant_for(i64, L, ())
        lines.extend(extra)
        i_two_l, extra = self._scalar_constant_for(i64, 2 * L, ())
        lines.extend(extra)
        #: The loop bound is the ITERATION count (``ceil(total / U)``); the
        #: unroll factor is itself a hoisted scalar so each body round can
        #: derive its global round index ``k*U + u`` without a per-round
        #: constant.
        i_iters, extra = self._scalar_constant_for(i64, n_iters, ())
        lines.extend(extra)
        i_unroll = None
        u_scalars = [i_zero]
        if U > 1:
            i_unroll, extra = self._scalar_constant_for(i64, U, ())
            lines.extend(extra)
            u_scalars.append(i_one)
            for u in range(2, U):
                i_u, extra = self._scalar_constant_for(i64, u, ())
                lines.extend(extra)
                u_scalars.append(i_u)
        # --- hoisted broadcasts of those scalars to (m,)/(n,) -------------
        l_m = self._bcast(i_l, i64, (), (m,), [], lines)
        l_n = self._bcast(i_l, i64, (), (n,), [], lines)
        two_n = self._bcast(i_two, i64, (), (n,), [], lines)
        two_l_n = self._bcast(i_two_l, i64, (), (n,), [], lines)
        zero_n = self._bcast(i_zero, i64, (), (n,), [], lines)
        zero_m = self._bcast(i_zero, i64, (), (m,), [], lines)
        f_m = f_n = None
        if even:
            i_fixed, extra = self._scalar_constant_for(i64, fixed, ())
            lines.extend(extra)
            f_m = self._bcast(i_fixed, i64, (), (m,), [], lines)
            f_n = self._bcast(i_fixed, i64, (), (n,), [], lines)
        # --- rotation-formula constants (per-SLOT and per-INDEX) ----------
        c_two, extra = self._scalar_constant_for(dtype, 2.0, batch)
        lines.extend(extra)
        c_one, extra = self._scalar_constant_for(dtype, 1.0, batch)
        lines.extend(extra)
        c_neg_one, extra = self._scalar_constant_for(dtype, -1.0, batch)
        lines.extend(extra)
        c_zero, extra = self._scalar_constant_for(dtype, 0.0, batch)
        lines.extend(extra)
        two_p = self._bcast(c_two, dtype, batch, pre_shape, batch_dims, lines)
        one_p = self._bcast(c_one, dtype, batch, pre_shape, batch_dims, lines)
        neg_one_p = self._bcast(c_neg_one, dtype, batch, pre_shape,
                                batch_dims, lines)
        zero_p = self._bcast(c_zero, dtype, batch, pre_shape, batch_dims, lines)
        # --- hoisted booleans (i1 scalars -> (n,) broadcasts) -------------
        b_true, extra = self._scalar_constant_for(i1, True, ())
        lines.extend(extra)
        b_false, extra = self._scalar_constant_for(i1, False, ())
        lines.extend(extra)
        b_true_n = self._bcast(b_true, i1, (), (n,), [], lines)
        b_false_n = self._bcast(b_false, i1, (), (n,), [], lines)
        # --- odd n: the STATIC per-SLOT bye mask (slot 0 is the bye) ------
        is_bye_b = None
        if not even:
            bye = self._new_name()
            lines.append(
                f"{bye} = stablehlo.constant "
                f"{self._constant_text(np.arange(m) == 0)} : "
                f"{self._type_str(i1, (m,))}"
            )
            is_bye_b = self._bcast(bye, i1, (m,), pre_shape, [0], lines)
        # --- eigh_early_exit: sweep-boundary convergence-check setup ------
        early_exit = self._eigh_early_exit
        m_neq = tol_sq = z_zero = None
        if early_exit:
            m_neq = self._new_name()
            lines.append(
                f"{m_neq} = stablehlo.constant "
                f"{self._constant_text(np.logical_not(np.eye(n, dtype=bool)).astype(dtype))}"
                f" : {self._type_str(dtype, (n, n))}"
            )
            if batch:  # broadcast to the carried (n, n, batch...) layout
                m_neq = self._bcast(m_neq, dtype, (n, n), t_shape, [0, 1], lines)
            # Same calibrated tolerances as the sequential composition
            # (see the tol comment in ``_emit_eigh_jacobi``).
            tol = 3e-5 if dtype == np.dtype("float32") else 1e-13
            tol_sq, extra = self._scalar_constant_for(dtype, tol * tol, batch)
            lines.extend(extra)
            z_zero, extra = self._scalar_constant_for(dtype, 0.0, ())
            lines.extend(extra)
        # --- the while loop: carries (a, v, k[, done]) --------------------
        a_type = self._type_str(dtype, t_shape)
        i64_scalar = self._elem_type(i64)
        i1_scalar = self._elem_type(i1)
        arg_types = [a_type, a_type, i64_scalar]
        if early_exit:
            arg_types.append(i1_scalar)
        carry_types = "(" + ", ".join(arg_types) + ")"
        n_carries = len(arg_types)
        k_index = 2
        cond_region = self._emit_eigh_cond_region(arg_types, i_iters, k_index)
        # --- the body: ``U`` consecutive ROUNDS of disjoint rotations -----
        body_args = [self._new_name() for _ in range(n_carries)]
        bl: list[str] = []
        prep = _RoundPrep(
            i64=i64, dtype=dtype, n=n, m=m, batch=batch, fixed=fixed,
            pre_shape=pre_shape, vec_shape=vec_shape, i_l=i_l, i_two=i_two,
            ti=ti, jj=jj, l_m=l_m, l_n=l_n, two_n=two_n, two_l_n=two_l_n,
            zero_n=zero_n, zero_m=zero_m, f_m=f_m, f_n=f_n,
            b_true_n=b_true_n, b_false_n=b_false_n, two_p=two_p,
            one_p=one_p, neg_one_p=neg_one_p, zero_p=zero_p,
            is_bye_b=is_bye_b,
        )
        a2, v2 = self._eigh_rounds(
            body_args[k_index], body_args[0], body_args[1], prep, bl,
            i_unroll=i_unroll, u_scalars=u_scalars,
        )
        # --- counter step (+ the early-exit k-clamp) ----------------------
        k2 = self._ew("add", body_args[k_index], i_one, i64, (), bl)
        if early_exit:
            # The sweep check counts ROUNDS, so the unrolled post-increment
            # iteration counter is scaled back to rounds before it.
            k_rounds = k2
            if i_unroll is not None:
                k_rounds = self._ew("multiply", k2, i_unroll, i64, (), bl)
            done2 = self._emit_eigh_sweep_check(
                bl, a2, body_args[k_index + 1], k_rounds, dtype, batch,
                t_shape, m_neq, tol_sq, z_zero, b_true, i_l, i_zero,
                i1_scalar,
            )
            k2 = self._sel(done2, i_iters, k2, i64, (), bl)
        ret_values = [a2, v2, k2]
        if early_exit:
            ret_values.append(done2)
        body_region = (
            "  ^bb0("
            + ", ".join(f"{nm} : {tp}" for nm, tp in zip(body_args, arg_types))
            + "):\n"
            + "\n".join("    " + line for line in bl)
            + f"\n    stablehlo.return {', '.join(ret_values)} : "
            + ", ".join(arg_types)
        )
        inits = [a, v, i_zero]
        if early_exit:
            inits.append(b_false)
        result_names = [self._new_name() for _ in range(n_carries)]
        lines.append(
            f"{', '.join(result_names)} = \"stablehlo.while\"("
            f"{', '.join(inits)}) ({{\n"
            + cond_region
            + "\n  }, {\n"
            + body_region
            + f"\n  }}) : {carry_types} -> {carry_types}"
        )
        a, v = result_names[0], result_names[1]
        self._emit_eigh_finish(op, a, v, dtype, x_shape, n, lines)
        return "\n".join(lines)

    def _eigh_rounds(self, k_name, a_name, v_name, prep, bl, *, i_unroll,
                     u_scalars):
        """Apply ``len(u_scalars)`` CONSECUTIVE schedule rounds inside the
        while body and return the resulting ``(a, v)`` names.

        Each round's GLOBAL index is ``k*U + u`` (``k`` the iteration
        counter), reduced mod ``L`` to the round number — so the sequence
        of applied rounds is the plain cyclic-Jacobi schedule
        ``0, 1, …``, padded at the very end by repeating its tail (see the
        module docstring). The round count is ``U`` and the loop bound
        ``ceil(total/U)``; both are hoisted scalars, so the body adds only
        one ``multiply`` + ``U-1`` ``add`` ops of round-index arithmetic on
        top of the per-round work.
        """
        if i_unroll is None:  # U == 1: the counter IS the round index
            base = k_name
        else:
            base = self._ew("multiply", k_name, i_unroll, prep.i64, (), bl)
        for u, i_u in enumerate(u_scalars):
            if u == 0:
                rnd = base
            else:
                rnd = self._ew("add", base, i_u, prep.i64, (), bl)
            r = self._ew("remainder", rnd, prep.i_l, prep.i64, (), bl)
            a_name, v_name = self._eigh_round(r, a_name, v_name, prep, bl)
        return a_name, v_name

    def _eigh_round(self, r, a_name, v_name, prep, bl) -> tuple[str, str]:
        """Emit ONE schedule round — the ``L``-modulus round index ``r`` is
        already reduced — and return the new ``(a, v)`` names.

        The round applies its ``m`` mutually-disjoint rotations at once
        (the per-slot (p, q) pair and the per-index (slot, partner, is_p)
        maps are derived arithmetically from the hoisted iotas), using the
        textbook ``tau = (a_qq - a_pp)/(2 a_pq)`` angle formulas: A takes
        the column then the row half-step, V the single dim-0 step (the
        carried V is ``V^T``). Every operand this needs is HOISTED — the
        only per-round inputs are ``r`` and the carried matrices.
        """
        n, m, batch, dtype = prep.n, prep.m, prep.batch, prep.dtype
        pre_shape, vec_shape = prep.pre_shape, prep.vec_shape
        i64, i1 = prep.i64, np.dtype("bool")
        p, q, slot, partner, is_p = self._eigh_rotation_maps(
            r, bl, i64=i64, m=m, n=n, fixed=prep.fixed, ti=prep.ti,
            jj=prep.jj, l_m=prep.l_m, l_n=prep.l_n, two_n=prep.two_n,
            two_l_n=prep.two_l_n, zero_n=prep.zero_n, zero_m=prep.zero_m,
            i_two=prep.i_two, f_m=prep.f_m, f_n=prep.f_n,
            b_true_n=prep.b_true_n, b_false_n=prep.b_false_n,
        )
        two_p, one_p = prep.two_p, prep.one_p
        neg_one_p, zero_p = prep.neg_one_p, prep.zero_p
        is_bye_b = prep.is_bye_b
        # --- per-SLOT rotation angles (identical formulas to the
        # sequential body, evaluated for the whole round at once) ---------
        idx_pq = self._eigh_index_pairs(p, q, m, bl)
        idx_pp = self._eigh_index_pairs(p, p, m, bl)
        idx_qq = self._eigh_index_pairs(q, q, m, bl)
        apq = self._eigh_gather_slots(a_name, idx_pq, dtype, n, m, batch, bl)
        app = self._eigh_gather_slots(a_name, idx_pp, dtype, n, m, batch, bl)
        aqq = self._eigh_gather_slots(a_name, idx_qq, dtype, n, m, batch, bl)
        t1 = self._ew("subtract", aqq, app, dtype, pre_shape, bl)
        t2 = self._ew("multiply", two_p, apq, dtype, pre_shape, bl)
        tau = self._ew("divide", t1, t2, dtype, pre_shape, bl)
        tau2 = self._ew("multiply", tau, tau, dtype, pre_shape, bl)
        oneptau2 = self._ew("add", one_p, tau2, dtype, pre_shape, bl)
        sq = self._ew("sqrt", oneptau2, None, dtype, pre_shape, bl)
        abst = self._ew("abs", tau, None, dtype, pre_shape, bl)
        denom = self._ew("add", abst, sq, dtype, pre_shape, bl)
        ge0 = self._cmp(tau, zero_p, "GE", dtype, pre_shape, bl)
        sgn = self._sel(ge0, one_p, neg_one_p, dtype, pre_shape, bl)
        t = self._ew("divide", sgn, denom, dtype, pre_shape, bl)
        t2b = self._ew("multiply", t, t, dtype, pre_shape, bl)
        onept2 = self._ew("add", one_p, t2b, dtype, pre_shape, bl)
        sqt = self._ew("sqrt", onept2, None, dtype, pre_shape, bl)
        c_calc = self._ew("divide", one_p, sqt, dtype, pre_shape, bl)
        s_calc = self._ew("multiply", t, c_calc, dtype, pre_shape, bl)
        apq_ne0 = self._cmp(apq, zero_p, "NE", dtype, pre_shape, bl)
        c = self._sel(apq_ne0, c_calc, one_p, dtype, pre_shape, bl)
        s = self._sel(apq_ne0, s_calc, zero_p, dtype, pre_shape, bl)
        if is_bye_b is not None:
            # Odd n: slot 0 is the BYE pseudo-rotation (p == q == r) —
            # force it to the identity AFTER the formulas (the formulas
            # would produce a 45-degree rotation, not a no-op).
            c = self._sel(is_bye_b, one_p, c, dtype, pre_shape, bl)
            s = self._sel(is_bye_b, zero_p, s, dtype, pre_shape, bl)
        # --- per-INDEX rotation coefficients ------------------------------
        idx_slot = self._eigh_index_column(slot, n, bl)
        cvec = self._eigh_gather_index(c, idx_slot, dtype, n, m, batch, bl)
        smag = self._eigh_gather_index(s, idx_slot, dtype, n, m, batch, bl)
        is_p_b = self._bcast(is_p, i1, (n,), vec_shape, [0], bl)
        sneg = self._ew("negate", smag, None, dtype, vec_shape, bl)
        svec = self._sel(is_p_b, sneg, smag, dtype, vec_shape, bl)
        # --- apply the round: A gets BOTH half-steps, V ONE dim-0 step ---
        idx_partner = self._eigh_index_column(partner, n, bl)
        mid = self._eigh_half_step(a_name, cvec, svec, idx_partner, 1, n,
                                  batch, dtype, bl)
        a2 = self._eigh_half_step(mid, cvec, svec, idx_partner, 0, n,
                                 batch, dtype, bl)
        v2 = self._eigh_half_step(v_name, cvec, svec, idx_partner, 0, n,
                                 batch, dtype, bl)
        return a2, v2

    def _eigh_rotation_maps(self, r, bl, *, i64, m, n, fixed, ti, jj,
                            l_m, l_n, two_n, two_l_n, zero_n, zero_m, i_two,
                            f_m, f_n, b_true_n, b_false_n):
        """The round-``r`` index maps, derived ARITHMETICALLY from hoisted
        iotas — no scatter, no per-element mask over the matrix.

        ``r`` is the ALREADY-REDUCED round index (``round mod L``, computed
        by the caller :meth:`_eigh_rounds` from the unrolled iteration
        counter). Per SLOT (``(m,)`` vectors):
        ``p``/``q`` are the rotation pair. Per INDEX (``(n,)`` vectors):
        ``slot`` (which slot's pair index ``jj`` belongs to), ``partner``
        (its partner vertex) and ``is_p`` (whether ``jj`` is the FIRST
        member of its pair — the ``s`` sign flip). Even ``n`` adds the FIXED
        vertex ``f = n-1``: ``f`` is always the ``p`` of slot 0 and the
        partner of ``r``. Every ``remainder`` numerator is provably
        NON-NEGATIVE (``r + ti <= 2L``, ``r - ti + L >= 0``, ``jj - r + L
        >= 0``, ``2r - jj + 2L >= 0``), so the emitted ``stablehlo.remainder``
        never depends on a sign convention. Returns ``(p, q, slot, partner,
        is_p)``.
        """
        r_m = self._bcast(r, i64, (), (m,), [], bl)
        r_n = self._bcast(r, i64, (), (n,), [], bl)
        # --- per-SLOT pair (p, q) -----------------------------------------
        sp = self._ew("add", r_m, ti, i64, (m,), bl)
        p_calc = self._ew("remainder", sp, l_m, i64, (m,), bl)
        sm = self._ew("subtract", r_m, ti, i64, (m,), bl)
        sm = self._ew("add", sm, l_m, i64, (m,), bl)
        q_calc = self._ew("remainder", sm, l_m, i64, (m,), bl)
        if fixed is not None:
            slot0 = self._cmp(ti, zero_m, "EQ", i64, (m,), bl)
            p = self._sel(slot0, f_m, p_calc, i64, (m,), bl)
            q = self._sel(slot0, r_m, q_calc, i64, (m,), bl)
        else:
            p, q = p_calc, q_calc
        # --- per-INDEX maps ------------------------------------------------
        d = self._ew("subtract", jj, r_n, i64, (n,), bl)
        d = self._ew("add", d, l_n, i64, (n,), bl)
        i1v = self._ew("remainder", d, l_n, i64, (n,), bl)
        l_minus = self._ew("subtract", l_n, i1v, i64, (n,), bl)
        slot = self._ew("minimum", i1v, l_minus, i64, (n,), bl)
        two_r_n = self._ew("multiply", r_n, two_n, i64, (n,), bl)
        pd = self._ew("subtract", two_r_n, jj, i64, (n,), bl)
        pd = self._ew("add", pd, two_l_n, i64, (n,), bl)
        partner = self._ew("remainder", pd, l_n, i64, (n,), bl)
        if fixed is not None:
            eqf = self._cmp(jj, f_n, "EQ", i64, (n,), bl)
            eqr = self._cmp(jj, r_n, "EQ", i64, (n,), bl)
            slot = self._sel(eqf, zero_n, slot, i64, (n,), bl)
            partner = self._sel(eqf, r_n, partner, i64, (n,), bl)
            partner = self._sel(eqr, f_n, partner, i64, (n,), bl)
        is_p = self._cmp(i1v, slot, "EQ", i64, (n,), bl)
        if fixed is not None:
            is_p = self._sel(eqf, b_true_n, is_p, np.dtype("bool"), (n,), bl)
            is_p = self._sel(eqr, b_false_n, is_p, np.dtype("bool"), (n,), bl)
        return p, q, slot, partner, is_p

    def _eigh_index_pairs(self, a_name, b_name, m, bl) -> str:
        """Two ``(m,)`` i64 vectors -> an ``(m, 2)`` i64 index tensor — the
        per-SLOT (row, col) gather index (``index_vector_dim = 1``, so the
        result keeps a leading ``(m,)`` axis)."""
        i64 = np.dtype("int64")
        cols = []
        for vec in (a_name, b_name):
            r = self._new_name()
            bl.append(
                f"{r} = stablehlo.reshape {vec} : "
                f"({self._type_str(i64, (m,))}) -> "
                f"{self._type_str(i64, (m, 1))}"
            )
            cols.append(r)
        idx = self._new_name()
        bl.append(
            f'{idx} = "stablehlo.concatenate"({cols[0]}, {cols[1]}) '
            f"{{dimension = 1 : i64}} : "
            f"({self._type_str(i64, (m, 1))}, {self._type_str(i64, (m, 1))})"
            f" -> {self._type_str(i64, (m, 2))}"
        )
        return idx

    def _eigh_index_column(self, vec_name, n, bl) -> str:
        """An ``(n,)`` i64 vector -> an ``(n, 1)`` index tensor
        (``index_vector_dim = 1``: one single-axis index per element, the
        result keeping a leading ``(n,)`` axis)."""
        i64 = np.dtype("int64")
        c = self._new_name()
        bl.append(
            f"{c} = stablehlo.reshape {vec_name} : "
            f"({self._type_str(i64, (n,))}) -> {self._type_str(i64, (n, 1))}"
        )
        return c

    def _eigh_gather_slots(self, src, idx_name, dtype, n, m, batch, bl) -> str:
        """``A[p, q, ...]`` for EVERY slot at once: an ``(m, 2)`` index
        tensor on the carried ``(n, n, batch...)`` matrix with BOTH matrix
        dims collapsed -> ``(m, batch...)`` (the ``_gather_scalar`` pattern,
        generalized to an index BATCH of size ``m``)."""
        i64 = np.dtype("int64")
        kb = len(batch)
        g = self._new_name()
        bl.append(
            f'{g} = "stablehlo.gather"({src}, {idx_name}) '
            f"{{dimension_numbers = #stablehlo.gather<offset_dims = "
            f"[{self._int_list(range(1, kb + 1))}], "
            f"collapsed_slice_dims = [0, 1], start_index_map = [0, 1], "
            f"index_vector_dim = 1>, "
            f"slice_sizes = {self._i64_array([1, 1] + list(batch))}}} : "
            f"({self._type_str(dtype, (n, n) + batch)}, "
            f"{self._type_str(i64, (m, 2))}) -> "
            f"{self._type_str(dtype, (m,) + batch)}"
        )
        return g

    def _eigh_gather_index(self, src, idx_name, dtype, n, m, batch, bl) -> str:
        """Per-INDEX gather on a per-SLOT vector: an ``(n, 1)`` index tensor
        on the ``(m, batch...)`` operand with the slot axis collapsed ->
        ``(n, batch...)`` — i.e. ``vec[slot[j], ...]`` for every matrix index
        ``j`` (the ``cvec``/``svec`` construction; NO scatter)."""
        i64 = np.dtype("int64")
        kb = len(batch)
        g = self._new_name()
        bl.append(
            f'{g} = "stablehlo.gather"({src}, {idx_name}) '
            f"{{dimension_numbers = #stablehlo.gather<offset_dims = "
            f"[{self._int_list(range(1, kb + 1))}], "
            f"collapsed_slice_dims = [0], start_index_map = [0], "
            f"index_vector_dim = 1>, "
            f"slice_sizes = {self._i64_array([1] + list(batch))}}} : "
            f"({self._type_str(dtype, (m,) + batch)}, "
            f"{self._type_str(i64, (n, 1))}) -> "
            f"{self._type_str(dtype, (n,) + batch)}"
        )
        return g

    def _eigh_half_step(self, x_name, cvec, svec, idx_name, axis, n, batch,
                        dtype, bl) -> str:
        """ONE half-step ``X' = c*X + s*take(X, partner, axis)`` on the
        carried ``(n, n, batch...)`` matrix.

        ``axis = 1`` (the column update ``B = A*J``) and ``axis = 0`` (the
        row update ``A' = J^T*B``, also the V update — the carried V is
        ``V^T``) mirror the sequential composition exactly; ``take`` is a
        single-axis ``stablehlo.gather`` with an ``(n, 1)`` index
        (``index_vector_dim = 1``), so the gathered axis lands back on
        ``axis`` of the result. The per-index coefficient vector is
        broadcast so its n axis maps to ``axis`` too (the existing
        ``col_dims``/``row_dims`` convention)."""
        kb = len(batch)
        t_shape = (n, n) + batch
        vec_shape = (n,) + batch
        take = self._eigh_take(x_name, idx_name, axis, dtype, n, batch, bl)
        dims = ([1] + list(range(2, kb + 2))) if axis == 1 else \
               ([0] + list(range(2, kb + 2)))
        c_b = self._bcast(cvec, dtype, vec_shape, t_shape, dims, bl)
        s_b = self._bcast(svec, dtype, vec_shape, t_shape, dims, bl)
        m1 = self._ew("multiply", c_b, x_name, dtype, t_shape, bl)
        m2 = self._ew("multiply", s_b, take, dtype, t_shape, bl)
        return self._ew("add", m1, m2, dtype, t_shape, bl)

    def _eigh_take(self, src, idx_name, axis, dtype, n, batch, bl) -> str:
        """``take(X, partner, axis)`` on the carried ``(n, n, batch...)``
        matrix — a single-axis ``stablehlo.gather`` collapsing the gathered
        axis, that axis coming back AT position ``axis`` of the result.

        The gather always collapses the LEADING dimension of its operand:
        iree 3.11's llvm-cpu gather vectorization mis-compiles a gather whose
        ``collapsed_slice_dims`` is a non-leading axis (``'vector.transfer_read'
        op inferred mask type ('vector<4xi1>') and mask operand type
        ('vector<1x4xi1>') don't match``), so a non-zero ``axis`` moves that
        axis to the front with a ``stablehlo.transpose`` and moves it back
        afterwards — the same static-shape permutation the frontend ``gather``
        emitter uses for ``axis != 0``."""
        i64 = np.dtype("int64")
        kb = len(batch)
        rank = 2 + kb
        t_shape = (n, n) + batch
        gx_name, gx_shape = src, t_shape
        perm = None
        if axis != 0:
            perm = [axis] + [i for i in range(rank) if i != axis]
            gx_shape = (t_shape[axis],) + tuple(
                d for i, d in enumerate(t_shape) if i != axis
            )
            gx_name = self._new_name()
            bl.append(
                f'{gx_name} = "stablehlo.transpose"({src}) '
                f"{{permutation = {self._i64_array(perm)}}} : "
                f"({self._type_str(dtype, t_shape)}) -> "
                f"{self._type_str(dtype, gx_shape)}"
            )
        offset_dims = [1] + list(range(2, rank))
        slice_sizes = [1, n] + list(batch)
        g = self._new_name()
        bl.append(
            f'{g} = "stablehlo.gather"({gx_name}, {idx_name}) '
            f"{{dimension_numbers = #stablehlo.gather<offset_dims = "
            f"[{self._int_list(offset_dims)}], "
            f"collapsed_slice_dims = [0], "
            f"start_index_map = [0], "
            f"index_vector_dim = 1>, "
            f"slice_sizes = {self._i64_array(slice_sizes)}}} : "
            f"({self._type_str(dtype, gx_shape)}, "
            f"{self._type_str(i64, (n, 1))}) -> "
            f"{self._type_str(dtype, gx_shape)}"
        )
        if perm is None:
            return g
        # Inverse permutation back to the carried layout.
        inv = [0] * rank
        for pos, src_dim in enumerate(perm):
            inv[src_dim] = pos
        out = self._new_name()
        bl.append(
            f'{out} = "stablehlo.transpose"({g}) '
            f"{{permutation = {self._i64_array(inv)}}} : "
            f"({self._type_str(dtype, gx_shape)}) -> "
            f"{self._type_str(dtype, t_shape)}"
        )
        return out
