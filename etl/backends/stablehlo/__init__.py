"""etl.backends.stablehlo — StableHLO MLIR export utility (v1: export-only).

Produces StableHLO MLIR text from verified EvoXIR so external compilers can
take over compilation, e.g. `iree-compile model.mlir -o model.vmfb`. This
module is NOT a `Backend`: it is not registered with the backend registry
and it never lowers, compiles, or runs anything. The compiler adapters that
CONSUME this export (`"iree"`, `"xla"`, `"tvm"`) are implemented in
`../adapters/`.

See `../../CONTEXT.md` (backends contract; "StableHLO exporter — v1 scope"
is binding) and `../../../CONTEXT.md` (root design principles).
"""

from __future__ import annotations

from etl.ir import Module, verify  # allowed: top-level imports are core/ir only

__all__ = ["export"]


def export(graph_or_module, options: dict | None = None) -> str:
    """Export a graph or module as StableHLO MLIR text.

    Args:
        graph_or_module: a `trace.Graph` (duck-typed — its `.module`
            attribute is used; no `etl.trace` import is needed here) or an
            `etl.ir.Module` directly.
        options: optional dict of exporter options. ``rng_bit_generator``
            accepts a bool (backward compat: ``True`` → both ciphers,
            ``False``/absent → none) or a collection of random algorithm
            names — the native ``stablehlo.rng_bit_generator`` emission is
            used per-algorithm iff the name is in the set (canonical names:
            ``"threefry2x32"``, ``"philox4x32_10"``; splitmix64 has no
            native form — always the inline expansion). Algorithms outside
            the set are expanded as bit-exact inline i32/ui32 elementwise
            subgraphs, so bare exports always work. The compiler adapters
            pass their ``Capabilities.rng_bit_generator`` set through here
            from ``lower()`` (overridable per call via the reserved
            ``rng_bit_generator`` lower option). ``sort_emission`` selects
            the argsort emission: ``"pair"`` (default; the two-operand
            (key, iota) ``stablehlo.sort``), ``"count"`` (the count-based
            O(n^2) composition with NO sort op — bit-exact vs numpy, used
            on iree cuda where multi-operand sorts at sorted-axis extent
            >= 32 cannot be bufferized), or ``"auto"`` (per argsort: count
            when the sorted-axis extent >= 32, else pair). The iree
            adapter defaults to ``"auto"`` (see
            ``CompilerBackend.default_sort_emission``). ``while_init_rewrite``
            (bool, default True) rewrites all-zero rank>=1 constant while
            INIT operands into computed zeros (the iree 3.11.0
            AffinityAnalysis SEGV workaround — see the Writer).
            ``eigh_early_exit`` (bool, default True) enables the ``eigh``
            while-Jacobi composition's convergence-based early exit: the
            loop additionally carries an i1 ``done`` flag and, at every
            sweep boundary (inside a nested ``stablehlo.if``), checks the
            scale-aware relative off-diagonal energy of the current A
            against a calibrated dtype tolerance (f32 tol 3e-5 — commit
            09e145d, fires sweep ~4 of 7 on dim-45/50 sample-covariance-
            like matrices; f64 tol 1e-13 — calibration home
            ``tests/backends/test_iree_eigh_diag_parity.py``), exiting
            once converged so converged matrices skip their remaining
            scheduled sweeps (see the Writer's ``_emit_eigh`` /
            ``_emit_eigh_sweep_check``). ``False`` emits the exact
            pre-option 5-carry text — the A/B measurement lever and
            safety valve.

    Returns:
        The StableHLO MLIR text (a ``str``) — compiler input for external
        tools, e.g. ``iree-compile model.mlir -o model.vmfb``.

    Behavior (binding):
        1. Unwrap: if the argument has a ``.module`` attribute
           (`trace.Graph`), use it; accept `etl.ir.Module` directly;
           otherwise raise ``TypeError``.
        2. Verify FIRST: call ``module.verify()`` — verification failures
           surface as `core.VerificationError`; never emit MLIR from
           invalid IR.
        3. Emit via ``writer.Writer``: functions in declaration order, ops
           in block order, symbolic dims as ``?``, constants as
           ``stablehlo.constant`` dense elements (see ``./writer.py``).
        4. Unsupported or deferred op ⇒ `core.BackendError` NAMING THE OP,
           with a message suggesting decomposition or a future adapter.
           Never silently skip an op.
        5. Export-only: this module is NOT registered as a Backend — it is
           intentionally absent from the backend registry and performs no
           lower/compile/load/run.

    Raises:
        TypeError: if ``graph_or_module`` is neither Graph-like nor an
            `etl.ir.Module`.
        core.VerificationError: if the IR fails verification.
        core.BackendError: if any op is unsupported/deferred in v1 (names
            the op).
    """
    module = getattr(graph_or_module, "module", graph_or_module)
    if not isinstance(module, Module):
        raise TypeError(
            "etl.backends.stablehlo.export: expected a trace.Graph or an "
            f"etl.ir.Module, got {type(graph_or_module).__name__}"
        )
    verify(module)
    from .writer import Writer

    return Writer(module, options=options).write()
