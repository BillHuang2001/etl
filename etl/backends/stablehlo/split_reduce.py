"""Two-stage ("split") reduction emission.

IREE's GPU backend lowers every ``stablehlo.reduce`` whose RESULT is rank-0
as ONE workgroup of ONE thread, serially reducing the whole extent — so a
rank-0 ``min``/``argmin`` over a 10001-element fitness vector costs ~0.7-0.9
ms while the entire 10000x100 data-parallel part of the same graph costs
~0.06 ms (measured on an AMD MI300X, iree-base 3.11.0, real evox DE step).

The fix is a **two-stage split**: reshape the reduced extent ``N`` into
``(R, C)``, reduce the INNER axis (producing ``R`` partials — a rank >= 1
result, which IREE parallelizes) and then reduce the partials. Padded slack
is filled with the reduction IDENTITY so it can never win. Measured at the
etl level: ``min`` over 10001 0.7608 -> 0.0957 ms (7.9x, exact 73x137
factorization) / 0.1067 ms (7.1x, generic padded path) and ``argmin`` over
10001 0.9386 -> 0.1162 ms (8.1x); 9x on iree/llvm-cpu as well. Reshaping to
keepdims ``(10001, 1)`` does NOT help — the intermediate result has to
actually be small, so the partials are reduced to the declared shape in a
second (now tiny) reduce.

Eligibility (binding; everything else keeps the single-stage text
BYTE-FOR-BYTE):

1. the ``split_reduction`` exporter option is True (default);
2. ALL dims of the operand shape are static ints (any ``Dim``/``DimExpr``/
   ``None`` keeps today's emission — no error);
3. the reduced dims are non-empty;
4. ``total_reduced`` = product of the reduced extents >=
   ``split_reduction_threshold`` (default 1024);
5. ``result_elems`` = product of the NON-reduced extents <= 1024 (rank-0 or
   otherwise tiny results);
6. the reduction's **exactness class** allows it:

   - ``min`` / ``max`` — including the i1 ``any``/``all`` guard shapes
     (bool ``reduce_max``/``reduce_min``, emitted as ``stablehlo.maximum``/
     ``minimum`` on i1) — split BY DEFAULT (extrema are associative and the
     identity padding never wins);
   - ``argmin`` / ``argmax`` — split BY DEFAULT: the composite
     ``(value, index)`` reduce makes ties exact (first occurrence, matching
     ``np.argmin``/``np.argmax``);
   - ``sum`` / ``prod`` over INTEGER/bool dtypes — split BY DEFAULT
     (wraparound integer arithmetic is associative, so the result is exact);
   - ``sum`` / ``prod`` over FLOAT dtypes — and therefore the ``reduce_mean``
     sum and ``var`` (a frontend composition over the mean) — are NOT split
     by default: splitting reassociates floating-point accumulation, so the
     result is not bit-exact vs the single-stage text. They split only when
     the ``split_reduction_non_exact`` exporter option is set True (measured
     gain is in ``stablehlo/CONTEXT.md``).

Both stages keep the caller's dtype, promotion ``convert``, ``keepdims``
reshape, ``reduce_mean`` divide and the arg-reduce bool->int8 key
conversion — only the reduce itself is replaced.

``argmin``/``argmax`` carry their index through BOTH stages with the SAME
comparator region; the index operand becomes ``iota(dim=pivot) * C +
iota(dim=pivot+1)`` (int64, full-rank over the reshaped shape), which is
exactly the ORIGINAL flat index along the reduced axis for real elements
(``r * C + c == n``) and lands in ``[N, R*C)`` — strictly larger than every
real index — for padded slots, so the ``idx LT`` tie-break can never let a
pad win.

This module imports only ``etl.core`` (+ numpy/stdlib). It deliberately does
NOT import ``writer`` — ``Writer`` inherits :class:`SplitReductionMixin`, so
every helper below reaches the emission primitives through ``self``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from etl.core import BackendError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from etl.ir import Op, Value

#: Divisors of the reduced extent are searched up to this bound (an exact
#: factorization with no padding is preferred over a padded one).
SPLIT_FACTOR_LIMIT = 1024
#: Clamp for the power-of-two ``R`` of the generic (padded) path.
SPLIT_ROWS_MIN = 16
SPLIT_ROWS_MAX = 256
#: A split only pays off when the stage-2 partial count (and the remaining
#: non-reduced work) is small: the result must have at most this many
#: elements (rank-0 included).
SPLIT_RESULT_ELEMS_LIMIT = 1024


def _is_static_extent(dim) -> bool:
    """True for a concrete int dim (``Dim``/``DimExpr``/``None`` are not)."""
    return isinstance(dim, (int, np.integer)) and not isinstance(dim, bool)


def normalize_split_reduction(value) -> bool:
    """Validate the ``split_reduction`` exporter option (bool, default
    ``True``): ``False`` emits the single-stage ``stablehlo.reduce`` text
    EXACTLY as before the option existed."""
    if value is None:
        return True
    if not isinstance(value, (bool, np.bool_)):
        raise BackendError(
            f"stablehlo export: invalid split_reduction option {value!r} "
            "(expected a bool: True enables the two-stage split reduction, "
            "False emits the single-stage reduce text unchanged)"
        )
    return bool(value)


def normalize_split_reduction_threshold(value) -> int:
    """Validate the ``split_reduction_threshold`` exporter option (int >=
    2, default 1024): the minimum ``total_reduced`` extent to split."""
    if value is None:
        return 1024
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise BackendError(
            f"stablehlo export: invalid split_reduction_threshold option "
            f"{value!r} (expected an int >= 2 — the minimum reduced extent "
            "that is split into a two-stage reduction)"
        )
    threshold = int(value)
    if threshold < 2:
        raise BackendError(
            f"stablehlo export: invalid split_reduction_threshold option "
            f"{value!r} (expected an int >= 2 — the minimum reduced extent "
            "that is split into a two-stage reduction)"
        )
    return threshold


def normalize_split_reduction_non_exact(value) -> bool:
    """Validate the ``split_reduction_non_exact`` exporter option (bool,
    default ``False``): opt IN to splitting float ``sum``/``prod`` (and
    therefore ``reduce_mean``), which reassociates the floating-point
    accumulation — not bit-exact vs the single-stage text."""
    if value is None:
        return False
    if not isinstance(value, (bool, np.bool_)):
        raise BackendError(
            f"stablehlo export: invalid split_reduction_non_exact option "
            f"{value!r} (expected a bool: True also splits float sum/prod — "
            "and therefore reduce_mean — reassociating the float "
            "accumulation, so the result is not bit-exact)"
        )
    return bool(value)


def split_extent(n: int) -> tuple[int, int, int] | None:
    """Choose ``(R, C, pad)`` for a reduced extent ``n`` (``R * C = n +
    pad``), or ``None`` when no useful split exists.

    Preferred: an EXACT factorization with no padding — the divisor ``d`` of
    ``n`` in ``[2, min(1024, n)]`` closest to ``sqrt(n)`` becomes ``R``
    (``C = n // R``); ``C >= 2`` is required. Otherwise ``R`` is the power of
    two nearest ``sqrt(n)`` (bit-length based), clamped to ``[16, 256]``,
    and ``C = ceil(n / R)`` so ``pad = R * C - n < R``.
    """
    n = int(n)
    if n < 2:
        return None
    root = math.isqrt(n)
    best: tuple[int, int, int] | None = None
    for d in range(2, min(SPLIT_FACTOR_LIMIT, n) + 1):
        if n % d:
            continue
        c = n // d
        if c < 2:
            continue
        score = abs(d - root)
        if best is None or score < best[0]:
            best = (score, d, c)
    if best is not None:
        return best[1], best[2], 0
    lo = 1 << max(root.bit_length() - 1, 0)
    hi = lo * 2
    r = lo if abs(lo - root) <= abs(hi - root) else hi
    r = max(SPLIT_ROWS_MIN, min(SPLIT_ROWS_MAX, r))
    c = -(-n // r)
    if c < 2:
        return None
    return r, c, r * c - n


def _simple_reduce_region(writer, elem_type: str, mnemonic: str) -> str:
    """Reducer region for a plain reduce stage (same shape as the writer's
    ``_emit_reduce_core`` region: fresh counter-named block args)."""
    body = writer._new_name()
    arg_a = writer._new_name()
    arg_b = writer._new_name()
    return (
        "({\n"
        f"  ^bb0({arg_a}: {elem_type}, {arg_b}: {elem_type}):\n"
        f"    {body} = stablehlo.{mnemonic} {arg_a}, {arg_b} : {elem_type}\n"
        f"    stablehlo.return {body} : {elem_type}\n"
        "  })"
    )


def _arg_reduce_region(writer, key_elem: str, direction: str) -> str:
    """Comparator region for the ``(value, index)`` arg-reduce stages:
    ``(value GT|LT rhs) OR (value EQ rhs AND idx LT rhs_idx)`` — first
    occurrence wins on ties (matching ``np.argmax``/``np.argmin``). Two
    operands GROUP their reduce-block args (operand0_lhs, operand1_lhs,
    operand0_rhs, operand1_rhs)."""
    a, b, c, d = (writer._new_name() for _ in range(4))
    c1, c2, c3, c4, c5 = (writer._new_name() for _ in range(5))
    sel_v, sel_i = writer._new_name(), writer._new_name()
    return (
        "({\n"
        f"  ^bb0({a}: {key_elem}, {b}: tensor<i64>, "
        f"{c}: {key_elem}, {d}: tensor<i64>):\n"
        f'    {c1} = "stablehlo.compare"({a}, {c}) '
        f"{{comparison_direction = #stablehlo<comparison_direction {direction}>}}"
        f" : ({key_elem}, {key_elem}) -> tensor<i1>\n"
        f'    {c2} = "stablehlo.compare"({a}, {c}) '
        f"{{comparison_direction = #stablehlo<comparison_direction EQ>}}"
        f" : ({key_elem}, {key_elem}) -> tensor<i1>\n"
        f'    {c3} = "stablehlo.compare"({b}, {d}) '
        f"{{comparison_direction = #stablehlo<comparison_direction LT>}}"
        f" : (tensor<i64>, tensor<i64>) -> tensor<i1>\n"
        f"    {c4} = stablehlo.and {c2}, {c3} : tensor<i1>\n"
        f"    {c5} = stablehlo.or {c1}, {c4} : tensor<i1>\n"
        f"    {sel_v} = stablehlo.select {c5}, {a}, {c} : "
        f"(tensor<i1>, {key_elem}, {key_elem}) -> {key_elem}\n"
        f"    {sel_i} = stablehlo.select {c5}, {b}, {d} : "
        f"(tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>\n"
        f"    stablehlo.return {sel_v}, {sel_i} : {key_elem}, tensor<i64>\n"
        "  })"
    )


def split_comment(pivot: int, n: int, r: int, c: int) -> str:
    """The marker comment emitted immediately before the stage-1 reduce (so
    export goldens can assert the split unambiguously)."""
    return (
        f"// etl split reduction: pivot axis {int(pivot)}, "
        f"{int(n)} -> {int(r)} x {int(c)}"
    )


class SplitReductionMixin:
    """Two-stage split reduction emission for :class:`writer.Writer` (see
    the module docstring for the rule, the exactness classification and the
    measured gains). Every helper reaches the emission primitives through
    ``self`` (the Writer instance)."""

    # -- eligibility ---------------------------------------------------

    def _split_reduce_plan(self, x: "Value", dims, kind: str, dtype) -> dict | None:
        """Decide whether a plain ``sum``/``max``/``min``/``prod`` reduce is
        split; returns the plan dict or ``None`` (today's text)."""
        if not self._split_reduction:
            return None
        shape = tuple(x.type.shape)
        dims = sorted(int(d) for d in dims)
        if not shape or not dims:
            return None
        if not all(_is_static_extent(d) for d in shape):
            return None
        if dtype.kind == "c":
            return None  # unreachable — the callers reject complex first
        if (
            kind in ("sum", "prod")
            and dtype.kind == "f"
            and not self._split_reduction_non_exact
        ):
            return None
        reduced = set(dims)
        total_reduced = 1
        for i in dims:
            total_reduced *= int(shape[i])
        result_elems = 1
        for i, d in enumerate(shape):
            if i not in reduced:
                result_elems *= int(d)
        if total_reduced < self._split_reduction_threshold:
            return None
        if result_elems > SPLIT_RESULT_ELEMS_LIMIT:
            return None
        pivot = max(dims, key=lambda i: (int(shape[i]), -i))
        n = int(shape[pivot])
        extent = split_extent(n)
        if extent is None:
            return None
        r, c, pad = extent
        return {
            "shape": shape,
            "dims": dims,
            "pivot": pivot,
            "n": n,
            "r": r,
            "c": c,
            "pad": pad,
        }

    # -- primitives ----------------------------------------------------

    def _emit_split_pad(
        self, operand: str, dtype, shape: tuple, pivot: int, pad: int, kind: str,
        lines: list,
    ) -> tuple[str, tuple]:
        """Pad ``pad`` reduction-identity elements onto the HIGH end of the
        ``pivot`` axis; returns ``(name, padded_shape)``."""
        rank = len(shape)
        padded_shape = tuple(
            int(d) + (pad if i == pivot else 0) for i, d in enumerate(shape)
        )
        pad_value, pad_lines = self._scalar_constant_for(
            dtype, self._reduce_init(kind, dtype), ()
        )
        lines.extend(pad_lines)
        low = [0] * rank
        high = [pad if i == pivot else 0 for i in range(rank)]
        padded = self._new_name()
        lines.append(
            f'{padded} = "stablehlo.pad"({operand}, {pad_value}) '
            f"{{edge_padding_low = {self._i64_array(low)}, "
            f"edge_padding_high = {self._i64_array(high)}, "
            f"interior_padding = {self._i64_array([0] * rank)}}} : "
            f"({self._type_str(dtype, shape)}, {self._elem_type(dtype)}) -> "
            f"{self._type_str(dtype, padded_shape)}"
        )
        return padded, padded_shape

    def _emit_split_reshape(
        self, operand: str, dtype, shape: tuple, pivot: int, r: int, c: int,
        lines: list,
    ) -> tuple[str, tuple]:
        """Reshape the ``pivot`` extent into ``(r, c)``; returns
        ``(name, reshaped_shape)``."""
        reshaped_shape = (
            tuple(int(d) for d in shape[:pivot])
            + (int(r), int(c))
            + tuple(int(d) for d in shape[pivot + 1:])
        )
        reshaped = self._new_name()
        lines.append(
            f"{reshaped} = stablehlo.reshape {operand} : "
            f"({self._type_str(dtype, shape)}) -> "
            f"{self._type_str(dtype, reshaped_shape)}"
        )
        return reshaped, reshaped_shape

    def _emit_split_reduce_stage(
        self, operand: str, dtype, in_shape: tuple, dims: list, kind: str,
        lines: list, comment: str | None = None,
    ) -> tuple[str, tuple]:
        """Emit ONE ``stablehlo.reduce`` stage over ``dims`` of a named
        intermediate (the plain kinds); returns ``(name, out_shape)``."""
        elem_type = self._elem_type(dtype)
        init_name = self._new_name()
        lines.append(
            f"{init_name} = stablehlo.constant "
            f"{self._constant_text(np.asarray(self._reduce_init(kind, dtype), dtype=dtype))}"
            f" : {elem_type}"
        )
        if comment is not None:
            lines.append(comment)
        out_shape = tuple(
            d for i, d in enumerate(in_shape) if i not in set(dims)
        )
        result_name = self._new_name()
        region = _simple_reduce_region(
            self, elem_type, _reduce_body_mnemonic(kind)
        )
        lines.append(
            f'{result_name} = "stablehlo.reduce"({operand}, {init_name}) '
            f"{region} {{dimensions = {self._i64_array(dims)}}} : "
            f"({self._type_str(dtype, in_shape)}, {elem_type}) -> "
            f"{self._type_str(dtype, out_shape)}"
        )
        return result_name, out_shape

    # -- the hooked entry points ---------------------------------------

    def _reduce_core_with_split(self, x: "Value", dims, kind: str):
        """Emit a plain reduce — two-stage when eligible, else the existing
        single-stage ``_emit_reduce_core`` (byte-identical). Returns
        ``(lines, result_name, reduced_shape)``."""
        dtype = np.dtype(x.type.dtype)
        plan = self._split_reduce_plan(x, dims, kind, dtype)
        if plan is None:
            return self._emit_reduce_core(x, dims, kind)
        return self._emit_split_reduce_core(x, plan, kind, dtype)

    def _emit_split_reduce_core(self, x: "Value", plan: dict, kind: str, dtype):
        """The two-stage ``sum``/``max``/``min``/``prod`` emission (see the
        module docstring for the ``(R, C)`` choice and the padding rule)."""
        shape = plan["shape"]
        pivot, n, r, c, pad = (
            plan["pivot"], plan["n"], plan["r"], plan["c"], plan["pad"],
        )
        lines: list[str] = []
        operand = self._name(x)
        in_shape = shape
        if pad:
            operand, in_shape = self._emit_split_pad(
                operand, dtype, in_shape, pivot, pad, kind, lines
            )
        operand, in_shape = self._emit_split_reshape(
            operand, dtype, in_shape, pivot, r, c, lines
        )
        # Stage 1 reduces the C axis (+ every other reduced dim, shifted by
        # the extra split axis) leaving R partials; stage 2 reduces the R
        # axis. Both use the reduction identity as init.
        stage1_dims = sorted(
            {pivot + 1}
            | {int(d) + 1 for d in plan["dims"] if int(d) > pivot}
            | {int(d) for d in plan["dims"] if int(d) < pivot}
        )
        partials, partial_shape = self._emit_split_reduce_stage(
            operand, dtype, in_shape, stage1_dims, kind, lines,
            comment=split_comment(pivot, n, r, c),
        )
        # The R axis keeps the ORIGINAL position of the pivot dim once the
        # other reduced dims are gone (dims before the pivot stay in place,
        # the C axis is consumed by stage 1).
        kept = [i for i in range(len(in_shape)) if i not in set(stage1_dims)]
        r_axis = kept.index(pivot)
        result_name, reduced_shape = self._emit_split_reduce_stage(
            partials, dtype, partial_shape, [r_axis], kind, lines
        )
        return lines, result_name, reduced_shape

    def _split_arg_reduce_plan(self, op: "Op") -> dict | None:
        """Eligibility + geometry for a two-stage ``argmax``/``argmin``.

        PURE (no emission): the writer's SSA name counter must not be
        consumed when an op keeps the single-stage text. The reduced axis is
        the single ``axis`` attribute (``None`` flattens first, like numpy
        and the single-stage emitter), so the ``(R, C)`` split applies to
        that one extent.
        """
        if not self._split_reduction:
            return None
        x = op.operands[0]
        x_shape = tuple(x.type.shape)
        if not all(_is_static_extent(d) for d in x_shape):
            return None
        axis_attr = op.attributes.get("axis")
        if axis_attr is None:
            flat = (int(np.prod(x_shape)),)
            red_shape_in = flat
            axis = 0
        else:
            flat = None
            axis = self._normalize_axis(axis_attr, x.type.rank, f"{op.name}.axis")
            red_shape_in = x_shape
        n = int(red_shape_in[axis])
        if n < self._split_reduction_threshold:
            return None
        result_elems = 1
        for i, d in enumerate(red_shape_in):
            if i != axis:
                result_elems *= int(d)
        if result_elems > SPLIT_RESULT_ELEMS_LIMIT:
            return None
        extent = split_extent(n)
        if extent is None:
            return None
        r, c, pad = extent
        return {
            "x": x,
            "kind": "max" if op.name == "argmax" else "min",
            "axis": axis,
            "axis_none": axis_attr is None,
            "flat": flat,
            "red_shape_in": red_shape_in,
            "keys_dtype": np.dtype(x.type.dtype),
            "n": n,
            "r": r,
            "c": c,
            "pad": pad,
        }

    def _emit_split_arg_reduce(self, op: "Op") -> str | None:
        """Two-stage ``argmax``/``argmin`` emission, or ``None`` when the op
        keeps the single-stage text (see the module docstring)."""
        plan = self._split_arg_reduce_plan(op)
        if plan is None:
            return None
        x = plan["x"]
        kind = plan["kind"]
        axis = plan["axis"]
        keys_dtype = plan["keys_dtype"]
        r, c, pad = plan["r"], plan["c"], plan["pad"]
        n = plan["n"]
        keepdims = bool(op.attributes.get("keepdims", False))
        lines: list[str] = []
        if plan["axis_none"]:
            flat = plan["flat"]
            xf = self._new_name()
            lines.append(
                f"{xf} = stablehlo.reshape {self._name(x)} : "
                f"({self._vt(x.type)}) -> {self._type_str(x.type.dtype, flat)}"
            )
            red_shape_in = flat
        else:
            xf = self._name(x)
            red_shape_in = plan["red_shape_in"]
        key_name = xf
        if keys_dtype.kind == "b":
            t = self._new_name()
            lines.append(
                f"{t} = stablehlo.convert {key_name} : "
                f"({self._type_str(keys_dtype, red_shape_in)}) -> "
                f"{self._type_str(np.dtype('int8'), red_shape_in)}"
            )
            key_name, keys_dtype = t, np.dtype("int8")
        in_shape = red_shape_in
        if pad:
            key_name, in_shape = self._emit_split_pad(
                key_name, keys_dtype, in_shape, axis, pad, kind, lines
            )
        key_name, in_shape = self._emit_split_reshape(
            key_name, keys_dtype, in_shape, axis, r, c, lines
        )
        # The composite index iota(pivot) * C + iota(pivot+1) carries the
        # ORIGINAL flat index along the reduced axis for real elements and a
        # value >= N for padded slots (so a pad can never win a tie).
        i64 = np.dtype("int64")
        row_iota = self._emit_iota(in_shape, axis, lines)
        col_iota = self._emit_iota(in_shape, axis + 1, lines)
        c_name, c_lines = self._scalar_constant_for(i64, c, in_shape)
        lines.extend(c_lines)
        scaled = self._new_name()
        lines.append(
            f"{scaled} = stablehlo.multiply {row_iota}, {c_name} : "
            f"{self._type_str(i64, in_shape)}"
        )
        index_name = self._new_name()
        lines.append(
            f"{index_name} = stablehlo.add {scaled}, {col_iota} : "
            f"{self._type_str(i64, in_shape)}"
        )
        key_elem = self._elem_type(keys_dtype)
        value_init, value_lines = self._scalar_constant_for(
            keys_dtype, self._reduce_init(kind, keys_dtype), ()
        )
        lines.extend(value_lines)
        index_init = self._new_name()
        lines.append(
            f"{index_init} = stablehlo.constant "
            f"{self._constant_text(np.asarray(0, dtype=i64))} : tensor<i64>"
        )
        region = _arg_reduce_region(
            self, key_elem, "GT" if kind == "max" else "LT"
        )
        partial_shape = tuple(
            d for i, d in enumerate(in_shape) if i != axis + 1
        )
        partial_keys, partial_idx = self._new_name(), self._new_name()
        lines.append(
            split_comment(axis, n, r, c)
        )
        lines.append(
            f'{partial_keys}, {partial_idx} = "stablehlo.reduce"'
            f"({key_name}, {index_name}, {value_init}, {index_init}) "
            f"{region} {{dimensions = {self._i64_array([axis + 1])}}} : "
            f"({self._type_str(keys_dtype, in_shape)}, "
            f"{self._type_str(i64, in_shape)}, {key_elem}, tensor<i64>) -> "
            f"({self._type_str(keys_dtype, partial_shape)}, "
            f"{self._type_str(i64, partial_shape)})"
        )
        red_shape_out = tuple(
            d for i, d in enumerate(partial_shape) if i != axis
        )
        fin_keys, fin_idx = self._new_name(), self._new_name()
        lines.append(
            f'{fin_keys}, {fin_idx} = "stablehlo.reduce"'
            f"({partial_keys}, {partial_idx}, {value_init}, {index_init}) "
            f"{region} {{dimensions = {self._i64_array([axis])}}} : "
            f"({self._type_str(keys_dtype, partial_shape)}, "
            f"{self._type_str(i64, partial_shape)}, {key_elem}, tensor<i64>) -> "
            f"({self._type_str(keys_dtype, red_shape_out)}, "
            f"{self._type_str(i64, red_shape_out)})"
        )
        cur = fin_idx
        if keepdims and tuple(op.result.type.shape) != tuple(red_shape_out):
            t = self._new_name()
            lines.append(
                f"{t} = stablehlo.reshape {cur} : "
                f"({self._type_str(i64, red_shape_out)}) -> "
                f"{self._vt(op.result.type)}"
            )
            cur = t
        self._names[id(op.result)] = cur
        return "\n".join(lines)


def _reduce_body_mnemonic(kind: str) -> str:
    """Reducer-region body mnemonic per etl reduction kind — the writer's
    canonical ``_REDUCE_BODY_MNEMONIC`` table, imported LAZILY (a
    module-level import would be a cycle: ``writer`` imports this module)."""
    from .writer import _REDUCE_BODY_MNEMONIC

    try:
        return _REDUCE_BODY_MNEMONIC[kind]
    except KeyError:
        raise BackendError(
            f"stablehlo export: unknown reduction kind {kind!r}"
        ) from None
