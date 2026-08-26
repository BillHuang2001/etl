"""StableHLO MLIR text emission.

The Writer walks a verified `etl.ir.Module` and emits StableHLO MLIR text.
Mapping data lives in `./ops.py`; this module contains NO mapping tables —
it only consumes them. Import rules (binding): top-level imports restricted
to `etl.core`/`etl.ir` (+ stdlib and `./ops.py`); `etl.pipeline` never
imported. The Writer receives an already-unwrapped `etl.ir.Module` from
`export()` (Graph handling stays in `__init__.py`).

Emission conventions (syntax verified against the StableHLO spec at
https://openxla.org/stablehlo/spec and stablehlo/dialect/StablehloOps.td):

- Ops **without attributes/regions** print in the standard MLIR custom-op
  form: ``%5 = stablehlo.add %2, %3 : tensor<3x4xf32>``. Ops **with**
  attributes and/or regions print in the generic form (the form the spec
  itself uses in its examples), e.g. ``%3 = "stablehlo.compare"(%0, %1)
  {comparison_direction = #stablehlo<comparison_direction EQ>} : (...)
  -> tensor<...>`` — guaranteed to parse for any attribute set.
- **Structure**: ``module { ... }`` wrapping one ``func.func`` per module
  function (args/results via `mlir_type`, body ops in program order —
  program order IS effect order — ending in ``func.return``).
- **SSA names**: op results ``%0``, ``%1``, ... sequentially (module-wide
  counter); function args ``%argN``; while-region args continue past the
  entry args (so enclosing values referenced inside regions are never
  shadowed). Decompositions introduce intermediates from the same counter.
- **Dispatch** on the op name via `mapping.status`: "v1" emits the mapped
  mnemonic; "decompose" emits `mapping.DECOMPOSITIONS` sub-ops; "deferred"
  or unknown raises `core.BackendError` NAMING THE OP (never silently
  skips an op, never partial output).
- **Symbolic dims**: ints render literally; `Dim`/`DimExpr`/`None` render
  as `?` (rank is always concrete in etl). Constants render via
  `render_constant` as ``stablehlo.constant`` + ``dense<[...]>``.
- **Control flow**: IR op names are ``if``/``while`` (trace lowers
  cond/while_loop to them). `if` branches emit with no block args (etl
  entry args bind to the op operands — StableHLO branches capture
  enclosing values implicitly); `while` cond/body regions emit fresh
  ``%argN`` block args bound to the loop-carried operands.
- **Collectives**: `replica_groups` built from the op's `group_size` attr
  (None — the world group — defaults to a single rank, matching v1
  single-process simulation). `broadcast_collective` maps to
  ``stablehlo.collective_broadcast`` — no effect-kind disambiguation
  needed (the shape op keeps the name `broadcast`).
"""

from __future__ import annotations

from itertools import count
from typing import TYPE_CHECKING

import numpy as np

from etl.core import BackendError, Dim, DimExpr

from . import ops as mapping

if TYPE_CHECKING:
    from etl.ir import Block, Function, Module, Op, Value

# Hex float spellings of IEEE special values (valid MLIR float literals;
# keyed by bit width). Used for inf/nan dense elements and extreme reduce
# init values.
_HEX_NAN = {16: "0x7E00", 32: "0x7FC00000", 64: "0x7FF8000000000000"}
_HEX_INF = {16: "0x7C00", 32: "0x7F800000", 64: "0x7FF0000000000000"}
_HEX_NINF = {16: "0xFC00", 32: "0xFF800000", 64: "0xFFF0000000000000"}

#: Reducer-region body mnemonic per etl reduction kind.
_REDUCE_BODY_MNEMONIC = {
    "sum": "add",
    "max": "maximum",
    "min": "minimum",
    "prod": "multiply",
}


def _fmt_float(value: float, bits: int = 64) -> str:
    """Format a float as a valid MLIR float literal (mandatory decimal
    point; inf/nan via IEEE hex spelling for the given bit width)."""
    value = float(value)
    if np.isnan(value):
        return _HEX_NAN[bits]
    if np.isinf(value):
        return _HEX_NINF[bits] if value < 0 else _HEX_INF[bits]
    text = repr(value)
    if "e" in text or "E" in text:
        mantissa, exponent = _re_split_exp(text)
        if "." not in mantissa:
            mantissa += ".0"
        return f"{mantissa}E{exponent}"
    if "." not in text:
        text += ".0"
    return text


def _re_split_exp(text: str):
    """Split a repr-format float into (mantissa, exponent) at e/E."""
    for i, ch in enumerate(text):
        if ch in "eE":
            return text[:i], text[i + 1 :]
    return text, ""


class Writer:
    """Emit StableHLO MLIR text for a verified `etl.ir.Module`; the
    module is assumed verified (``export()`` runs ``module.verify()``
    first), the Writer only type-checks it defensively."""

    def __init__(self, module: Module) -> None:
        """Store the verified module and initialize per-write state.

        State: SSA name counter (module-wide), per-value name map
        (``id(Value) -> "%N"``).
        Raises TypeError if `module` is not an `etl.ir.Module` (export()
        pre-validates; this is a defensive check only).
        """
        from etl.ir import Module

        if not isinstance(module, Module):
            raise TypeError(
                f"Writer requires an etl.ir.Module, got {type(module).__name__}"
            )
        self.module = module
        self._counter = count()
        self._names: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self) -> str:
        """Return the complete MLIR text: `module { ... }` wrapping each
        function (via `write_function`) in declaration order."""
        lines = ["module {"]
        for fn in self.module.functions:
            lines.append(self._indent(self.write_function(fn), 1))
        lines.append("}")
        return "\n".join(lines) + "\n"

    def write_function(self, fn: Function) -> str:
        """Emit one `func.func` definition: name + tensor-typed args/
        results, then the entry block's ops in program order and a
        `func.return` of the terminator's operands."""
        entry = fn.entry_block
        for i, arg in enumerate(entry.arguments):
            self._names[id(arg)] = f"%arg{i}"
        # While-region block args are named %argN continuing past the
        # entry args so they never shadow function parameters referenced
        # from inside the regions (e.g. a cond capturing `n`).
        self._arg_base = len(entry.arguments)
        args = ", ".join(
            f"%arg{i}: {self._vt(arg.type)}"
            for i, arg in enumerate(entry.arguments)
        )
        outputs = fn.output_types
        if not outputs:
            results = ""
        elif len(outputs) == 1:
            results = f" -> {self._vt(outputs[0])}"
        else:
            results = " -> (" + ", ".join(self._vt(t) for t in outputs) + ")"
        body = self._block_body_lines(entry, terminator="func.return")
        text = "\n".join(body)
        indented = self._indent(text, 1) if text else ""
        return f"func.func @{fn.name}({args}){results} {{\n{indented}\n}}"

    def write_op(self, op: Op) -> str:
        """Dispatch one op to its StableHLO emission (one string); see the
        module docstring for the dispatch/error conventions."""
        return self._emit_op(op)

    def mlir_type(self, dtype, shape) -> str:
        """Map (numpy dtype, etl shape) → StableHLO tensor type string:
        dtype via `mapping.mlir_dtype` (unknown ⇒ BackendError naming it);
        int dims literal; `Dim`/`DimExpr`/`None` → `?`."""
        try:
            m = mapping.mlir_dtype(dtype)
        except KeyError:
            raise BackendError(
                f"stablehlo export: dtype {dtype!r} has no StableHLO type "
                "mapping"
            ) from None
        dims = "x".join(self._dim_str(d) for d in shape)
        if dims:
            dims += "x"
        return f"tensor<{dims}{m}>"

    def render_attr(self, value) -> str:
        """Render a Python value as MLIR attribute syntax.

        Covers int/float/bool (bare literals — bools as `true`/`false`,
        floats MLIR-formatted), str (`"..."`), tuples/lists (`[a, b]`),
        numpy arrays (`dense<[...]>`), dicts (`k = v` pairs, sorted keys).
        Symbolic dims render as `?`. Op-specific typed layouts
        (`dense<[...]> : tensor<...xi64>` for axis lists, i64 integer
        attrs, etc.) are built by the op emitters directly.
        """
        if value is None:
            return "?"
        if isinstance(value, (bool, np.bool_)):
            return "true" if value else "false"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return _fmt_float(float(value))
        if isinstance(value, (complex, np.complexfloating)):
            return (
                f"({_fmt_float(float(value.real))}, "
                f"{_fmt_float(float(value.imag))})"
            )
        if isinstance(value, str):
            return f'"{value}"'
        if isinstance(value, np.ndarray):
            return self._constant_text(value)
        if isinstance(value, (tuple, list)):
            return "[" + ", ".join(self.render_attr(v) for v in value) + "]"
        if isinstance(value, dict):
            pairs = (f"{k} = {self.render_attr(value[k])}" for k in sorted(value))
            return "{" + ", ".join(pairs) + "}"
        if isinstance(value, (Dim, DimExpr)):
            return "?"
        return str(value)

    def render_constant(self, value: Value) -> str:
        """Render a `constant` op's result as `stablehlo.constant` with a
        `dense<[...]>` elements attribute + result tensor type."""
        op = value.defining_op
        if op is None or op.name != "constant":
            raise TypeError(
                f"render_constant requires a constant op result, got {value!r}"
            )
        data = np.asarray(op.attributes["value"])
        name = self._bind_value(value)
        return (
            f"{name} = stablehlo.constant {self._constant_text(data)}"
            f" : {self._vt(value.type)}"
        )

    # --- Dispatch ---

    def _emit_op(self, op: Op) -> str:
        name = op.name
        if name == "return":
            return ""  # terminators are emitted by the block emitters
        status = mapping.status(name)
        if status == "deferred":
            raise self._unsupported(op)
        if status == "decompose":
            return self._emit_decomposed(op)
        if name in mapping.ELEMENTWISE_MAP:
            return self._emit_elementwise(op)
        if name == "select":
            return self._emit_select(op)
        if name in mapping.COMPARISON_MAP:
            return self._emit_compare(op)
        if name == "constant":
            return self.render_constant(op.result)
        if name in mapping.SHAPE_MAP:
            return self._emit_shape(op)
        if name == "if":
            return self._emit_if(op)
        if name == "while":
            return self._emit_while(op)
        if name in mapping.COLLECTIVE_MAP:
            return self._emit_collective(op)
        raise self._unsupported(op)

    def _emit_elementwise(self, op: Op) -> str:
        """Unary/binary elementwise ops (incl. cast) in custom print form."""
        mnemonic = mapping.lookup_mapping(op.name)
        result_name = self._bind_results(op)[0]
        operands = ", ".join(self._name(v) for v in op.operands)
        return (
            f"{result_name} = {mnemonic} {operands} : {self._vt(op.result.type)}"
        )

    def _emit_select(self, op: Op) -> str:
        result_name = self._bind_results(op)[0]
        operands = ", ".join(self._name(v) for v in op.operands)
        op_types = ", ".join(self._vt(v.type) for v in op.operands)
        return (
            f"{result_name} = stablehlo.select {operands} : "
            f"({op_types}) -> {self._vt(op.result.type)}"
        )

    def _emit_compare(self, op: Op) -> str:
        direction = mapping.lookup_mapping(op.name)
        result_name = self._bind_results(op)[0]
        operands = ", ".join(self._name(v) for v in op.operands)
        op_types = ", ".join(self._vt(v.type) for v in op.operands)
        return (
            f'{result_name} = "stablehlo.compare"({operands}) '
            f"{{comparison_direction = #stablehlo<comparison_direction "
            f"{direction}>}} : ({op_types}) -> {self._vt(op.result.type)}"
        )

    def _emit_shape(self, op: Op) -> str:
        name = op.name
        if name == "broadcast":
            return self._emit_broadcast(op)
        if name == "reshape":
            return self._emit_reshape(op)
        if name == "transpose":
            return self._emit_transpose(op)
        if name == "slice":
            return self._emit_slice(op)
        if name == "concatenate":
            return self._emit_concatenate(op)
        if name == "pad":
            return self._emit_pad(op)
        if name.startswith("reduce_"):
            return self._emit_reduce(op)
        if name == "dot":
            return self._emit_dot(op)
        if name == "conv":
            return self._emit_conv(op)
        raise self._unsupported(op)

    def _emit_broadcast(self, op: Op) -> str:
        x = op.operands[0]
        in_rank, out_rank = x.type.rank, op.result.type.rank
        rank_diff = out_rank - in_rank
        # numpy trailing alignment: operand dim i lives at result dim
        # i + rank_diff; leading result dims are implicit broadcast dims.
        dims = list(range(rank_diff, rank_diff + in_rank))
        result_name = self._bind_results(op)[0]
        return (
            f'{result_name} = "stablehlo.broadcast_in_dim"({self._name(x)}) '
            f"{{broadcast_dimensions = {self._dense_1d(dims)}}} : "
            f"({self._vt(x.type)}) -> {self._vt(op.result.type)}"
        )

    def _emit_reshape(self, op: Op) -> str:
        result_name = self._bind_results(op)[0]
        return (
            f"{result_name} = stablehlo.reshape {self._name(op.operands[0])}"
            f" : {self._vt(op.result.type)}"
        )

    def _emit_transpose(self, op: Op) -> str:
        x = op.operands[0]
        rank = x.type.rank
        permutation = op.attributes.get("permutation")
        if permutation is None:
            permutation = tuple(range(rank - 1, -1, -1))  # numpy None = reverse
        result_name = self._bind_results(op)[0]
        return (
            f'{result_name} = "stablehlo.transpose"({self._name(x)}) '
            f"{{permutation = {self._dense_1d(permutation)}}} : "
            f"({self._vt(x.type)}) -> {self._vt(op.result.type)}"
        )

    def _emit_slice(self, op: Op) -> str:
        x = op.operands[0]
        starts = tuple(op.attributes["start_indices"])
        limits = tuple(op.attributes["limit_indices"])
        strides = tuple(op.attributes.get("strides") or (1,) * len(starts))
        result_name = self._bind_results(op)[0]
        return (
            f'{result_name} = "stablehlo.slice"({self._name(x)}) '
            f"{{start_indices = {self._dense_1d(starts)}, "
            f"limit_indices = {self._dense_1d(limits)}, "
            f"strides = {self._dense_1d(strides)}}} : "
            f"({self._vt(x.type)}) -> {self._vt(op.result.type)}"
        )

    def _emit_concatenate(self, op: Op) -> str:
        axis = op.attributes["axis"]
        result_name = self._bind_results(op)[0]
        operands = ", ".join(self._name(v) for v in op.operands)
        op_types = ", ".join(self._vt(v.type) for v in op.operands)
        return (
            f'{result_name} = "stablehlo.concatenate"({operands}) '
            f"{{dimension = {int(axis)} : i64}} : ({op_types}) -> "
            f"{self._vt(op.result.type)}"
        )

    def _emit_pad(self, op: Op) -> str:
        x = op.operands[0]
        pairs = [tuple(p) for p in op.attributes["padding_config"]]
        value = op.attributes.get("value", 0.0)
        dtype = np.dtype(x.type.dtype)
        rank = len(pairs)
        lines = []
        pad_value = self._new_name()
        fill = np.zeros((), dtype=dtype)
        if dtype.kind != "b":
            try:
                fill[()] = value
            except (ValueError, TypeError, OverflowError):
                raise BackendError(
                    f"stablehlo export: pad value {value!r} cannot be stored "
                    f"in the operand dtype {dtype}"
                ) from None
        else:
            fill[()] = bool(value)
        lines.append(
            f"{pad_value} = stablehlo.constant {self._constant_text(fill)}"
            f" : {self._elem_type(dtype)}"
        )
        low = self._dense_1d([p[0] for p in pairs])
        high = self._dense_1d([p[1] for p in pairs])
        interior = self._dense_1d([0] * rank)
        result_name = self._bind_results(op)[0]
        lines.append(
            f'{result_name} = "stablehlo.pad"({self._name(x)}, {pad_value}) '
            f"{{edge_padding_low = {low}, edge_padding_high = {high}, "
            f"interior_padding = {interior}}} : ({self._vt(x.type)}, "
            f"{self._elem_type(dtype)}) -> {self._vt(op.result.type)}"
        )
        return "\n".join(lines)

    # --- Reductions ---

    def _emit_reduce(self, op: Op) -> str:
        """reduce_sum/max/min/prod → `stablehlo.reduce` (+ init constant)."""
        x = op.operands[0]
        kind = op.attributes["reduce_op"]
        axes = tuple(op.attributes.get("axes", ()))
        keepdims = bool(op.attributes.get("keepdims", False))
        dims = self._reduce_dims(x, axes)
        elem_dtype = np.dtype(x.type.dtype)
        if elem_dtype.kind == "c":
            raise BackendError(
                f"stablehlo export: op '{op.name}'{self._loc(op)} over "
                "complex dtype is not supported in v1 — complex-number "
                "computation beyond cast is deferred"
            )
        lines, red_name, reduced_shape = self._emit_reduce_core(x, dims, kind)
        cur = red_name
        # numpy promotion guard: stablehlo.reduce keeps the operand dtype;
        # etl sum/prod promote (ints → int64, bool → int64).
        if np.dtype(op.result.type.dtype) != elem_dtype:
            converted = self._new_name()
            lines.append(
                f"{converted} = stablehlo.convert {cur} : "
                f"{self._type_str(op.result.type.dtype, reduced_shape)}"
            )
            cur = converted
        if keepdims and tuple(op.result.type.shape) != tuple(reduced_shape):
            reshaped = self._new_name()
            lines.append(
                f"{reshaped} = stablehlo.reshape {cur} : "
                f"{self._vt(op.result.type)}"
            )
            cur = reshaped
        self._names[id(op.result)] = cur
        return "\n".join(lines)

    def _emit_reduce_mean(self, op: Op) -> str:
        """reduce_mean → reduce-sum, then divide by the (static) count."""
        x = op.operands[0]
        axes = tuple(op.attributes.get("axes", ()))
        keepdims = bool(op.attributes.get("keepdims", False))
        dims = self._reduce_dims(x, axes)
        count = 1
        for i in dims:
            dim = x.type.shape[i]
            if not isinstance(dim, int) or isinstance(dim, bool):
                raise BackendError(
                    f"stablehlo export: op 'reduce_mean'{self._loc(op)} "
                    "reduces over a non-static dimension — the element "
                    "count cannot be computed statically; decompose it "
                    "manually or use a future adapter"
                )
            count *= dim
        out_dtype = np.dtype(op.result.type.dtype)
        lines, sum_name, reduced_shape = self._emit_reduce_core(x, dims, "sum")
        converted = self._new_name()
        lines.append(
            f"{converted} = stablehlo.convert {sum_name} : "
            f"{self._type_str(out_dtype, reduced_shape)}"
        )
        count_name = self._new_name()
        lines.append(
            f"{count_name} = stablehlo.constant "
            f"{self._constant_text(np.asarray(count, dtype=out_dtype))}"
            f" : {self._elem_type(out_dtype)}"
        )
        divided = self._new_name()
        lines.append(
            f"{divided} = stablehlo.divide {converted}, {count_name} : "
            f"{self._type_str(out_dtype, reduced_shape)}"
        )
        cur = divided
        if keepdims and tuple(op.result.type.shape) != tuple(reduced_shape):
            reshaped = self._new_name()
            lines.append(
                f"{reshaped} = stablehlo.reshape {cur} : "
                f"{self._vt(op.result.type)}"
            )
            cur = reshaped
        self._names[id(op.result)] = cur
        return "\n".join(lines)

    def _emit_reduce_core(self, x: Value, dims: list, kind: str):
        """Emit `stablehlo.reduce` (+ init constant + reducer region);
        returns ``(lines, result_name, reduced_shape)`` with the operand
        element dtype (callers add convert/reshape as needed)."""
        elem_dtype = np.dtype(x.type.dtype)
        init_value = self._reduce_init(kind, elem_dtype)
        body_mnemonic = _REDUCE_BODY_MNEMONIC[kind]
        elem_type = self._elem_type(elem_dtype)
        lines = []
        init_name = self._new_name()
        lines.append(
            f"{init_name} = stablehlo.constant "
            f"{self._constant_text(np.asarray(init_value, dtype=elem_dtype))}"
            f" : {elem_type}"
        )
        body_name = self._new_name()
        region = (
            "({\n"
            f"  ^bb0(%arg0: {elem_type}, %arg1: {elem_type}):\n"
            f"    {body_name} = stablehlo.{body_mnemonic} %arg0, %arg1"
            f" : {elem_type}\n"
            f"    stablehlo.return {body_name} : {elem_type}\n"
            "  })"
        )
        reduced_shape = tuple(
            d for i, d in enumerate(x.type.shape) if i not in set(dims)
        )
        reduced_type = self._type_str(elem_dtype, reduced_shape)
        result_name = self._new_name()
        lines.append(
            f'{result_name} = "stablehlo.reduce"({self._name(x)}, '
            f"{init_name}) {region} {{dimensions = {self._dense_1d(dims)}}}"
            f" : ({self._vt(x.type)}, {elem_type}) -> {reduced_type}"
        )
        return lines, result_name, reduced_shape

    def _emit_reduce_region(self, op: Op, kind: str) -> tuple[str, str]:
        """Reducer region for collectives (no init value): a
        two-argument block over the scalar element type; returns
        ``(region_text, body_name)``."""
        elem_dtype = np.dtype(op.operands[0].type.dtype)
        body_mnemonic = _REDUCE_BODY_MNEMONIC[kind]
        elem_type = self._elem_type(elem_dtype)
        body_name = self._new_name()
        region = (
            "({\n"
            f"  ^bb0(%arg0: {elem_type}, %arg1: {elem_type}):\n"
            f"    {body_name} = stablehlo.{body_mnemonic} %arg0, %arg1"
            f" : {elem_type}\n"
            f"    stablehlo.return {body_name} : {elem_type}\n"
            "  })"
        )
        return region, body_name

    @staticmethod
    def _reduce_dims(x: Value, axes: tuple) -> list:
        rank = x.type.rank
        if axes:
            return sorted({int(a) for a in axes})
        return list(range(rank))  # empty axes = reduce over ALL axes

    @staticmethod
    def _reduce_init(kind: str, dtype: np.dtype):
        """Identity/extreme init value for a reduction kind + dtype."""
        k = dtype.kind
        if kind == "sum":
            return False if k == "b" else 0
        if kind == "prod":
            return True if k == "b" else 1
        if kind == "max":
            if k == "b":
                return False
            if k in "iu":
                return np.iinfo(dtype).min
            return -np.inf
        if kind == "min":
            if k == "b":
                return True
            if k in "iu":
                return np.iinfo(dtype).max
            return np.inf
        raise BackendError(
            f"stablehlo export: unknown reduction kind {kind!r}"
        )

    # --- Linear algebra ---

    def _emit_dot(self, op: Op) -> str:
        a, b = op.operands
        la, lb = a.type.rank, b.type.rank
        # etl dot = batched matmul (numpy contract, rank >= 2 both sides):
        # contracting dims are the last of `a` / second-to-last of `b`,
        # batch dims are the leading ones.
        dnums = (
            "#stablehlo.dot<"
            f"lhs_batching_dimensions = [{self._int_list(range(la - 2))}], "
            f"rhs_batching_dimensions = [{self._int_list(range(lb - 2))}], "
            f"lhs_contracting_dimensions = [{la - 1}], "
            f"rhs_contracting_dimensions = [{lb - 2}]>"
        )
        elem_dtype = np.dtype(a.type.dtype)
        dot_type = self._type_str(elem_dtype, op.result.type.shape)
        dot_name = self._new_name()
        lines = [
            f'{dot_name} = "stablehlo.dot_general"({self._name(a)}, '
            f"{self._name(b)}) {{dot_dimension_numbers = {dnums}}} : "
            f"({self._vt(a.type)}, {self._vt(b.type)}) -> {dot_type}"
        ]
        cur = dot_name
        # dot_general does not promote (XLA keeps the operand element type);
        # etl dot promotes dtypes per numpy.
        if np.dtype(op.result.type.dtype) != elem_dtype:
            converted = self._new_name()
            lines.append(
                f"{converted} = stablehlo.convert {cur} : "
                f"{self._vt(op.result.type)}"
            )
            cur = converted
        self._names[id(op.result)] = cur
        return "\n".join(lines)

    def _emit_conv(self, op: Op) -> str:
        x, w = op.operands
        rank = x.type.rank
        n_spatial = rank - 2
        attrs = op.attributes
        strides = tuple(attrs.get("strides") or (1,) * n_spatial)
        padding = attrs.get("padding", "VALID")
        in_dilation = tuple(attrs.get("input_dilation") or (1,) * n_spatial)
        k_dilation = tuple(attrs.get("kernel_dilation") or (1,) * n_spatial)
        feature_groups = attrs.get("feature_group_count", 1)
        batch_groups = attrs.get("batch_group_count", 1)

        spatial = self._int_list(range(n_spatial))
        # etl conv is NCHW (N, C_in, *spatial) / (C_out, C_in/g, *spatial).
        dnums = (
            "#stablehlo.conv<"
            f"[b, {spatial}, f]x[{spatial}, i, o]->[b, {spatial}, f]>"
        )
        attr_parts = [f"dimension_numbers = {dnums}"]
        if isinstance(padding, str):
            mode = padding.upper()
            if mode == "VALID":
                attr_parts.append(f"padding = dense<0> : tensor<{n_spatial}x2xi64>")
            elif mode == "SAME":
                pairs = self._same_padding_pairs(op, x, w, strides, k_dilation)
                attr_parts.append(f"padding = {self._dense_2d(pairs)}")
            else:
                raise BackendError(
                    f"stablehlo export: op 'conv'{self._loc(op)} has unknown "
                    f"padding mode {padding!r} (expected 'VALID', 'SAME', or "
                    "per-dim (lo, hi) pairs)"
                )
        else:
            pairs = [tuple(p) for p in padding]
            attr_parts.append(f"padding = {self._dense_2d(pairs)}")
        if any(s != 1 for s in strides):
            attr_parts.append(f"window_strides = {self._dense_1d(strides)}")
        if any(d != 1 for d in in_dilation):
            attr_parts.append(f"lhs_dilation = {self._dense_1d(in_dilation)}")
        if any(d != 1 for d in k_dilation):
            attr_parts.append(f"rhs_dilation = {self._dense_1d(k_dilation)}")
        if feature_groups != 1:
            attr_parts.append(f"feature_group_count = {int(feature_groups)} : i64")
        if batch_groups != 1:
            attr_parts.append(f"batch_group_count = {int(batch_groups)} : i64")

        elem_dtype = np.dtype(x.type.dtype)
        conv_type = self._type_str(elem_dtype, op.result.type.shape)
        conv_name = self._new_name()
        lines = [
            f'{conv_name} = "stablehlo.convolution"({self._name(x)}, '
            f"{self._name(w)}) {{{', '.join(attr_parts)}}} : "
            f"({self._vt(x.type)}, {self._vt(w.type)}) -> {conv_type}"
        ]
        cur = conv_name
        if np.dtype(op.result.type.dtype) != elem_dtype:
            converted = self._new_name()
            lines.append(
                f"{converted} = stablehlo.convert {cur} : "
                f"{self._vt(op.result.type)}"
            )
            cur = converted
        self._names[id(op.result)] = cur
        return "\n".join(lines)

    def _same_padding_pairs(self, op: Op, x: Value, w: Value, strides, k_dilation):
        """Explicit (lo, hi) SAME padding pairs (XLA rule): total =
        max((out - 1) * stride + kdil * (k - 1) + 1 - in, 0), lo =
        total // 2. Symbolic in/out/kernel dims raise BackendError."""
        rank = x.type.rank
        pairs = []
        for i in range(rank - 2):
            in_dim = x.type.shape[2 + i]
            out_dim = op.result.type.shape[2 + i]
            k_dim = w.type.shape[2 + i]
            for dim, label in ((in_dim, "input"), (out_dim, "output"), (k_dim, "kernel")):
                if not isinstance(dim, int) or isinstance(dim, bool):
                    raise BackendError(
                        f"stablehlo export: op 'conv'{self._loc(op)} with "
                        f"'SAME' padding needs a static {label} dimension, "
                        f"got {dim!r} — use explicit (lo, hi) pairs instead"
                    )
            total = max(
                (out_dim - 1) * strides[i]
                + k_dilation[i] * (k_dim - 1)
                + 1
                - in_dim,
                0,
            )
            pairs.append((total // 2, total - total // 2))
        return pairs

    # --- Control flow ---

    def _emit_if(self, op: Op) -> str:
        pred = op.operands[0]
        result_names = self._bind_results(op)
        lhs = f"{', '.join(result_names)} = " if result_names else ""
        lines = [f'{lhs}"stablehlo.if"({self._name(pred)}) ({{']
        lines.append(self._emit_if_region(op.regions[0], op))
        lines.append("  }, {")
        lines.append(self._emit_if_region(op.regions[1], op))
        lines.append(
            f"  }}) : ({self._vt(pred.type)}) -> "
            f"{self._result_types_str(op.results)}"
        )
        return "\n".join(lines)

    def _emit_if_region(self, region, op: Op) -> str:
        """One `if` branch: no block args (StableHLO branches capture the
        enclosing values implicitly) — the etl entry args bind to the op
        operands, so references resolve to the enclosing SSA names."""
        out = []
        for i, block in enumerate(region.blocks):
            for arg, operand in zip(block.arguments, op.operands):
                self._names[id(arg)] = self._name(operand)
            out.append(f"  ^bb{i}:")
            out.extend("    " + line for line in self._block_body_lines(block))
        return "\n".join(out)

    def _emit_while(self, op: Op) -> str:
        operand_names = ", ".join(self._name(v) for v in op.operands)
        result_names = self._bind_results(op)
        lhs = f"{', '.join(result_names)} = " if result_names else ""
        types = ", ".join(self._vt(v.type) for v in op.operands)
        lines = [f'{lhs}"stablehlo.while"({operand_names}) ({{']
        lines.append(self._emit_while_region(op.regions[0]))
        lines.append("  }, {")
        lines.append(self._emit_while_region(op.regions[1]))
        lines.append(f"  }}) : ({types}) -> ({types})")
        return "\n".join(lines)

    def _emit_while_region(self, region) -> str:
        """One `while` region (cond/body): fresh `%argN` block args bound
        positionally to the loop-carried operands, named past the
        enclosing function's entry args (see `_arg_base`) so captured
        values keep their names."""
        out = []
        for i, block in enumerate(region.blocks):
            args = []
            for j, arg in enumerate(block.arguments):
                name = f"%arg{self._arg_base + j}"
                self._names[id(arg)] = name
                args.append(f"{name}: {self._vt(arg.type)}")
            out.append(f"  ^bb{i}({', '.join(args)}):")
            out.extend("    " + line for line in self._block_body_lines(block))
        return "\n".join(out)

    # --- Collectives ---

    def _emit_collective(self, op: Op) -> str:
        name = op.name
        mnemonic = mapping.lookup_mapping(name)
        x = op.operands[0]
        attrs = op.attributes
        group_size = attrs.get("group_size")
        replica_groups = self._replica_groups(group_size)
        result_name = self._bind_results(op)[0]
        signature = (
            f"({self._vt(x.type)}) -> {self._vt(op.result.type)}"
        )
        if name in ("all_reduce", "reduce_scatter"):
            kind = attrs["reduce_op"]
            region, _ = self._emit_reduce_region(op, kind)
            if name == "all_reduce":
                attr_str = f"{{replica_groups = {replica_groups}}}"
            else:
                axis = attrs.get("axis", 0)
                attr_str = (
                    f"{{scatter_dimension = {int(axis)} : i64, "
                    f"replica_groups = {replica_groups}}}"
                )
            return (
                f'{result_name} = "{mnemonic}"({self._name(x)}) {region} '
                f"{attr_str} : {signature}"
            )
        if name == "all_gather":
            axis = attrs.get("axis", 0)
            attr_str = (
                f"{{all_gather_dim = {int(axis)} : i64, "
                f"replica_groups = {replica_groups}}}"
            )
            return (
                f'{result_name} = "{mnemonic}"({self._name(x)}) '
                f"{attr_str} : {signature}"
            )
        if name == "all_to_all":
            split_axis = attrs["split_axis"]
            concat_axis = attrs["concat_axis"]
            # StableHLO requires a static split_count; with an unknown world
            # group (group_size None) emit the single-rank count (v1
            # single-process simulation semantics).
            if isinstance(group_size, int) and not isinstance(group_size, bool) and group_size > 0:
                split_count = group_size
            else:
                split_count = 1
            attr_str = (
                f"{{split_dimension = {int(split_axis)} : i64, "
                f"concat_dimension = {int(concat_axis)} : i64, "
                f"split_count = {int(split_count)} : i64, "
                f"replica_groups = {replica_groups}}}"
            )
            return (
                f'{result_name} = "{mnemonic}"({self._name(x)}) '
                f"{attr_str} : {signature}"
            )
        if name == "broadcast_collective":
            # etl's src_rank attribute has no StableHLO counterpart
            # (collective_broadcast semantics are group-wide); channels are
            # only required for cross-partition communication, which v1
            # single-process export does not use.
            attr_str = f"{{replica_groups = {replica_groups}}}"
            return (
                f'{result_name} = "{mnemonic}"({self._name(x)}) '
                f"{attr_str} : {signature}"
            )
        if name == "collective_permute":
            pairs = attrs["source_target_pairs"]
            attr_str = f"{{source_target_pairs = {self._dense_2d(pairs)}}}"
            return (
                f'{result_name} = "{mnemonic}"({self._name(x)}) '
                f"{attr_str} : {signature}"
            )
        raise self._unsupported(op)

    # --- Decompositions ---

    def _emit_decomposed(self, op: Op) -> str:
        name = op.name
        if name == "square":
            x = self._name(op.operands[0])
            result_name = self._bind_results(op)[0]
            return (
                f"{result_name} = stablehlo.multiply {x}, {x} : "
                f"{self._vt(op.result.type)}"
            )
        if name == "relu":
            x = op.operands[0]
            dtype = np.dtype(x.type.dtype)
            zero_name = self._new_name()
            result_name = self._bind_results(op)[0]
            return (
                f"{zero_name} = stablehlo.constant "
                f"{self._constant_text(np.zeros((), dtype=dtype))}"
                f" : {self._elem_type(dtype)}\n"
                f"{result_name} = stablehlo.maximum {self._name(x)}, "
                f"{zero_name} : {self._vt(op.result.type)}"
            )
        if name == "stop_gradient":
            self._names[id(op.result)] = self._name(op.operands[0])
            return ""
        if name == "reduce_mean":
            return self._emit_reduce_mean(op)
        raise self._unsupported(op)

    # --- Block/region emission ---

    def _block_body_lines(self, block: Block, terminator: str = "stablehlo.return"):
        """The op lines of one block, ending with its terminator line."""
        lines = []
        for op in block.ops:
            if op.name == "return":
                if op.operands:
                    values = ", ".join(self._name(v) for v in op.operands)
                    types = ", ".join(self._vt(v.type) for v in op.operands)
                    lines.append(f"{terminator} {values} : {types}")
                else:
                    lines.append(terminator)
                break
            text = self._emit_op(op)
            if text:
                lines.append(text)
        return lines

    # --- Naming & formatting helpers ---

    def _bind_results(self, op: Op):
        return [self._bind_value(r) for r in op.results]

    def _bind_value(self, value: Value) -> str:
        name = self._names.get(id(value))
        if name is None:
            name = self._new_name()
            self._names[id(value)] = name
        return name

    def _name(self, value: Value) -> str:
        return self._bind_value(value)

    def _new_name(self) -> str:
        return f"%{next(self._counter)}"

    def _vt(self, vt) -> str:
        return self.mlir_type(vt.dtype, vt.shape)

    def _type_str(self, dtype, shape) -> str:
        return self.mlir_type(dtype, shape)

    def _elem_type(self, dtype) -> str:
        return f"tensor<{mapping.mlir_dtype(dtype)}>"

    def _result_types_str(self, results) -> str:
        types = [self._vt(r.type) for r in results]
        return types[0] if len(types) == 1 else "(" + ", ".join(types) + ")"

    @staticmethod
    def _dim_str(dim) -> str:
        if isinstance(dim, int) and not isinstance(dim, bool):
            return str(dim)
        if dim is None or isinstance(dim, (Dim, DimExpr)):
            return "?"
        return str(dim)

    @staticmethod
    def _int_list(values) -> str:
        return ", ".join(str(int(v)) for v in values)

    @staticmethod
    def _dense_1d(values) -> str:
        inner = ", ".join(str(int(v)) for v in values)
        n = len(values)
        if n == 0:
            return "dense<> : tensor<0xi64>"
        return f"dense<[{inner}]> : tensor<{n}xi64>"

    @staticmethod
    def _dense_2d(pairs) -> str:
        pairs = [tuple(int(v) for v in p) for p in pairs]
        if not pairs:
            return "dense<> : tensor<0x2xi64>"
        rows = ", ".join("[" + ", ".join(map(str, p)) + "]" for p in pairs)
        return f"dense<[{rows}]> : tensor<{len(pairs)}x{len(pairs[0])}xi64>"

    @staticmethod
    def _replica_groups(group_size) -> str:
        ok = isinstance(group_size, int) and not isinstance(group_size, bool)
        n = group_size if ok and group_size > 0 else 1
        return f"dense<[[{', '.join(map(str, range(n)))}]]> : tensor<1x{n}xi64>"

    def _constant_text(self, array) -> str:
        """`dense<[...]>` elements attribute for a numpy array payload."""
        arr = np.asarray(array)
        if arr.size == 0:
            return "dense<>"
        bits = arr.dtype.itemsize * 8

        def fmt_item(v):
            if isinstance(v, (bool, np.bool_)):
                return "true" if v else "false"
            if isinstance(v, (int, np.integer)):
                return str(int(v))
            if isinstance(v, (float, np.floating)):
                return _fmt_float(float(v), bits)
            if isinstance(v, (complex, np.complexfloating)):
                return (
                    f"({_fmt_float(float(v.real), bits)}, "
                    f"{_fmt_float(float(v.imag), bits)})"
                )
            return str(v)

        def nest(a):
            if a.ndim == 0:
                return fmt_item(a.item())
            return "[" + ", ".join(nest(a[i]) for i in range(a.shape[0])) + "]"

        return "dense<" + nest(arr) + ">"

    @staticmethod
    def _loc(op: Op) -> str:
        if op.location is None:
            return ""
        return f" at {op.location.file}:{op.location.line}"

    def _unsupported(self, op: Op) -> BackendError:
        return BackendError(
            f"stablehlo export: op '{op.name}'{self._loc(op)} is not "
            "supported in v1 — decompose it into supported ops or use a "
            "future compiler adapter"
        )

    @staticmethod
    def _indent(text: str, level: int) -> str:
        pad = "  " * level
        return pad + text.replace("\n", "\n" + pad)
