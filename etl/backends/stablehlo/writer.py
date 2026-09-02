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
  counter); function args ``%argN``; region block args (reduce/while) use
  FRESH counter names — never ``%argN`` (MLIR SSA names are
  function-scoped and would collide with the enclosing entry args).
  Decompositions introduce intermediates from the same counter.
- **Index-list attributes** render in the modern DenseI64ArrayAttr syntax
  ``array<i64: 1, 2>`` (empty: ``array<i64>``) — the legacy
  ``dense<[...]> : tensor<Nxi64>`` spelling is REJECTED by the installed
  compiler generation ("failed to satisfy constraint: i64 dense array
  attribute"). Dense TENSOR attributes (conv ``padding``,
  ``replica_groups``, ``source_target_pairs``) keep the ``dense<...>``
  form.
- **Elementwise shape equalization**: StableHLO elementwise ops
  (add/maximum/compare/select/...) require ALL operands and the result to
  share one shape, so scalar operands (Python scalars auto-promoted to
  scalar constants at trace time, decomposed literals) are broadcast to
  the result shape before use; a scalar-only computation keeps the plain
  scalar constant. Broadcasts whose RESULT shape is fully static use
  ``stablehlo.broadcast_in_dim``. Broadcasts whose result has dynamic
  dims use ``stablehlo.dynamic_broadcast_in_dim`` with the runtime
  ``output_dimensions`` built from the shape source via
  ``stablehlo.get_dimension_size`` (+ ``reshape`` / ``concatenate``) —
  ``stablehlo.broadcast_in_dim`` to a dynamic result is forbidden by the
  StableHLO verifier, and this opset generation has no
  ``stablehlo.shape_of``. A dynamic-target broadcast with NO operand
  carrying the full result shape raises ``BackendError`` naming
  "dynamic broadcast" (the runtime output dimensions have no source).
- **convert/reshape** emit the functional-type custom form
  ``: (tensor<...>) -> tensor<...>`` (their operand and result types
  differ; the single-type form is rejected by the compiler).
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
  counter-named block args bound to the loop-carried operands.
- **Collectives**: `replica_groups` built from the op's `group_size` attr
  (None — the world group — defaults to a single rank, matching v1
  single-process simulation). `broadcast_collective` maps to
  ``stablehlo.collective_broadcast`` — no effect-kind disambiguation
  needed (the shape op keeps the name `broadcast`). `all_gather` /
  `reduce_scatter` result axis dims are rendered CONCRETE (operand dim ×/
  // emitted group size) so the emitted program is internally consistent
  — a symbolic `?` there is rejected by the compiler. KNOWN COMPILER-SIDE
  LIMITATION: iree-compile (20241104) parses and verifies
  ``stablehlo.collective_broadcast`` but cannot LEGALIZE it for llvm-cpu
  ("failed to legalize operation ... explicitly marked illegal") in any
  attribute form — an emission change cannot fix that; newer compilers
  with collective_broadcast lowering accept the emitted text as-is.
"""

from __future__ import annotations

from itertools import count
from typing import TYPE_CHECKING

import numpy as np

from etl.core import BackendError, Dim, DimExpr

from . import ops as mapping
from . import random_export

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


def _is_static_dim(dim) -> bool:
    """True for concrete int dims (bool excluded — it is not a size)."""
    return isinstance(dim, int) and not isinstance(dim, bool)


def _is_static_one(dim) -> bool:
    """True for a provable size-1 dim (a concrete int 1 only — symbolic
    dims are never provably 1 at compile time)."""
    return _is_static_dim(dim) and dim == 1


def _shape_is_static(shape) -> bool:
    """True when every dim of ``shape`` is a concrete int."""
    return all(_is_static_dim(d) for d in shape)


def _normalize_rng_bit_generator(value) -> frozenset:
    """Normalize the ``rng_bit_generator`` exporter option to a frozenset
    of algorithm names (see ``Writer.__init__``): ``True`` (legacy bool) →
    ``{"threefry2x32", "philox4x32_10"}``, ``False``/``None`` →
    ``frozenset()``, any collection of algorithm names → the names present
    (unknown names are ignored — the native path only ever applies to the
    threefry2x32/philox4x32_10 ciphers)."""
    if value is None or value is False:
        return frozenset()
    if value is True:
        return frozenset(
            {random_export.ALGORITHM_THREEFRY2X32, random_export.ALGORITHM_PHILOX4X32_10}
        )
    return frozenset(value)


_SORT_EMISSION_MODES = ("pair", "count", "auto")


def _normalize_sort_emission(value) -> str:
    """Validate the ``sort_emission`` exporter option (see
    ``Writer.__init__``): ``"pair"`` | ``"count"`` | ``"auto"``."""
    if value not in _SORT_EMISSION_MODES:
        raise BackendError(
            f"stablehlo export: invalid sort_emission option {value!r} "
            f"(expected one of {', '.join(_SORT_EMISSION_MODES)})"
        )
    return value


class Writer:
    """Emit StableHLO MLIR text for a verified `etl.ir.Module`; the
    module is assumed verified (``export()`` runs ``module.verify()``
    first), the Writer only type-checks it defensively."""

    def __init__(self, module: Module, options: dict | None = None) -> None:
        """Store the verified module and initialize per-write state.

        State: SSA name counter (module-wide), per-value name map
        (``id(Value) -> "%N"``), and the ``rng_bit_generator`` exporter
        option (normalized to a frozenset of algorithm names — the set of
        threefry2x32/philox4x32_10-style random algorithms for which the
        NATIVE ``stablehlo.rng_bit_generator`` emission is selected; the
        empty set = the bit-exact inline expansions, always available; the
        legacy bool ``True`` means both ciphers, ``False``/absent means
        none — consumed by ``random_export``).
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
        #: Per-value type-text overrides (value id -> rendered MLIR type),
        #: used where the emitted program's types legitimately differ from
        #: the IR's symbolic types (all_gather/reduce_scatter axis dims).
        self._type_overrides: dict[int, str] = {}
        #: rng_bit_generator exporter option (see random_export): frozenset
        #: of algorithm names with native emission enabled.
        self._rng_bit_generator = _normalize_rng_bit_generator(
            (options or {}).get("rng_bit_generator", False)
        )
        #: sort_emission exporter option: ``"pair"`` (default — the
        #: two-operand (key, iota) ``stablehlo.sort`` composition), ``"count"``
        #: (the count-based O(n^2) composition with NO sort op, bit-exact vs
        #: numpy on both llvm-cpu and cuda), or ``"auto"`` (per argsort:
        #: ``"count"`` whenever the sorted-axis extent >= 32, else ``"pair"``).
        #: The iree-cuda HAL cannot bufferize multi-operand sorts at sorted
        #: axis >= 32 (upstream iree 3.11.0 bug) — the iree adapter passes
        #: ``"auto"`` by default (see CompilerBackend.default_sort_emission).
        self._sort_emission = _normalize_sort_emission(
            (options or {}).get("sort_emission", "pair")
        )
        #: Ops already emitted at a hoisted position (by ``_emit_if``'s
        #: while-in-if workaround, see ``_hoist_if_branch_whiles``): their
        #: result names are bound, so ``_block_body_lines`` must SKIP them
        #: when the branch regions are emitted (their lines already landed
        #: BEFORE the enclosing ``stablehlo.if``).
        self._hoisted_ops: set[int] = set()
        #: while_init_rewrite exporter option (bool, default True): replace
        #: ALL-ZERO rank>=1 constant while-INIT operands with computed zeros
        #: (``not``/``and`` + a dtype-changing ``convert`` + ``reduce`` over a
        #: same-shaped non-constant pre-loop value, or an ``iota``-derived
        #: value — never ``x - x`` / broadcast-of-scalar-0, which iree folds
        #: back into constants). iree 3.11.0 SEGVs in
        #: ``IREE::Stream::AffinityAnalysis::run()`` on dense rank>=1
        #: constants used as while inits (upstream bug, see
        #: ``stablehlo/CONTEXT.md`` Known Issues); computed zeros are
        #: bit-exact and compile clean (10/10 validated on llvm-cpu + cuda).
        self._while_init_rewrite = bool(
            (options or {}).get("while_init_rewrite", True)
        )
        #: eigh_early_exit exporter option (bool, default True): the
        #: ``eigh`` while-Jacobi composition additionally carries an i1
        #: ``done`` flag and, at every sweep boundary (inside a nested
        #: ``stablehlo.if``), checks the scale-aware relative off-diagonal
        #: energy of the current A against a dtype tolerance, exiting once
        #: converged (skips the remaining scheduled sweeps). ``False``
        #: emits the EXACT pre-option text (5 carries, cond ``k < total``)
        #: — the A/B measurement lever and safety valve (see ``_emit_eigh``
        #: and ``_emit_eigh_sweep_check``).
        self._eigh_early_exit = bool(
            (options or {}).get("eigh_early_exit", True)
        )

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
        # Emit the body FIRST: op emission populates per-value type
        # overrides (collective axis dims) that the function signature
        # rendering below must already see.
        body = self._block_body_lines(entry, terminator="func.return")
        args = ", ".join(
            f"%arg{i}: {self._vt(arg.type)}"
            for i, arg in enumerate(entry.arguments)
        )
        # Function outputs come off the return terminator's operands (the
        # same values `fn.output_types` reads types from) so per-value type
        # overrides (collective axis dims) stay consistent with the body.
        outputs = entry.terminator.operands if entry.terminator else ()
        if not outputs:
            results = ""
        elif len(outputs) == 1:
            results = f" -> {self._vt_value(outputs[0])}"
        else:
            results = " -> (" + ", ".join(self._vt_value(t) for t in outputs) + ")"
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
        if name in mapping.SPECIAL_EMITTERS:
            return self._emit_special(op)
        raise self._unsupported(op)

    def _emit_special(self, op: Op) -> str:
        """Dispatch ops that need dedicated multi-op emitters (see
        ``mapping.SPECIAL_EMITTERS``): gather/scatter/sort/argsort/argmax/
        argmin/tile, the ``eigh`` Jacobi composition, the ``diag``
        compositions, and the SplitMix64 random expansions."""
        family = mapping.SPECIAL_EMITTERS[op.name]
        if family == "gather":
            return self._emit_gather(op)
        if family == "scatter":
            return self._emit_scatter(op)
        if family == "sort":
            return self._emit_sort(op)
        if family == "argsort":
            return self._emit_argsort(op)
        if family == "arg_reduce":
            return self._emit_arg_reduce(op)
        if family == "tile":
            return self._emit_tile(op)
        if family == "eigh":
            return self._emit_eigh(op)
        if family == "diag":
            return self._emit_diag(op)
        if family == "random":
            return random_export.emit_random_op(self, op)
        raise self._unsupported(op)

    def _emit_elementwise(self, op: Op) -> str:
        """Unary/binary elementwise ops (incl. cast).

        StableHLO elementwise ops require ALL operands and the result to
        share one shape (the verifier rejects mixed scalar/non-scalar
        operands), so scalar operands — Python scalars auto-promoted to
        scalar constants at trace time — are broadcast to the result shape
        first. They must ALSO share one element type: etl binary ops
        promote operands per ``np.result_type`` while StableHLO keeps the
        operand types, so after shape equalization every operand whose
        dtype differs from the result dtype is ``stablehlo.convert``-ed to
        it (the broadcast preserves the operand dtype, so converting after
        equalization is always well-typed, including dynamic ``?`` dims).
        Unary ops never need this (their dtype is preserved). ``cast``
        emits ``stablehlo.convert`` in the functional-type custom form
        (operand and result types differ by definition).
        """
        mnemonic = mapping.lookup_mapping(op.name)
        if op.name == "bitwise_right_shift":
            # numpy `np.right_shift` is dtype-natural: arithmetic on signed,
            # logical on unsigned. Operands are converted to the result dtype
            # below, so the result dtype decides the StableHLO mnemonic.
            shift_dtype = np.dtype(op.result.type.dtype)
            if shift_dtype.kind == "u":
                mnemonic = "stablehlo.shift_right_logical"
            elif shift_dtype.kind == "i":
                mnemonic = "stablehlo.shift_right_arithmetic"
            else:
                raise BackendError(
                    f"stablehlo export: op 'bitwise_right_shift'{self._loc(op)} "
                    f"over dtype {shift_dtype!r} is not supported (the "
                    "frontend requires integer operands)"
                )
        result_shape = tuple(op.result.type.shape)
        result_name = self._bind_results(op)[0]
        shape_source = self._shape_source(op.operands, result_shape)
        lines: list[str] = []
        names = []
        for operand in op.operands:
            name, extra = self._equalize_operand(operand, result_shape, shape_source)
            lines.extend(extra)
            names.append(name)
        if op.name == "cast":
            operand_type = self._vt(op.operands[0].type)
            lines.append(
                f"{result_name} = stablehlo.convert {names[0]} : "
                f"({operand_type}) -> {self._vt(op.result.type)}"
            )
        else:
            result_dtype = np.dtype(op.result.type.dtype)
            for i, (name, operand) in enumerate(zip(names, op.operands)):
                if np.dtype(operand.type.dtype) != result_dtype:
                    converted = self._new_name()
                    lines.append(
                        f"{converted} = stablehlo.convert {name} : "
                        f"({self._type_str(operand.type.dtype, result_shape)}) -> "
                        f"{self._type_str(result_dtype, result_shape)}"
                    )
                    names[i] = converted
            lines.append(
                f"{result_name} = {mnemonic} {', '.join(names)} : "
                f"{self._vt(op.result.type)}"
            )
        return "\n".join(lines)

    def _shape_source(self, operands, result_shape):
        """An operand whose type shape equals ``result_shape`` — the runtime
        source for dynamic-broadcast output dimensions; ``None`` when the
        result shape is static or no operand carries the full shape."""
        if _shape_is_static(result_shape):
            return None
        for operand in operands:
            if tuple(operand.type.shape) == result_shape:
                return operand
        return None

    def _emit_dynamic_broadcast(
        self,
        value_name: str,
        dtype,
        operand_shape,
        result_shape,
        dims,
        src_name: str,
        src_dtype,
        src_shape,
    ) -> tuple[str, list]:
        """Emit ``stablehlo.dynamic_broadcast_in_dim`` for a broadcast whose
        RESULT has dynamic dims.

        The runtime ``output_dimensions`` (a ``tensor<Rxi32>``) are built
        from the shape source ``src``: static dims via constants, dynamic
        dims via ``stablehlo.get_dimension_size`` (+ ``reshape`` to
        ``tensor<1xi32>``), joined by ``stablehlo.concatenate``. Returns
        ``(broadcast_name, lines)``. Requires a shape source whose shape
        equals ``result_shape`` (callers enforce this) — this opset
        generation has no ``stablehlo.shape_of``.
        """
        lines: list[str] = []
        pieces: list[str] = []
        src_type = self._type_str(src_dtype, src_shape)
        for i, dim in enumerate(result_shape):
            if _is_static_dim(dim):
                piece = self._new_name()
                lines.append(
                    f"{piece} = stablehlo.constant "
                    f"{self._constant_text(np.asarray([int(dim)], dtype=np.int32))}"
                    f" : tensor<1xi32>"
                )
            else:
                size_name = self._new_name()
                lines.append(
                    f'{size_name} = "stablehlo.get_dimension_size"({src_name}) '
                    f"{{dimension = {i} : i64}} : ({src_type}) -> tensor<i32>"
                )
                piece = self._new_name()
                lines.append(
                    f"{piece} = stablehlo.reshape {size_name} : "
                    f"(tensor<i32>) -> tensor<1xi32>"
                )
            pieces.append(piece)
        if len(pieces) == 1:
            dims_name = pieces[0]
        else:
            dims_name = self._new_name()
            piece_types = ", ".join("tensor<1xi32>" for _ in pieces)
            lines.append(
                f"{dims_name} = stablehlo.concatenate {', '.join(pieces)}, "
                f"dim = 0 : ({piece_types}) -> tensor<{len(pieces)}xi32>"
            )
        bcast_name = self._new_name()
        lines.append(
            f'{bcast_name} = "stablehlo.dynamic_broadcast_in_dim"'
            f"({value_name}, {dims_name}) "
            f"{{broadcast_dimensions = {self._i64_array(dims)}}} : "
            f"({self._type_str(dtype, operand_shape)}, "
            f"tensor<{len(pieces)}xi32>) -> {self._type_str(dtype, result_shape)}"
        )
        return bcast_name, lines

    def _equalize_operand(
        self, value: Value, result_shape: tuple, shape_source=None
    ) -> tuple[str, list]:
        """SSA name of ``value`` valid inside an elementwise op whose result
        has ``result_shape``, plus any broadcast lines to prepend.

        Operands whose shape already equals the result shape are used
        directly. A rank-0 operand is broadcast with an empty
        ``broadcast_dimensions`` (``array<i64>``). A non-scalar operand with
        a differing shape is broadcast when every mismatched dim is size 1
        (numpy-style broadcasting — the etl shape-inference contract);
        anything else raises ``BackendError`` naming the operand (never
        silently emits an op the compiler would reject).

        Static result shapes emit ``stablehlo.broadcast_in_dim``. Dynamic
        result shapes emit ``stablehlo.dynamic_broadcast_in_dim`` with the
        runtime output dimensions sourced from ``shape_source`` (an operand
        whose type shape equals the result shape); without one the
        broadcast cannot be computed — ``BackendError`` naming
        "dynamic broadcast".
        """
        name = self._name(value)
        shape = tuple(value.type.shape)
        if shape == result_shape:
            return name, []
        rank_diff = len(result_shape) - len(shape)
        dims = [rank_diff + i for i in range(len(shape))]
        for i, dim in enumerate(shape):
            target = result_shape[rank_diff + i]
            if _is_static_dim(dim):
                if dim != 1 and dim != target:
                    raise BackendError(
                        f"stablehlo export: cannot broadcast operand of "
                        f"shape {shape!r} to the elementwise result shape "
                        f"{result_shape!r} — StableHLO elementwise ops "
                        "require equal operand shapes"
                    )
            elif dim != target:
                raise BackendError(
                    f"stablehlo export: cannot broadcast operand of shape "
                    f"{shape!r} to the elementwise result shape "
                    f"{result_shape!r} — symbolic operand dimension {dim!r} "
                    "does not match the result dimension"
                )
        if _shape_is_static(result_shape):
            bcast_name = self._new_name()
            line = (
                f'{bcast_name} = "stablehlo.broadcast_in_dim"({name}) '
                f"{{broadcast_dimensions = {self._i64_array(dims)}}} : "
                f"({self._vt(value.type)}) -> "
                f"{self._type_str(value.type.dtype, result_shape)}"
            )
            return bcast_name, [line]
        if shape_source is None:
            raise BackendError(
                f"stablehlo export: cannot broadcast operand of shape "
                f"{shape!r} to the dynamic result shape {result_shape!r} — "
                "no operand carries the full result shape to source the "
                "runtime output dimensions (dynamic broadcast)"
            )
        return self._emit_dynamic_broadcast(
            name,
            value.type.dtype,
            shape,
            result_shape,
            dims,
            self._name(shape_source),
            shape_source.type.dtype,
            tuple(shape_source.type.shape),
        )

    def _scalar_constant_for(
        self, dtype, value, target_shape, shape_source=None
    ) -> tuple[str, list]:
        """Emit a scalar constant and — when ``target_shape`` is non-scalar —
        its broadcast to that shape; returns ``(name, lines)``.

        StableHLO elementwise ops require equal shapes, so decomposed
        literals (``relu``'s zero, ``reduce_mean``'s divisor) interacting
        with a non-scalar tensor must be broadcast first. When the target
        IS a scalar the constant is used directly (broadcast_in_dim with
        empty dims is invalid on scalars). Static targets emit
        ``stablehlo.broadcast_in_dim``; dynamic targets emit
        ``stablehlo.dynamic_broadcast_in_dim`` with ``shape_source`` (an
        SSA ``(name, dtype, shape)`` tuple whose shape equals the target —
        the runtime output-dimension source); without one,
        ``BackendError`` naming "dynamic broadcast".
        """
        name = self._new_name()
        lines = [
            f"{name} = stablehlo.constant "
            f"{self._constant_text(np.asarray(value, dtype=dtype))}"
            f" : {self._elem_type(dtype)}"
        ]
        if target_shape:
            if _shape_is_static(target_shape):
                bcast_name = self._new_name()
                lines.append(
                    f'{bcast_name} = "stablehlo.broadcast_in_dim"({name}) '
                    f"{{broadcast_dimensions = array<i64>}} : "
                    f"({self._elem_type(dtype)}) -> "
                    f"{self._type_str(dtype, target_shape)}"
                )
                return bcast_name, lines
            if shape_source is None:
                raise BackendError(
                    f"stablehlo export: cannot broadcast a scalar constant "
                    f"to the dynamic target shape {target_shape!r} — no "
                    "shape source for the runtime output dimensions "
                    "(dynamic broadcast)"
                )
            bcast_name, extra = self._emit_dynamic_broadcast(
                name, dtype, (), target_shape, (), *shape_source
            )
            lines.extend(extra)
            return bcast_name, lines
        return name, lines

    def _emit_select(self, op: Op) -> str:
        """`stablehlo.select` — all operands equalized to the result shape
        (StableHLO requires equal non-scalar shapes), and the on_true /
        on_false branches converted to the promoted result dtype (the
        non-predicate operands and the result must share one element type;
        the pred stays i1 — etl select promotes branches per
        ``np.result_type`` while StableHLO keeps the operand types)."""
        result_shape = tuple(op.result.type.shape)
        result_name = self._bind_results(op)[0]
        shape_source = self._shape_source(op.operands, result_shape)
        lines: list[str] = []
        names = []
        for operand in op.operands:
            name, extra = self._equalize_operand(operand, result_shape, shape_source)
            lines.extend(extra)
            names.append(name)
        branch_dtype = np.dtype(op.result.type.dtype)
        branch_types = []
        for i, name in enumerate(names[1:], start=1):
            dtype = np.dtype(op.operands[i].type.dtype)
            if dtype != branch_dtype:
                converted = self._new_name()
                lines.append(
                    f"{converted} = stablehlo.convert {name} : "
                    f"({self._type_str(dtype, result_shape)}) -> "
                    f"{self._type_str(branch_dtype, result_shape)}"
                )
                name = converted
                names[i] = name
            branch_types.append(self._type_str(branch_dtype, result_shape))
        op_types = ", ".join(
            [self._type_str(op.operands[0].type.dtype, result_shape)]
            + branch_types
        )
        lines.append(
            f"{result_name} = stablehlo.select {', '.join(names)} : "
            f"({op_types}) -> {self._vt(op.result.type)}"
        )
        return "\n".join(lines)

    def _emit_compare(self, op: Op) -> str:
        """`stablehlo.compare` — both operands equalized to the result
        shape (StableHLO requires equal non-scalar shapes) and converted to
        their numpy-promoted dtype ``np.result_type(lhs, rhs)`` (etl
        comparisons promote per numpy while StableHLO keeps the operand
        types; the i1 result is NOT the conversion target — the operands
        must keep the promoted numeric dtype, matching the numpy
        interpreter's promotion)."""
        direction = mapping.lookup_mapping(op.name)
        result_shape = tuple(op.result.type.shape)
        result_name = self._bind_results(op)[0]
        shape_source = self._shape_source(op.operands, result_shape)
        lines: list[str] = []
        names = []
        for operand in op.operands:
            name, extra = self._equalize_operand(operand, result_shape, shape_source)
            lines.extend(extra)
            names.append(name)
        promoted = np.result_type(*(np.dtype(v.type.dtype) for v in op.operands))
        op_types = []
        for i, (name, operand) in enumerate(zip(names, op.operands)):
            if np.dtype(operand.type.dtype) != promoted:
                converted = self._new_name()
                lines.append(
                    f"{converted} = stablehlo.convert {name} : "
                    f"({self._type_str(operand.type.dtype, result_shape)}) -> "
                    f"{self._type_str(promoted, result_shape)}"
                )
                name = converted
                names[i] = name
            op_types.append(self._type_str(promoted, result_shape))
        lines.append(
            f'{result_name} = "stablehlo.compare"({", ".join(names)}) '
            f"{{comparison_direction = #stablehlo<comparison_direction "
            f"{direction}>}} : ({', '.join(op_types)}) -> {self._vt(op.result.type)}"
        )
        return "\n".join(lines)

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
        result_shape = tuple(op.result.type.shape)
        if _shape_is_static(result_shape):
            return (
                f'{result_name} = "stablehlo.broadcast_in_dim"({self._name(x)}) '
                f"{{broadcast_dimensions = {self._i64_array(dims)}}} : "
                f"({self._vt(x.type)}) -> {self._vt(op.result.type)}"
            )
        # Dynamic result: only the identity broadcast (the operand already
        # carries the full result shape) can source its runtime output
        # dimensions — broadcast_in_dim to a dynamic result is forbidden by
        # the StableHLO verifier, and there is no other shape source.
        if tuple(x.type.shape) == result_shape:
            bcast_name, lines = self._emit_dynamic_broadcast(
                self._name(x),
                x.type.dtype,
                tuple(x.type.shape),
                result_shape,
                dims,
                self._name(x),
                x.type.dtype,
                tuple(x.type.shape),
            )
            self._names[id(op.result)] = bcast_name
            return "\n".join(lines)
        raise BackendError(
            f"stablehlo export: op 'broadcast'{self._loc(op)} to the dynamic "
            f"result shape {result_shape!r} cannot be computed — the operand "
            f"of shape {tuple(x.type.shape)!r} does not carry the full "
            "result shape, so the runtime output dimensions have no source "
            "(dynamic broadcast)"
        )

    def _reject_dynamic_dims(self, op: Op, shape, what: str, op_name=None) -> None:
        """Raise ``core.BackendError`` when ``shape`` carries dynamic dims
        (``Dim``/``DimExpr``/``None`` — ints are static; bool is an int
        subclass and counts as static). ``what`` names the shape's role in
        the op (e.g. ``"operand shape"``); ``op_name`` overrides the op name
        in the message (the keepdims reshapes inside the reduce emitters
        report as 'reshape'). This keeps invalid MLIR away from the
        compilers: a dynamic-dims op must fail here, at export/lower time,
        never with an obscure iree-compile parse error later."""
        dynamic = tuple(d for d in shape if not isinstance(d, (int, np.integer)))
        if not dynamic:
            return
        name = op_name or op.name
        raise BackendError(
            f"stablehlo export: op '{name}'{self._loc(op)} {what} has "
            f"dynamic dims {tuple(shape)!r} (offending dims {dynamic!r}) — "
            "dynamic shapes are not supported by the StableHLO compiler "
            "backends in v1 (use concrete static shapes or the numpy backend)"
        )

    def _emit_reshape(self, op: Op) -> str:
        """`stablehlo.reshape` uses the functional-type custom form
        (operand and result types differ — the single-type form is
        rejected by the compiler as "invalid kind of type specified").

        The StableHLO verifier requires statically shaped (or single bounded
        dimension) reshape operands/results, so any dynamic dim fails here
        with a clear BackendError — never invalid MLIR."""
        self._reject_dynamic_dims(op, tuple(op.operands[0].type.shape), "operand shape")
        self._reject_dynamic_dims(op, tuple(op.result.type.shape), "result shape")
        result_name = self._bind_results(op)[0]
        return (
            f"{result_name} = stablehlo.reshape {self._name(op.operands[0])}"
            f" : ({self._vt(op.operands[0].type)}) -> {self._vt(op.result.type)}"
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
            f"{{permutation = {self._i64_array(permutation)}}} : "
            f"({self._vt(x.type)}) -> {self._vt(op.result.type)}"
        )

    def _emit_slice(self, op: Op) -> str:
        x = op.operands[0]
        # EMPIRICAL (iree 3.11.0): iree-compile parses a dynamic-operand
        # slice but the runtime ABORTS on it ("hal.fence.await" failure) —
        # reject here at export/lower time, never emit it (see
        # _reject_dynamic_dims).
        self._reject_dynamic_dims(op, tuple(x.type.shape), "operand shape")
        self._reject_dynamic_dims(op, tuple(op.result.type.shape), "result shape")
        starts = tuple(op.attributes["start_indices"])
        limits = tuple(op.attributes["limit_indices"])
        strides = tuple(op.attributes.get("strides") or (1,) * len(starts))
        result_name = self._bind_results(op)[0]
        return (
            f'{result_name} = "stablehlo.slice"({self._name(x)}) '
            f"{{start_indices = {self._i64_array(starts)}, "
            f"limit_indices = {self._i64_array(limits)}, "
            f"strides = {self._i64_array(strides)}}} : "
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
        # EMPIRICAL (iree 3.11.0): iree-compile parses dynamic-shape pad but
        # the runtime ABORTS on it ("hal.fence.await" failure) — reject here
        # at export/lower time, never emit it (see _reject_dynamic_dims).
        self._reject_dynamic_dims(op, tuple(x.type.shape), "operand shape")
        self._reject_dynamic_dims(op, tuple(op.result.type.shape), "result shape")
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
        low = self._i64_array([p[0] for p in pairs])
        high = self._i64_array([p[1] for p in pairs])
        interior = self._i64_array([0] * rank)
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
                f"({self._type_str(elem_dtype, reduced_shape)}) -> "
                f"{self._type_str(op.result.type.dtype, reduced_shape)}"
            )
            cur = converted
        if keepdims and tuple(op.result.type.shape) != tuple(reduced_shape):
            # The keepdims reshape must be static: a dynamic dim moved
            # between reduced/result shapes is invalid StableHLO reshape.
            self._reject_dynamic_dims(
                op, tuple(reduced_shape), "keepdims reshape operand", op_name="reshape"
            )
            self._reject_dynamic_dims(
                op, tuple(op.result.type.shape), "keepdims reshape result", op_name="reshape"
            )
            reshaped = self._new_name()
            lines.append(
                f"{reshaped} = stablehlo.reshape {cur} : "
                f"({self._type_str(op.result.type.dtype, reduced_shape)}) -> "
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
        dynamic_reduced = tuple(
            x.type.shape[i] for i in dims
            if not isinstance(x.type.shape[i], (int, np.integer))
        )
        if dynamic_reduced:
            raise BackendError(
                f"stablehlo export: op 'reduce_mean'{self._loc(op)} reduces "
                f"over dynamic dims {dynamic_reduced!r} of shape "
                f"{tuple(x.type.shape)!r} — the element count cannot be "
                "computed statically; dynamic shapes are not supported by "
                "the StableHLO compiler backends in v1 (decompose it "
                "manually or use a future adapter)"
            )
        count = 1
        for i in dims:
            count *= x.type.shape[i]
        out_dtype = np.dtype(op.result.type.dtype)
        lines, sum_name, reduced_shape = self._emit_reduce_core(x, dims, "sum")
        converted = self._new_name()
        lines.append(
            f"{converted} = stablehlo.convert {sum_name} : "
            f"({self._type_str(x.type.dtype, reduced_shape)}) -> "
            f"{self._type_str(out_dtype, reduced_shape)}"
        )
        # The divisor constant is scalar; StableHLO elementwise ops require
        # equal shapes, so broadcast it when the reduced tensor is not.
        # A dynamic reduced shape (non-reduced symbolic dims) sources the
        # runtime output dimensions from the converted sum itself.
        count_name, count_lines = self._scalar_constant_for(
            out_dtype,
            count,
            reduced_shape,
            shape_source=(converted, out_dtype, reduced_shape),
        )
        lines.extend(count_lines)
        divided = self._new_name()
        lines.append(
            f"{divided} = stablehlo.divide {converted}, {count_name} : "
            f"{self._type_str(out_dtype, reduced_shape)}"
        )
        cur = divided
        if keepdims and tuple(op.result.type.shape) != tuple(reduced_shape):
            # The keepdims reshape must be static: a dynamic dim moved
            # between reduced/result shapes is invalid StableHLO reshape.
            self._reject_dynamic_dims(
                op, tuple(reduced_shape), "keepdims reshape operand", op_name="reshape"
            )
            self._reject_dynamic_dims(
                op, tuple(op.result.type.shape), "keepdims reshape result", op_name="reshape"
            )
            reshaped = self._new_name()
            lines.append(
                f"{reshaped} = stablehlo.reshape {cur} : "
                f"({self._type_str(out_dtype, reduced_shape)}) -> "
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
        # Region block args use FRESH counter names (%N), never %argN:
        # MLIR SSA names are function-scoped, so reusing %arg0 inside the
        # region collides with the enclosing function's entry arguments.
        arg_a = self._new_name()
        arg_b = self._new_name()
        region = (
            "({\n"
            f"  ^bb0({arg_a}: {elem_type}, {arg_b}: {elem_type}):\n"
            f"    {body_name} = stablehlo.{body_mnemonic} {arg_a}, {arg_b}"
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
            f"{init_name}) {region} {{dimensions = {self._i64_array(dims)}}}"
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
        # Fresh counter names — see _emit_reduce_core (never %argN, which
        # collides with the enclosing function's entry arguments).
        arg_a = self._new_name()
        arg_b = self._new_name()
        region = (
            "({\n"
            f"  ^bb0({arg_a}: {elem_type}, {arg_b}: {elem_type}):\n"
            f"    {body_name} = stablehlo.{body_mnemonic} {arg_a}, {arg_b}"
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

    # --- Indexing / sorting / tile / arg-reduce emitters ---
    #
    # Syntax notes (empirically verified at iree 3.11.0): `stablehlo.sort`
    # with multiple operands interleaves the comparator block args
    # (lhs0, rhs0, lhs1, rhs1 — per operand, lhs before rhs);
    # `stablehlo.reduce` with multiple operands GROUPS them
    # (operand0_lhs, operand1_lhs, operand0_rhs, operand1_rhs). The
    # comparator bodies follow the `_emit_reduce_core` region conventions
    # (fresh counter-named block args, `stablehlo.return`).

    @staticmethod
    def _normalize_axis(axis, rank: int, op_name: str) -> int:
        if rank == 0:
            return 0
        return int(axis) % rank

    def _emit_iota(self, shape: tuple, axis: int, lines: list) -> str:
        """Full-rank int64 iota along ``axis`` of the static ``shape``.

        Custom-form emission: iree 3.11.0 rejects the generic attribute
        form (``"stablehlo.iota" {dim = ...}`` — "expected '(' to start
        operand list"); ``stablehlo.iota dim = N : tensor<...>`` parses.
        """
        name = self._new_name()
        lines.append(
            f"{name} = stablehlo.iota dim = {axis} : "
            f"{self._type_str(np.dtype('int64'), shape)}"
        )
        return name

    def _emit_to_i64(self, name: str, dtype, shape: tuple, lines: list) -> str:
        """Convert an SSA value to int64 (identity when already int64)."""
        if np.dtype(dtype) == np.dtype("int64"):
            return name
        t = self._new_name()
        lines.append(
            f"{t} = stablehlo.convert {name} : ({self._type_str(dtype, shape)}) -> "
            f"{self._type_str(np.dtype('int64'), shape)}"
        )
        return t

    def _emit_index_normalize(self, iname: str, idx_shape: tuple, axis_len: int,
                              lines: list) -> str:
        """Normalize indices to numpy semantics: wrap negatives by adding
        the axis length (``select(idx < 0, idx + axis_len, idx)``) — numpy
        ``take``/``put_along_axis`` wrap negative indices while StableHLO
        treats out-of-range as poison."""
        i64 = np.dtype("int64")
        zero, zl = self._scalar_constant_for(i64, 0, idx_shape)
        lines.extend(zl)
        neg = self._new_name()
        lines.append(
            f'{neg} = "stablehlo.compare"({iname}, {zero}) '
            f"{{comparison_direction = #stablehlo<comparison_direction LT>}}"
            f" : ({self._type_str(i64, idx_shape)}, "
            f"{self._type_str(i64, idx_shape)}) -> "
            f"{self._type_str(np.dtype('bool'), idx_shape)}"
        )
        alen, al = self._scalar_constant_for(i64, axis_len, idx_shape)
        lines.extend(al)
        plus = self._new_name()
        lines.append(
            f"{plus} = stablehlo.add {iname}, {alen} : {self._type_str(i64, idx_shape)}"
        )
        wrap = self._new_name()
        lines.append(
            f"{wrap} = stablehlo.select {neg}, {plus}, {iname} : "
            f"({self._type_str(np.dtype('bool'), idx_shape)}, "
            f"{self._type_str(i64, idx_shape)}, {self._type_str(i64, idx_shape)})"
            f" -> {self._type_str(i64, idx_shape)}"
        )
        return wrap

    def _emit_broadcast_static(self, name: str, src_dtype, src_shape: tuple,
                               target_shape: tuple, lines: list, op: Op) -> str:
        """Numpy-broadcast an SSA value to a static target shape (trailing
        alignment, size-1 dims expand); BackendError when numpy broadcasting
        would fail."""
        if len(src_shape) > len(target_shape):
            raise BackendError(
                f"stablehlo export: op '{op.name}'{self._loc(op)} cannot "
                f"broadcast shape {src_shape!r} to {target_shape!r} (numpy "
                "broadcasting would fail)"
            )
        pad = len(target_shape) - len(src_shape)
        for sd, td in zip(src_shape, target_shape[pad:]):
            if sd != td and sd != 1:
                raise BackendError(
                    f"stablehlo export: op '{op.name}'{self._loc(op)} cannot "
                    f"broadcast shapes {src_shape!r} and {target_shape!r} "
                    "(numpy broadcasting would fail)"
                )
        if src_shape == target_shape:
            return name
        b = self._new_name()
        dims = list(range(pad, len(target_shape)))
        lines.append(
            f'{b} = "stablehlo.broadcast_in_dim"({name}) '
            f"{{broadcast_dimensions = {self._i64_array(dims)}}} : "
            f"({self._type_str(src_dtype, src_shape)}) -> "
            f"{self._type_str(src_dtype, target_shape)}"
        )
        return b

    def _emit_gather(self, op: Op) -> str:
        """``gather`` → ``stablehlo.gather`` with numpy ``take`` semantics
        (single gathered axis; result = x[:axis] + idx + x[axis+1:]).

        axis=0 is direct (offset_dims = [1..], collapsed slice dim 0,
        start_index_map [0], index_vector_dim = indices rank); a general
        static axis transposes the operand so the axis moves to 0, gathers,
        and transposes back (exact for static shapes). Indices are
        converted to i64, 0-d index tensors (the etl.scan counter pattern)
        are promoted to (1,), and negative indices are wrapped by adding
        the axis length (np.take semantics — StableHLO out-of-range is
        poison). Multi-axis gather defers exactly like the numpy kernel."""
        x, idx = op.operands
        axes = op.attributes.get("axes", (0,))
        if isinstance(axes, int):
            axes = (axes,)
        rank = x.type.rank
        normalized = sorted(
            {self._normalize_axis(a, rank, "gather.axes") for a in axes}
        )
        if len(normalized) != 1:
            raise BackendError(
                f"stablehlo export: op 'gather'{self._loc(op)} over axes "
                f"{tuple(normalized)} is not supported (multi-axis gather "
                "has no StableHLO emission; the numpy kernel defers it too)"
            )
        axis = normalized[0]
        x_shape = tuple(x.type.shape)
        idx_shape = tuple(idx.type.shape)
        self._reject_dynamic_dims(op, x_shape, "operand shape")
        self._reject_dynamic_dims(op, idx_shape, "indices shape")
        lines = []
        i64 = np.dtype("int64")
        # Indices → i64 (0-d promoted to (1,)).
        iname = self._name(idx)
        if np.dtype(idx.type.dtype) != i64:
            t = self._new_name()
            lines.append(
                f"{t} = stablehlo.convert {iname} : ({self._vt(idx.type)}) -> "
                f"{self._type_str(i64, idx_shape)}"
            )
            iname = t
        idx_was_scalar = idx_shape == ()
        if idx_was_scalar:
            t = self._new_name()
            lines.append(
                f"{t} = stablehlo.reshape {iname} : (tensor<i64>) -> tensor<1xi64>"
            )
            iname = t
            idx_shape = (1,)
        # Negative wrap (skipped for constant indices with no negatives —
        # the common static-index case).
        if idx.defining_op is not None and idx.defining_op.name == "constant":
            vals = np.asarray(idx.defining_op.attributes["value"])
            needs_wrap = bool(vals.size) and np.any(vals < 0)
        else:
            needs_wrap = True
        if needs_wrap:
            iname = self._emit_index_normalize(iname, idx_shape, x_shape[axis], lines)
        # Append the index-vector dim: (idx_shape..., 1).
        m = len(idx_shape)
        iv_shape = idx_shape + (1,)
        iv = self._new_name()
        lines.append(
            f"{iv} = stablehlo.reshape {iname} : "
            f"({self._type_str(i64, idx_shape)}) -> {self._type_str(i64, iv_shape)}"
        )
        # Move the gathered axis to 0 when needed.
        if axis == 0:
            gx_name = self._name(x)
            gx_shape = x_shape
        else:
            perm = [axis] + [i for i in range(rank) if i != axis]
            gx_shape = (x_shape[axis],) + tuple(
                d for i, d in enumerate(x_shape) if i != axis
            )
            gx_name = self._new_name()
            lines.append(
                f'{gx_name} = "stablehlo.transpose"({self._name(x)}) '
                f"{{permutation = {self._i64_array(perm)}}} : "
                f"({self._vt(x.type)}) -> {self._type_str(x.type.dtype, gx_shape)}"
            )
        rest = gx_shape[1:]
        offset_dims = list(range(m, m + len(rest)))
        gres_shape = tuple(idx_shape) + tuple(rest)
        gres = self._new_name()
        lines.append(
            f'{gres} = "stablehlo.gather"({gx_name}, {iv}) '
            f"{{dimension_numbers = #stablehlo.gather<offset_dims = "
            f"[{self._int_list(offset_dims)}], collapsed_slice_dims = [0], "
            f"start_index_map = [0], index_vector_dim = {m}>, "
            f"slice_sizes = {self._i64_array([1] + list(rest))}}} : "
            f"({self._type_str(x.type.dtype, gx_shape)}, "
            f"{self._type_str(i64, iv_shape)}) -> "
            f"{self._type_str(x.type.dtype, gres_shape)}"
        )
        cur = gres
        if axis != 0:
            perm_back = (
                list(range(m, m + axis))
                + list(range(m))
                + list(range(m + axis, m + len(rest)))
            )
            tshape = x_shape[:axis] + tuple(idx_shape) + x_shape[axis + 1 :]
            cur = self._new_name()
            lines.append(
                f'{cur} = "stablehlo.transpose"({gres}) '
                f"{{permutation = {self._i64_array(perm_back)}}} : "
                f"({self._type_str(x.type.dtype, gres_shape)}) -> "
                f"{self._type_str(x.type.dtype, tshape)}"
            )
        if idx_was_scalar:
            src = cur
            cur = self._new_name()
            lines.append(
                f"{cur} = stablehlo.reshape {src} : "
                f"({self._type_str(x.type.dtype, tshape if axis != 0 else gres_shape)}) -> "
                f"{self._vt(op.result.type)}"
            )
        self._names[id(op.result)] = cur
        return "\n".join(lines)

    def _emit_scatter(self, op: Op) -> str:
        """``scatter`` → ``stablehlo.scatter`` with numpy ``put_along_axis``
        semantics (functional; the kernel copies the operand first).

        The kernel reshapes lower-rank indices to full rank
        ``(1,)*axis + idx.shape + (1,)*(rank-axis-1)`` and broadcasts them
        against the updates; every broadcast position p writes
        ``out[p[:axis], idx[p], p[axis+1:]] = updates[p]``. v1 supports:
        (a) rank-1 axis=0 (scalar-window form: any 0-d/1-d indices, updates
        broadcast to (K,)); (b) rank-2 axis=0 row scatter (indices 0-d /
        (1,) / (K,) / (K,1), updates broadcast to (K, D) — the
        update-window form, used by the etl.scan stack pattern and NSGA2
        crowding distance); (c) rank-2 axis=1 via a transpose of all
        operands into (b). The general multi-row broadcast (indices and
        updates both with >1 varying dims) has no single-scatter form —
        explicit BackendError (never silent). Duplicate index targets:
        numpy resolves last-write-wins, StableHLO leaves the result
        undefined — document, don't guess.
        """
        x, idx, upd = op.operands
        rank = x.type.rank
        axis = self._normalize_axis(op.attributes.get("axis", 0), rank, "scatter.axis")
        x_shape = tuple(x.type.shape)
        idx_shape = tuple(idx.type.shape)
        upd_shape = tuple(upd.type.shape)
        self._reject_dynamic_dims(op, x_shape, "operand shape")
        self._reject_dynamic_dims(op, idx_shape, "indices shape")
        self._reject_dynamic_dims(op, upd_shape, "updates shape")
        if rank == 1 and axis == 0:
            return self._emit_scatter_rank1(op, x, idx, upd, x_shape, idx_shape, upd_shape)
        if rank == 2 and axis == 0:
            return self._emit_scatter_row(op, x, idx, upd, x_shape, idx_shape, upd_shape)
        if rank == 2 and axis == 1:
            # Transpose all operands: axis 1 of x becomes axis 0 of x^T.
            lines = []
            xt = self._new_name()
            lines.append(
                f'{xt} = "stablehlo.transpose"({self._name(x)}) '
                f"{{permutation = array<i64: 1, 0>}} : ({self._vt(x.type)}) -> "
                f"{self._type_str(x.type.dtype, (x_shape[1], x_shape[0]))}"
            )
            ut = self._new_name()
            if upd_shape == (x_shape[0],):
                lines.append(
                    f"{ut} = stablehlo.reshape {self._name(upd)} : "
                    f"({self._vt(upd.type)}) -> "
                    f"{self._type_str(upd.type.dtype, (1, x_shape[0]))}"
                )
                ut_shape = (1, x_shape[0])
            else:
                lines.append(
                    f'{ut} = "stablehlo.transpose"({self._name(upd)}) '
                    f"{{permutation = array<i64: 1, 0>}} : ({self._vt(upd.type)}) -> "
                    f"{self._type_str(upd.type.dtype, (upd_shape[1], upd_shape[0]))}"
                )
                ut_shape = (upd_shape[1], upd_shape[0])
            idx_shape_t = self._transposed_idx_shape(idx_shape)
            iname = self._name(idx)
            iname = self._emit_to_i64(iname, idx.type.dtype, idx_shape, lines)
            iname = self._emit_index_normalize(iname, idx_shape, x_shape[1], lines)
            if idx_shape != idx_shape_t:
                t = self._new_name()
                lines.append(
                    f"{t} = stablehlo.reshape {iname} : "
                    f"({self._type_str(np.dtype('int64'), idx_shape)}) -> "
                    f"{self._type_str(np.dtype('int64'), idx_shape_t)}"
                )
                iname = t
            body = self._emit_scatter_row_impl(
                op, xt, x.type.dtype, idx, ut, ut_shape, (x_shape[1], x_shape[0]),
                idx_shape_t, self._type_str(x.type.dtype, (x_shape[1], x_shape[0])),
                iname=iname, upd_dtype=upd.type.dtype,
            )
            lines.extend(body)
            out = self._new_name()
            lines.append(
                f'{out} = "stablehlo.transpose"({self._scatter_row_result}) '
                f"{{permutation = array<i64: 1, 0>}} : "
                f"({self._type_str(x.type.dtype, (x_shape[1], x_shape[0]))}) -> "
                f"{self._vt(op.result.type)}"
            )
            self._names[id(op.result)] = out
            return "\n".join(lines)
        raise BackendError(
            f"stablehlo export: op 'scatter'{self._loc(op)} on a "
            f"rank-{rank} operand along axis {axis} is not supported by the "
            "v1 exporter (numpy put_along_axis broadcast scatter is "
            "expressible for rank-1 axis=0 and rank-2 row/column writes; "
            "use the numpy backend for the general case)"
        )

    @staticmethod
    def _transposed_idx_shape(idx_shape: tuple) -> tuple:
        """The axis=1 full-rank index grid (1,K) reshaped to (K,1) — the
        same values, interpreted column-major (see _emit_scatter)."""
        if idx_shape in ((), (1,)):
            return (1, 1)
        if len(idx_shape) == 1:  # (K,)
            return (idx_shape[0], 1)
        if idx_shape[0] == 1:  # (1, K) — kernel full-rank form for axis=1
            return (idx_shape[1], 1)
        return idx_shape  # (K, 1) passes through

    def _emit_scatter_rank1(self, op, x, idx, upd, x_shape, idx_shape, upd_shape) -> str:
        """Rank-1 axis=0 scalar-window scatter: indices (K,) or 0-d, updates
        broadcast to (K,) — the general 1-d put_along_axis."""
        i64 = np.dtype("int64")
        lines = []
        if idx_shape == ():
            iname = self._name(idx)
            iname = self._emit_to_i64(iname, idx.type.dtype, (), lines)
            src = iname
            iname = self._new_name()
            lines.append(
                f"{iname} = stablehlo.reshape {src} : (tensor<i64>) -> tensor<1xi64>"
            )
            k = 1
        else:
            k = idx_shape[0]
            iname = self._name(idx)
            iname = self._emit_to_i64(iname, idx.type.dtype, (k,), lines)
            iname = self._emit_index_normalize(iname, (k,), x_shape[0], lines)
        # updates → (K,), converted to the operand dtype.
        uname = self._name(upd)
        ushape = upd_shape
        if upd_shape == ():
            t = self._new_name()
            lines.append(
                f"{t} = stablehlo.reshape {uname} : ({self._vt(upd.type)}) -> "
                f"tensor<1x{self._elem_dtype_str(upd.type.dtype)}>"
            )
            uname = t
            ushape = (1,)
        if np.dtype(upd.type.dtype) != np.dtype(x.type.dtype):
            t = self._new_name()
            lines.append(
                f"{t} = stablehlo.convert {uname} : "
                f"({self._type_str(upd.type.dtype, ushape)}) -> "
                f"{self._type_str(x.type.dtype, ushape)}"
            )
            uname = t
        uname = self._emit_broadcast_static(
            uname, x.type.dtype, ushape, (k,), lines, op
        )
        iv = self._new_name()
        lines.append(
            f"{iv} = stablehlo.reshape {iname} : (tensor<{k}xi64>) -> "
            f"tensor<{k}x1xi64>"
        )
        elem = self._elem_type(x.type.dtype)
        upd_arg, cur_v = self._new_name(), self._new_name()
        region = (
            "({\n"
            f"  ^bb0({cur_v}: {elem}, {upd_arg}: {elem}):\n"
            f"    stablehlo.return {upd_arg} : {elem}\n"
            "  })"
        )
        result_name = self._bind_results(op)[0]
        lines.append(
            f'{result_name} = "stablehlo.scatter"({self._name(x)}, {iv}, {uname}) '
            f"{region} "
            f"{{scatter_dimension_numbers = #stablehlo.scatter<"
            f"update_window_dims = [], inserted_window_dims = [0], "
            f"scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}} : "
            f"({self._vt(x.type)}, tensor<{k}x1xi64>, "
            f"tensor<{k}x{self._elem_dtype_str(x.type.dtype)}>) -> "
            f"{self._vt(op.result.type)}"
        )
        self._names[id(op.result)] = result_name
        return "\n".join(lines)

    def _emit_scatter_row(self, op, x, idx, upd, x_shape, idx_shape, upd_shape) -> str:
        """Rank-2 axis=0 row scatter via the update-window form."""
        lines = []
        d = x_shape[1]
        # Full-rank index grid: 0-d/(1,)/(K,)/(K,1) → (K,1).
        if idx_shape == ():
            iname = self._name(idx)
            iname = self._emit_to_i64(iname, idx.type.dtype, (), lines)
            src = iname
            iname = self._new_name()
            lines.append(
                f"{iname} = stablehlo.reshape {src} : (tensor<i64>) -> tensor<1x1xi64>"
            )
            k = 1
        elif idx_shape == (1,):
            iname = self._name(idx)
            iname = self._emit_to_i64(iname, idx.type.dtype, (1,), lines)
            src = iname
            iname = self._new_name()
            lines.append(
                f"{iname} = stablehlo.reshape {src} : (tensor<1xi64>) -> tensor<1x1xi64>"
            )
            k = 1
        elif len(idx_shape) == 1:
            k = idx_shape[0]
            iname = self._name(idx)
            iname = self._emit_to_i64(iname, idx.type.dtype, (k,), lines)
            iname = self._emit_index_normalize(iname, (k,), x_shape[0], lines)
            t = self._new_name()
            lines.append(
                f"{t} = stablehlo.reshape {iname} : (tensor<{k}xi64>) -> "
                f"tensor<{k}x1xi64>"
            )
            iname = t
        elif idx_shape == (1, 1):
            iname = self._name(idx)
            iname = self._emit_to_i64(iname, idx.type.dtype, (1, 1), lines)
            iname = self._emit_index_normalize(iname, (1, 1), x_shape[0], lines)
            k = 1
        elif len(idx_shape) == 2 and idx_shape[1] == 1:
            k = idx_shape[0]
            iname = self._name(idx)
            iname = self._emit_to_i64(iname, idx.type.dtype, idx_shape, lines)
            iname = self._emit_index_normalize(iname, idx_shape, x_shape[0], lines)
        else:
            raise BackendError(
                f"stablehlo export: op 'scatter'{self._loc(op)} with indices "
                f"shape {idx_shape!r} on a rank-2 operand is not supported "
                "(v1 supports 0-d/(1,)/(K,)/(K,1) indices — the etl.scan and "
                "NSGA2 patterns; use the numpy backend for the general case)"
            )
        body = self._emit_scatter_row_impl(
            op, self._name(x), x.type.dtype, idx, upd, upd_shape, x_shape, (k, 1),
            self._vt(op.result.type), iname=iname
        )
        lines.extend(body)
        self._names[id(op.result)] = self._scatter_row_result
        return "\n".join(lines)

    def _emit_scatter_row_impl(self, op, x_name, x_dtype, idx, upd, upd_shape,
                               x_shape, idx_shape, result_type, iname=None,
                               upd_dtype=None) -> list:
        """Shared row-window scatter body (rank-2 axis=0): indices (K,1)
        [given as ``iname`` or emitted from ``idx``], updates broadcast to
        (K, D), scatter with update_window_dims=[1], inserted=[0],
        scatter_dims_to_operand_dims=[0], index_vector_dim=1. ``x_name`` /
        ``x_dtype`` name the operand (a transpose alias in the axis=1
        path), ``result_type`` the scatter result's MLIR type. ``upd`` is
        either the updates Value (its SSA name emitted here) or an
        already-materialized SSA name (the axis=1 transpose path) — in the
        latter case ``upd_dtype`` supplies the dtype (``upd.type`` does not
        exist on a str). The update_computation region implements the
        frontend's copy-and-assign (``put_along_axis``) semantics: return
        the updates value (replace, never accumulate). Returns the lines;
        the result name lands in ``self._scatter_row_result``."""
        lines = []
        i64 = np.dtype("int64")
        k, d = idx_shape[0], x_shape[1]
        if iname is None:
            iname = self._name(idx)
            iname = self._emit_to_i64(iname, idx.type.dtype, idx_shape, lines)
            iname = self._emit_index_normalize(iname, idx_shape, x_shape[0], lines)
        # updates → (K, D), converted to the operand dtype.
        uname = upd if isinstance(upd, str) else self._name(upd)
        udtype = np.dtype(upd_dtype if upd_dtype is not None else upd.type.dtype)
        ushape = upd_shape
        if upd_shape == (x_shape[0],):
            t = self._new_name()
            lines.append(
                f"{t} = stablehlo.reshape {uname} : ({self._type_str(udtype, upd_shape)}) -> "
                f"{self._type_str(udtype, (1, x_shape[0]))}"
            )
            uname, ushape = t, (1, x_shape[0])
        if udtype != np.dtype(x_dtype):
            t = self._new_name()
            lines.append(
                f"{t} = stablehlo.convert {uname} : "
                f"({self._type_str(udtype, ushape)}) -> "
                f"{self._type_str(x_dtype, ushape)}"
            )
            uname = t
        uname = self._emit_broadcast_static(
            uname, x_dtype, ushape, (k, d), lines, op
        )
        elem = self._elem_type(x_dtype)
        upd_arg, cur_v = self._new_name(), self._new_name()
        region = (
            "({\n"
            f"  ^bb0({cur_v}: {elem}, {upd_arg}: {elem}):\n"
            f"    stablehlo.return {upd_arg} : {elem}\n"
            "  })"
        )
        result_name = self._bind_results(op)[0]
        lines.append(
            f'{result_name} = "stablehlo.scatter"({x_name}, {iname}, {uname}) '
            f"{region} "
            f"{{scatter_dimension_numbers = #stablehlo.scatter<"
            f"update_window_dims = [1], inserted_window_dims = [0], "
            f"scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}} : "
            f"({self._type_str(x_dtype, x_shape)}, {self._type_str(i64, idx_shape)}, "
            f"{self._type_str(x_dtype, (k, d))}) -> {result_type}"
        )
        self._scatter_row_result = result_name
        return lines

    def _emit_sort(self, op: Op) -> str:
        """``sort`` → ``stablehlo.sort`` with an LT comparator region;
        descending = ``stablehlo.reverse`` after (the numpy composition).
        Equal elements are indistinguishable in the output, so the
        ``stable`` attr only fixes tie ORDER for argsort (via the iota
        tie-break) — here it is irrelevant. Bool operands sort via an i8
        key (StableHLO compare on i1 only supports EQ/NE)."""
        x = op.operands[0]
        rank = x.type.rank
        axis = self._normalize_axis(op.attributes.get("axis", -1), rank, "sort.axis")
        descending = bool(op.attributes.get("descending", False))
        x_shape = tuple(x.type.shape)
        self._reject_dynamic_dims(op, x_shape, "operand shape")
        lines = []
        keys_dtype = np.dtype(x.type.dtype)
        key_name = self._name(x)
        if keys_dtype.kind == "b":
            t = self._new_name()
            lines.append(
                f"{t} = stablehlo.convert {key_name} : "
                f"({self._type_str(keys_dtype, x_shape)}) -> "
                f"{self._type_str(np.dtype('int8'), x_shape)}"
            )
            key_name, keys_dtype = t, np.dtype("int8")
        elem = self._elem_type(keys_dtype)
        a1, a2 = self._new_name(), self._new_name()
        cmp = self._new_name()
        region = (
            "({\n"
            f"  ^bb0({a1}: {elem}, {a2}: {elem}):\n"
            f'    {cmp} = "stablehlo.compare"({a1}, {a2}) '
            f"{{comparison_direction = #stablehlo<comparison_direction LT>}}"
            f" : ({elem}, {elem}) -> tensor<i1>\n"
            f"    stablehlo.return {cmp} : tensor<i1>\n"
            "  })"
        )
        s = self._new_name()
        lines.append(
            f'{s} = "stablehlo.sort"({key_name}) {region} '
            f"{{dimension = {axis} : i64}} : "
            f"({self._type_str(keys_dtype, x_shape)}) -> "
            f"{self._type_str(keys_dtype, x_shape)}"
        )
        cur = s
        if keys_dtype.kind != np.dtype(x.type.dtype).kind:
            t = self._new_name()
            lines.append(
                f"{t} = stablehlo.convert {cur} : "
                f"({self._type_str(keys_dtype, x_shape)}) -> "
                f"{self._type_str(x.type.dtype, x_shape)}"
            )
            cur = t
        if descending:
            t = self._new_name()
            lines.append(
                f'{t} = "stablehlo.reverse"({cur}) '
                f"{{dimensions = {self._i64_array([axis])}}} : "
                f"({self._vt(op.result.type)}) -> {self._vt(op.result.type)}"
            )
            cur = t
        self._names[id(op.result)] = cur
        return "\n".join(lines)

    def _emit_argsort(self, op: Op) -> str:
        """``argsort`` → pair sort of (value, full-rank iota) with the
        stable tie-break comparator ``(ka < kb) OR (ka == kb AND ia < ib)``
        — numpy stable semantics; descending = reverse of the index column
        (the numpy composition). Bool keys sort via an i8 conversion."""
        x = op.operands[0]
        rank = x.type.rank
        axis = self._normalize_axis(op.attributes.get("axis", -1), rank, "argsort.axis")
        descending = bool(op.attributes.get("descending", False))
        x_shape = tuple(x.type.shape)
        self._reject_dynamic_dims(op, x_shape, "operand shape")
        lines = []
        keys_dtype = np.dtype(x.type.dtype)
        key_name = self._name(x)
        if keys_dtype.kind == "b":
            t = self._new_name()
            lines.append(
                f"{t} = stablehlo.convert {key_name} : "
                f"({self._type_str(keys_dtype, x_shape)}) -> "
                f"{self._type_str(np.dtype('int8'), x_shape)}"
            )
            key_name, keys_dtype = t, np.dtype("int8")
        sv, si = self._emit_stable_argsort(key_name, keys_dtype, x_shape, axis, lines)
        cur = si
        if descending:
            t = self._new_name()
            lines.append(
                f'{t} = "stablehlo.reverse"({cur}) '
                f"{{dimensions = {self._i64_array([axis])}}} : "
                f"({self._type_str(np.dtype('int64'), x_shape)}) -> "
                f"{self._type_str(np.dtype('int64'), x_shape)}"
            )
            cur = t
        self._names[id(op.result)] = cur
        return "\n".join(lines)

    def _emit_stable_argsort(self, keys_name: str, keys_dtype, keys_shape: tuple,
                             axis: int, lines: list) -> tuple[str, str]:
        """Pair-sort (keys, iota) ascending with the stable tie-break
        comparator (iota order on equal keys — numpy stable semantics);
        returns the sorted-keys column name AND the iota-column name (both
        ``keys_shape``; the iota column is int64). Shared by the ``argsort``
        op, the permutation expansion, and the ``eigh`` eigenvalue sort.

        Emission mode (writer option ``sort_emission``): ``"pair"`` (this
        two-operand sort — iree-cuda cannot bufferize multi-operand sorts
        whenever the sorted-axis extent is >= 32, upstream iree 3.11.0
        bug), ``"count"`` (the count-based O(n^2) composition with NO sort
        op — bit-exact vs numpy on both llvm-cpu and cuda), or ``"auto"``
        (per call: ``"count"`` when the sorted-axis extent >= 32, else
        ``"pair"``). The iree adapter defaults to ``"auto"``."""
        if self._count_argsort_active(keys_shape, axis):
            return self._emit_count_argsort(
                keys_name, keys_dtype, keys_shape, axis, lines
            )
        key_elem = self._elem_type(keys_dtype)
        i64_type = self._type_str(np.dtype("int64"), keys_shape)
        iota = self._emit_iota(keys_shape, axis, lines)
        ka, kb, ia, ib = (self._new_name() for _ in range(4))
        c1, c2, c3, c4, c5 = (self._new_name() for _ in range(5))
        region = (
            "({\n"
            f"  ^bb0({ka}: {key_elem}, {kb}: {key_elem}, "
            f"{ia}: tensor<i64>, {ib}: tensor<i64>):\n"
            f'    {c1} = "stablehlo.compare"({ka}, {kb}) '
            f"{{comparison_direction = #stablehlo<comparison_direction LT>}}"
            f" : ({key_elem}, {key_elem}) -> tensor<i1>\n"
            f'    {c2} = "stablehlo.compare"({ka}, {kb}) '
            f"{{comparison_direction = #stablehlo<comparison_direction EQ>}}"
            f" : ({key_elem}, {key_elem}) -> tensor<i1>\n"
            f'    {c3} = "stablehlo.compare"({ia}, {ib}) '
            f"{{comparison_direction = #stablehlo<comparison_direction LT>}}"
            f" : (tensor<i64>, tensor<i64>) -> tensor<i1>\n"
            f"    {c4} = stablehlo.and {c2}, {c3} : tensor<i1>\n"
            f"    {c5} = stablehlo.or {c1}, {c4} : tensor<i1>\n"
            f"    stablehlo.return {c5} : tensor<i1>\n"
            "  })"
        )
        key_type = self._type_str(keys_dtype, keys_shape)
        sv, si = self._new_name(), self._new_name()
        lines.append(
            f'{sv}, {si} = "stablehlo.sort"({keys_name}, {iota}) {region} '
            f"{{dimension = {axis} : i64}} : ({key_type}, {i64_type}) -> "
            f"({key_type}, {i64_type})"
        )
        return sv, si

    # --- eigh (while-loop cyclic-Jacobi) ---
    def _count_argsort_active(self, keys_shape: tuple, axis: int) -> bool:
        """Whether the count-based argsort emission applies for this call
        under the configured ``sort_emission`` mode."""
        mode = self._sort_emission
        if mode == "count":
            return True
        if mode == "pair":
            return False
        extent = keys_shape[axis]
        return _is_static_dim(extent) and extent >= 32

    def _emit_count_argsort(self, keys_name: str, keys_dtype, keys_shape: tuple,
                            axis: int, lines: list) -> tuple[str, str]:
        """Count-based STABLE argsort — NO ``stablehlo.sort`` at all.

        iree 3.11.0 cannot bufferize multi-operand ``stablehlo.sort`` on
        cuda whenever the sorted-axis extent is >= 32 (upstream bug; every
        iota dtype / is_stable / operand-order variant fails — llvm-cpu is
        unaffected, 1-operand sorts are unaffected). This composition
        computes the stable rank of every element by counting, then
        inverts the rank permutation:

          1. ``pos[j] = #{k: x[k] < x[j]} + #{k: x[k] == x[j] and k < j}``
             (two broadcast comparisons + an iota ``k < j`` tie-break + a
             reduce over k) — pos is the STABLE rank permutation;
          2. ``idx[p] = min_j (j if pos[j] == p else n)`` (iota grids, an
             EQ mask, a select with the sentinel ``n`` = sorted-axis
             extent, a reduce-min over the e axis) — the argsort index
             array, bit-exact vs ``np.argsort(kind="stable")``;
          3. sorted keys (consumed by the ``eigh`` emission) via a masked
             select of the values + a reduce-min over the same mask
             (sentinel ``+inf`` / ``iinfo.max`` — pos is a bijection, so
             exactly one element survives the mask).

        Probe-validated bit-exact on llvm-cpu AND cuda at rank 1/2, axes
        0/1, n<o and n>o, ascending and descending (descending = a
        ``stablehlo.reverse`` of the index column, the numpy composition).
        NO ``stablehlo.transpose`` of computed values anywhere (iree
        llvm-cpu miscompiles that pattern in this context — measured).
        """
        rank = len(keys_shape)
        n = int(keys_shape[axis])
        i64 = np.dtype("int64")
        keys_t = self._type_str(keys_dtype, keys_shape)
        out_t = self._type_str(i64, keys_shape)
        # mid space (k, d0..d_{R-1}): k at 0, the original dims shifted +1.
        mid_shape = (n,) + tuple(keys_shape)
        midk_t = self._type_str(keys_dtype, mid_shape)
        mid_t = self._type_str(i64, mid_shape)
        midb_t = self._type_str(np.dtype("bool"), mid_shape)
        nn_t = self._type_str(i64, (n, n))
        # xj[k, ...] = x[..., k as the sorted axis]; xk[k, ...] = x[...] —
        # so compare(xj, xk) is x[k] < x[j] at every (k, j) pair.
        xk_dims = list(range(1, rank + 1))
        xj_dims = [0 if d == axis else d + 1 for d in range(rank)]
        xk = self._new_name()
        lines.append(
            f"{xk} = stablehlo.broadcast_in_dim {keys_name}, dims = "
            f"{self._list_attr(xk_dims)} : ({keys_t}) -> {midk_t}"
        )
        xj = self._new_name()
        lines.append(
            f"{xj} = stablehlo.broadcast_in_dim {keys_name}, dims = "
            f"{self._list_attr(xj_dims)} : ({keys_t}) -> {midk_t}"
        )
        less = self._new_name()
        lines.append(
            f'{less} = "stablehlo.compare"({xj}, {xk}) '
            f"{{comparison_direction = #stablehlo<comparison_direction LT>}}"
            f" : ({midk_t}, {midk_t}) -> {midb_t}"
        )
        eq = self._new_name()
        lines.append(
            f'{eq} = "stablehlo.compare"({xj}, {xk}) '
            f"{{comparison_direction = #stablehlo<comparison_direction EQ>}}"
            f" : ({midk_t}, {midk_t}) -> {midb_t}"
        )
        # Tie-break: k at mid dim 0, j at mid dim axis+1 (the sorted-axis
        # position in the mid space).
        ik = self._emit_iota((n, n), 0, lines)
        ij = self._emit_iota((n, n), 1, lines)
        ikb = self._new_name()
        lines.append(
            f"{ikb} = stablehlo.broadcast_in_dim {ik}, dims = "
            f"{self._list_attr([0, axis + 1])} : ({nn_t}) -> {mid_t}"
        )
        ijb = self._new_name()
        lines.append(
            f"{ijb} = stablehlo.broadcast_in_dim {ij}, dims = "
            f"{self._list_attr([0, axis + 1])} : ({nn_t}) -> {mid_t}"
        )
        kltj = self._new_name()
        lines.append(
            f'{kltj} = "stablehlo.compare"({ikb}, {ijb}) '
            f"{{comparison_direction = #stablehlo<comparison_direction LT>}}"
            f" : ({mid_t}, {mid_t}) -> {midb_t}"
        )
        eqbef = self._new_name()
        lines.append(f"{eqbef} = stablehlo.and {eq}, {kltj} : {midb_t}")
        cl = self._new_name()
        lines.append(f"{cl} = stablehlo.convert {less} : ({midb_t}) -> {mid_t}")
        ce = self._new_name()
        lines.append(f"{ce} = stablehlo.convert {eqbef} : ({midb_t}) -> {mid_t}")
        zero = self._new_name()
        lines.append(f"{zero} = stablehlo.constant dense<0> : tensor<i64>")
        suml = self._emit_count_reduce(cl, mid_t, out_t, [0], zero, lines)
        sume = self._emit_count_reduce(ce, mid_t, out_t, [0], zero, lines)
        pos = self._new_name()
        lines.append(f"{pos} = stablehlo.add {suml}, {sume} : {out_t}")
        # Inverse space (d0..d_{axis-1}, p, d_{axis+1}..d_{R-1}, e): the
        # sorted axis is replaced by the rank p; e is appended at the end.
        inv_shape = keys_shape[:axis] + (n,) + keys_shape[axis + 1 :] + (n,)
        inv_t = self._type_str(i64, inv_shape)
        invb_t = self._type_str(np.dtype("bool"), inv_shape)
        pos_dims = list(range(rank))
        pos_dims[axis] = rank  # pos's sorted axis → the appended e slot
        posb = self._new_name()
        lines.append(
            f"{posb} = stablehlo.broadcast_in_dim {pos}, dims = "
            f"{self._list_attr(pos_dims)} : ({out_t}) -> {inv_t}"
        )
        ri = self._emit_iota((n, n), 0, lines)
        ji = self._emit_iota((n, n), 1, lines)
        rib = self._new_name()
        lines.append(
            f"{rib} = stablehlo.broadcast_in_dim {ri}, dims = "
            f"{self._list_attr([axis, rank])} : ({nn_t}) -> {inv_t}"
        )
        jib = self._new_name()
        lines.append(
            f"{jib} = stablehlo.broadcast_in_dim {ji}, dims = "
            f"{self._list_attr([axis, rank])} : ({nn_t}) -> {inv_t}"
        )
        pmask = self._new_name()
        lines.append(
            f'{pmask} = "stablehlo.compare"({posb}, {rib}) '
            f"{{comparison_direction = #stablehlo<comparison_direction EQ>}}"
            f" : ({inv_t}, {inv_t}) -> {invb_t}"
        )
        nc = self._new_name()
        lines.append(f"{nc} = stablehlo.constant dense<{n}> : tensor<i64>")
        nbig = self._new_name()
        lines.append(
            f"{nbig} = stablehlo.broadcast_in_dim {nc}, dims = "
            f"{self._list_attr([])} : (tensor<i64>) -> {inv_t}"
        )
        jsel = self._new_name()
        lines.append(
            f"{jsel} = stablehlo.select {pmask}, {jib}, {nbig} : "
            f"({invb_t}, {inv_t}, {inv_t}) -> {inv_t}"
        )
        idx = self._emit_count_reduce_min(
            jsel, inv_t, out_t, [rank], nc, self._elem_type(i64), lines
        )
        # Sorted keys (consumed by the eigh emission; argsort/permutation
        # ignore them): masked select of the values + reduce-min.
        if keys_dtype.kind == "f":
            sentinel = np.inf
        elif keys_dtype.kind in "iu":
            sentinel = np.iinfo(keys_dtype).max
        else:  # complex/bool keys: the LT compare above is already invalid
            # for complex (same failure as the pair emission); emit a
            # placeholder so the composition stays well-formed.
            sentinel = 0
        invk_t = self._type_str(keys_dtype, inv_shape)
        wb = self._new_name()
        lines.append(
            f"{wb} = stablehlo.broadcast_in_dim {keys_name}, dims = "
            f"{self._list_attr(pos_dims)} : ({keys_t}) -> {invk_t}"
        )
        sent_const = self._count_sentinel_constant(sentinel, keys_dtype, lines)
        sbig = self._new_name()
        lines.append(
            f"{sbig} = stablehlo.broadcast_in_dim {sent_const}, dims = "
            f"{self._list_attr([])} : "
            f"({self._elem_type(keys_dtype)}) -> {invk_t}"
        )
        wsel = self._new_name()
        lines.append(
            f"{wsel} = stablehlo.select {pmask}, {wb}, {sbig} : "
            f"({invb_t}, {invk_t}, {invk_t}) -> {invk_t}"
        )
        sv = self._emit_count_reduce_min(
            wsel,
            invk_t,
            keys_t,
            [rank],
            sent_const,
            self._elem_type(keys_dtype),
            lines,
        )
        return sv, idx

    def _count_sentinel_constant(self, value, dtype, lines: list) -> str:
        """A scalar constant of `value` in `dtype` (fresh name + line)."""
        name = self._new_name()
        lines.append(
            f"{name} = stablehlo.constant "
            f"{self._constant_text(np.asarray(value, dtype=dtype))} : "
            f"{self._elem_type(dtype)}"
        )
        return name

    def _emit_count_reduce(self, operand_name: str, operand_t: str, out_t: str,
                           dims: list, init_name: str, lines: list) -> str:
        """i64 add-reduce (the count composition's rank sum)."""
        a1, a2 = self._new_name(), self._new_name()
        s = self._new_name()
        region = (
            "({\n"
            f"  ^bb0({a1}: tensor<i64>, {a2}: tensor<i64>):\n"
            f"    {s} = stablehlo.add {a1}, {a2} : tensor<i64>\n"
            f"    stablehlo.return {s} : tensor<i64>\n"
            "  })"
        )
        r = self._new_name()
        lines.append(
            f'{r} = "stablehlo.reduce"({operand_name}, {init_name}) {region} '
            f"{{dimensions = {self._i64_array(dims)}}} : "
            f"({operand_t}, tensor<i64>) -> {out_t}"
        )
        return r

    def _emit_count_reduce_min(self, operand_name: str, operand_t: str, out_t: str,
                               dims: list, init_name: str, elem_t: str,
                               lines: list) -> str:
        """min-reduce (the inverse-permutation step); ``elem_t`` is the
        scalar element type of the operand (``tensor<i64>`` for the index
        column, the keys scalar type for the sorted-keys step)."""
        a1, a2 = self._new_name(), self._new_name()
        mn = self._new_name()
        region = (
            "({\n"
            f"  ^bb0({a1}: {elem_t}, {a2}: {elem_t}):\n"
            f"    {mn} = stablehlo.minimum {a1}, {a2} : {elem_t}\n"
            f"    stablehlo.return {mn} : {elem_t}\n"
            "  })"
        )
        r = self._new_name()
        lines.append(
            f'{r} = "stablehlo.reduce"({operand_name}, {init_name}) {region} '
            f"{{dimensions = {self._i64_array(dims)}}} : "
            f"({operand_t}, {elem_t}) -> {out_t}"
        )
        return r


    #: Adaptive cyclic-Jacobi sweep count for the ``eigh`` composition (see
    #: ``_eigh_sweeps``). Each sweep zeroes every off-diagonal (p, q) once;
    #: the off-diagonal norm converges quadratically with a sharp knee
    #: (measured on the numpy spike, 10x10 fp32: sweep 4 → 1.1e-5, sweep 6 →
    #: 3.7e-7). The sweep loop runs inside a ``stablehlo.while`` at runtime
    #: (``_emit_eigh``), so the count costs O(1) MLIR text per extra sweep —
    #: it only trades runtime against accuracy, never compile time.
    _EIGH_SWEEPS_F64 = 8

    @staticmethod
    def _eigh_sweeps(n: int, dtype) -> int:
        """Cyclic-Jacobi sweep count for the while-loop ``eigh`` composition.

        fp32: dim-based schedule (n ≤ 5 → 3, n = 6 → 4, 7 ≤ n ≤ 15 → 5,
        16 ≤ n ≤ 30 → 6, 31 ≤ n ≤ 50 → 7, n > 50 → 8), validated by a
        bit-exact numpy simulation of the emission's rotation scheme
        (worst-over-20-matrices reconstruction AND eigenvalue error
        ≤ 1e-3 with ≥ 2× margin on CMAES-like PD / rank-deficient /
        ill-conditioned families; the fp32 knee is flat above the minimum —
        e.g. dim 10: rec 1.18e-5 at 5 sweeps, identical to 8). The existing
        parity tests pass with margin at every size (3×3/10×10/batched f32,
        int32→f64). f64 keeps 8: it needs ≥ the fp32 counts for its tighter
        (1e-12) targets (3×3 needs 4, dim 10 needs 7, dim ≥ 20 needs 8).
        """
        if np.dtype(dtype) != np.dtype("float32"):
            return Writer._EIGH_SWEEPS_F64
        if n <= 5:
            return 3
        if n == 6:
            return 4
        if n <= 15:
            return 5
        if n <= 30:
            return 6
        if n <= 50:
            return 7
        return 8

    def _emit_eigh(self, op: Op) -> str:
        """``eigh`` → while-loop cyclic-Jacobi symmetric eigensolver.

        StableHLO 1.0 removed ``stablehlo.eigh``/``qr``/``svd`` (iree 3.11
        ships only ``cholesky`` of the factorizations), so the symmetric
        eigendecomposition is composed from v1 ops: a ``stablehlo.while``
        loop carrying ``(a, v, p, q, k)`` applies
        ``_eigh_sweeps(n, dtype) × n(n−1)/2`` cyclic (p, q) rotations —
        each zeroing the (p, q) off-diagonal with the textbook rotation
        ``J = [[c, s], [-s, c]]`` via ``B = A·J`` (column updates),
        ``A' = J^T·B`` (row updates), ``V = V·J`` (column updates; V starts
        as the identity). The carried pair counter walks the sweep order
        (0,1), (0,2), …, (n−2, n−1) and ``k`` counts total rotations (cond
        ``k < total``). With the ``eigh_early_exit`` exporter option (bool,
        default True) the loop additionally carries an i1 ``done`` flag:
        at every sweep boundary a nested ``stablehlo.if`` checks the
        scale-aware relative off-diagonal energy of the current A (masked
        squared Frobenius vs the full squared energy, per-matrix with an
        AND across the batch) and sets ``done`` once it is below the dtype
        tolerance (f32 tol 3e-5, f64 tol 1e-13 — CALIBRATED f32 value, see
        the tol comment at the constant below). The loop then exits by
        CLAMPING the
        rotation counter: when ``done`` fires the body returns
        ``k := total`` (a scalar ``select`` on the post-increment counter),
        so the next cond check ``k < total`` fails — converged matrices
        skip their remaining scheduled sweeps (exit only after a completed
        sweep — the final scheduled sweep always completes). The exit is
        deliberately NOT expressed in the cond region (``and(k < total,
        not(done))``): using a loop-carried i1 through ``not``/``and`` in
        the cond crashes iree-compile 3.11's AffinityAnalysis
        deterministically (measured 0/10 SIGSEGV on 3x3 f32, same
        use-graph-topology family as the while SEGV lore in CONTEXT.md),
        while the k-clamp is 10/10 clean and keeps the cond region
        byte-identical to the pre-option text. The cond region stays
        exactly ``k < total`` in BOTH modes. ``False`` emits the exact
        pre-option 5-carry text (the A/B measurement lever). Dynamic (p, q)
        indexing uses gathers (never
        ``stablehlo.slice`` — its start indices must be static), and every
        constant/iota is hoisted OUTSIDE the loop (iree crashes on
        constants inside while bodies — ``util.global.load`` undefined
        ``@__hoisted_*``; see Known Issues). The emitted text is O(1) in n
        and in the sweep count — the unrolled predecessor's O(n²·sweeps)
        text (10x10 ~17 s build) is gone, so dim ≥ 25 compiles (the dense
        eye constant for the V init already made the text O(n²) in the
        matrix data; the early-exit i≠j mask keeps it O(n²)). A stable
        pair-sort then orders the final diagonal ASCENDING (numpy
        ``linalg.eigh`` contract) and a gather reorders the V columns by
        the sorted permutation. Every rotation is applied with full-matrix
        selects (mask = ``iota == index``), never scatter regions.

        Dtype rule mirrors the numpy kernel: int/bool → float64 (converted
        first); float32/float64 pass through; complex and float16 defer
        with an explicit BackendError (complex beyond cast is not in v1;
        f16 has no defined parity). Dynamic dims → explicit BackendError.

        Parity note: LAPACK-based numpy vs the Jacobi composition agree to
        fp32 tolerance, NOT bit-exact. The rotation sequence and arithmetic
        are identical to the old unrolled emission (same ops, same order —
        slices become gathers, fresh constants become hoisted names), so
        the numerics are bit-for-bit unchanged. Jacobi converges
        off-diagonals to the fp32 floor (measured ~4e-7 for 10x10 in 8
        sweeps), eigenvalue ordering among (near-)degenerate values is
        fixed by the sort, and the eigenvectors of degenerate eigenspaces
        are basis-dependent — tests compare reconstruction
        ``A ≈ v diag(w) v^T`` and column orthogonality, never elementwise
        v.
        """
        x = op.operands[0]
        x_shape = tuple(x.type.shape)
        self._reject_dynamic_dims(op, x_shape, "operand shape")
        rank = len(x_shape)
        if rank < 2:
            raise BackendError(
                f"stablehlo export: op 'eigh'{self._loc(op)} requires "
                f"rank >= 2, got rank {rank}"
            )
        n = x_shape[-1]
        if x_shape[-2] != n:
            raise BackendError(
                f"stablehlo export: op 'eigh'{self._loc(op)} requires a "
                f"square matrix, got {x_shape[-2]} vs {n}"
            )
        in_dtype = np.dtype(x.type.dtype)
        if in_dtype.kind == "c":
            raise BackendError(
                f"stablehlo export: op 'eigh'{self._loc(op)} over complex "
                "dtype is not supported in v1 — complex-number computation "
                "beyond cast is deferred"
            )
        if in_dtype.kind in "biu":
            dtype = np.dtype("float64")
        elif in_dtype in (np.dtype("float32"), np.dtype("float64")):
            dtype = in_dtype
        else:
            raise BackendError(
                f"stablehlo export: op 'eigh'{self._loc(op)} over dtype "
                f"{in_dtype!r} is not supported in v1 (float32/float64 "
                "inputs; int/bool upcast to float64)"
            )
        batch = x_shape[:-2]
        i64 = np.dtype("int64")
        lines = []
        # int/bool → float64 (numpy linalg upcast rule).
        a = self._name(x)
        if dtype != in_dtype:
            t = self._new_name()
            lines.append(
                f"{t} = stablehlo.convert {a} : "
                f"({self._type_str(in_dtype, x_shape)}) -> "
                f"{self._type_str(dtype, x_shape)}"
            )
            a = t
        # --- loop-invariant setup (hoisted OUTSIDE the while — iree
        # crashes on constants/iotas defined inside while bodies, see
        # Known Issues) ---
        # Layout: the carried matrices live TRANSPOSED as (n, n, batch...)
        # (batch dims trailing — the `_emit_diagonal_extract` pattern).
        # The runtime-index gathers then have STATIC trailing batch offset
        # dims, so the indices are tiny (2,)/(1,) tensors of the loop
        # carries (p, q) — no per-batch index arithmetic inside the loop.
        t_shape = (n, n) + batch
        # Hoisted index masks: iota along the transposed matrix axes,
        # compared against rank-0 broadcasts of the carried (p, q) per
        # iteration.
        iota_row = self._emit_iota(t_shape, 0, lines)
        iota_col = self._emit_iota(t_shape, 1, lines)
        # The carried A, transposed into that layout (identity for
        # batch = (), where the (n, n) block already sits at dims 0,1; the
        # matrix data stays bit-identical because A remains symmetric
        # through the rotations).
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
        # V starts as the identity, in the same transposed layout. V is NOT
        # symmetric, so the data transpose is real at every rank (the same
        # permutation formula covers rank 2: [1, 0]) — the body's dim-0
        # "row gathers" then address V's columns (and A's columns, which
        # coincide with A's rows while A stays symmetric).
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
        # Rotation-formula scalar constants (batch,)-shaped.
        c_two, extra = self._scalar_constant_for(dtype, 2.0, batch)
        lines.extend(extra)
        c_one, extra = self._scalar_constant_for(dtype, 1.0, batch)
        lines.extend(extra)
        c_neg_one, extra = self._scalar_constant_for(dtype, -1.0, batch)
        lines.extend(extra)
        c_zero, extra = self._scalar_constant_for(dtype, 0.0, batch)
        lines.extend(extra)
        # Rank-0 i64 loop constants: counter init / pair-step / bound.
        i_zero, extra = self._scalar_constant_for(i64, 0, ())
        lines.extend(extra)
        i_one, extra = self._scalar_constant_for(i64, 1, ())
        lines.extend(extra)
        i_two, extra = self._scalar_constant_for(i64, 2, ())
        lines.extend(extra)
        i_n_minus_1, extra = self._scalar_constant_for(i64, n - 1, ())
        lines.extend(extra)
        i_n_minus_2, extra = self._scalar_constant_for(i64, n - 2, ())
        lines.extend(extra)
        total = self._eigh_sweeps(n, dtype) * n * (n - 1) // 2
        i_total, extra = self._scalar_constant_for(i64, total, ())
        lines.extend(extra)
        # --- eigh_early_exit: sweep-boundary convergence check ------------
        # When enabled (the default) the loop carries an extra i1 ``done``
        # flag: at every sweep boundary (k % rps == 0, rps = n(n-1)/2
        # rotations per sweep) a nested ``stablehlo.if`` computes the
        # scale-aware relative off-diagonal energy of the current A and
        # sets ``done`` once it is below the dtype tolerance; when ``done``
        # fires the body CLAMPS k to the rotation total, so the (unchanged)
        # cond ``k < total`` exits — converged matrices skip their
        # remaining scheduled sweeps. All constants/masks are hoisted
        # OUTSIDE the loop (constants inside while regions crash iree —
        # ``util.global.load`` undefined ``@__hoisted_*``). The option
        # defaults True; ``False`` emits the exact pre-option text.
        early_exit = self._eigh_early_exit
        if early_exit:
            i1 = np.dtype("bool")
            # The (n, n) i != j mask as a 0/1 matrix in the computation
            # dtype — multiplying a*a by it keeps exactly the off-diagonal
            # energy (exact for 0/1 factors; equivalent to the select
            # formulation without a zero operand inside the loop).
            m_neq = self._new_name()
            lines.append(
                f"{m_neq} = stablehlo.constant "
                f"{self._constant_text(np.logical_not(np.eye(n, dtype=bool)).astype(dtype))}"
                f" : {self._type_str(dtype, (n, n))}"
            )
            if batch:  # broadcast to the carried (n, n, batch...) layout
                m_neq = self._bcast(m_neq, dtype, (n, n), t_shape, [0, 1], lines)
            # Rotations per sweep (the remainder divisor) and the squared
            # dtype tolerance. f32 tol CALIBRATED to 3e-5 (WS-C task 2
            # step B, measured on iree-llvm-cpu 3.11 — calibration data in
            # /tmp/eigh_calib.py; parity home:
            # tests/backends/test_iree_eigh_diag_parity.py): the full parity
            # suite stays 9/9 green with fat margins at every looser
            # candidate (3e-5 keeps dim25/batched-3x3 on the FULL schedule,
            # identical to the pre-option text), while 3e-5 fires the exit
            # at sweep ~4 of 7 on dim50 sample-covariance-like matrices
            # (measured 11-21% savings; 1e-5 fires sweep ~5/6 inconsistently,
            # 1e-6 essentially never). 1e-4 was REJECTED: no extra dim50
            # savings, and parity-family matrices start exiting early
            # (dim25 reconstruction error grows ~10x, batched 3x3 ~22x).
            # f64 stays 1e-13 (fixed 8-sweep schedule — parity passes).
            # tol_sq is
            # broadcast to (batch,) when batched (the per-matrix sq terms
            # are (batch,)-shaped).
            tol = 3e-5 if dtype == np.dtype("float32") else 1e-13
            i_rps, extra = self._scalar_constant_for(i64, n * (n - 1) // 2, ())
            lines.extend(extra)
            tol_sq, extra = self._scalar_constant_for(dtype, tol * tol, batch)
            lines.extend(extra)
            # Rank-0 inits: the done carry (false), the and-reduce init
            # (true), and the sum-reduce init (zero).
            b_false, extra = self._scalar_constant_for(i1, False, ())
            lines.extend(extra)
            b_true, extra = self._scalar_constant_for(i1, True, ())
            lines.extend(extra)
            z_zero, extra = self._scalar_constant_for(dtype, 0.0, ())
            lines.extend(extra)
        # --- the while loop: carries (a, v, p, q, k[, done]) --------------
        a_type = self._type_str(dtype, t_shape)
        i64_scalar = self._elem_type(i64)
        i1_scalar = self._elem_type(np.dtype("bool"))
        arg_types = [a_type, a_type, i64_scalar, i64_scalar, i64_scalar]
        if early_exit:
            arg_types.append(i1_scalar)
        carry_types = "(" + ", ".join(arg_types) + ")"
        n_carries = len(arg_types)
        # Cond region: k < total — ALWAYS, in both modes. The early-exit
        # `done` flag never appears here: a loop-carried i1 through
        # not/and in the cond crashes iree-compile 3.11's AffinityAnalysis
        # (measured 0/10 SIGSEGV); the body instead clamps k to total when
        # done fires (see below), so this cond region is byte-identical to
        # the pre-option text. No closing brace — `_emit_while` style.
        cond_args = [self._new_name() for _ in range(n_carries)]
        cond_lines = []
        cnd = self._cmp(cond_args[4], i_total, "LT", i64, (), cond_lines)
        cond_region = (
            "  ^bb0("
            + ", ".join(f"{nm} : {tp}" for nm, tp in zip(cond_args, arg_types))
            + "):\n"
            + "\n".join("    " + line for line in cond_lines)
            + f"\n    stablehlo.return {cnd} : tensor<i1>"
        )
        # Body region: one rotation on (p, q), then the pair/counter step.
        body_args = [self._new_name() for _ in range(n_carries)]
        body_lines = []
        a2, v2 = self._emit_jacobi_rotation_loop_body(
            body_args[0], body_args[1], body_args[2], body_args[3],
            n, batch, dtype, x_shape, iota_row, iota_col,
            c_two, c_one, c_neg_one, c_zero, body_lines,
        )
        # k' = k+1; if q == n-1: (p', q') = (p+1, p+2) else (p, q+1).
        # The pair (n-2, n-1) ends a sweep: the NEXT pair must wrap to
        # (0, 1) — without the sweep-end reset the pair walks out of range
        # ((n-1, n)) after the FIRST sweep's last pair, corrupting gathers.
        # (k' == total only at the end of the LAST sweep.)
        k2 = self._ew("add", body_args[4], i_one, i64, (), body_lines)
        q_last = self._cmp(body_args[3], i_n_minus_1, "EQ", i64, (), body_lines)
        p_next = self._ew("add", body_args[2], i_one, i64, (), body_lines)
        q_after = self._ew("add", body_args[2], i_two, i64, (), body_lines)
        q_inc = self._ew("add", body_args[3], i_one, i64, (), body_lines)
        p_end = self._cmp(body_args[2], i_n_minus_2, "EQ", i64, (), body_lines)
        sweep_end = self._ew("and", p_end, q_last, np.dtype("bool"), (), body_lines)
        p2 = self._sel(q_last, p_next, body_args[2], i64, (), body_lines)
        q2 = self._sel(q_last, q_after, q_inc, i64, (), body_lines)
        p2 = self._sel(sweep_end, i_zero, p2, i64, (), body_lines)
        q2 = self._sel(sweep_end, i_one, q2, i64, (), body_lines)
        if early_exit:
            # Sweep-boundary convergence check (nested stablehlo.if); the
            # pair counter's own sweep_end (above) is the (p, q) wrap
            # condition — this is the ROTATION-count sweep boundary
            # (k2 % rps == 0), which fires at the same iterations.
            done2 = self._emit_eigh_sweep_check(
                body_lines, a2, body_args[5], k2, dtype, batch, t_shape,
                m_neq, tol_sq, z_zero, b_true, i_rps, i_zero, i1_scalar,
            )
            # Exit via the k-clamp: when done fires, return k := total so
            # the NEXT cond check (k < total) fails — the loop stops one
            # rotation after convergence is detected, exactly like a cond
            # on `done` would, but WITHOUT touching the loop-carried i1 in
            # the cond region (iree AffinityAnalysis SEGV, see the docstring
            # and the cond comment above). O(1) scalar op per rotation.
            k2 = self._sel(done2, i_total, k2, i64, (), body_lines)
        ret_values = [a2, v2, p2, q2, k2]
        if early_exit:
            ret_values.append(done2)
        body_region = (
            "  ^bb0("
            + ", ".join(f"{nm} : {tp}" for nm, tp in zip(body_args, arg_types))
            + "):\n"
            + "\n".join("    " + line for line in body_lines)
            + f"\n    stablehlo.return {', '.join(ret_values)} : "
            + ", ".join(arg_types)
        )
        inits = [a, v, i_zero, i_one, i_zero]
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
        # The carried matrices are in the transposed layout — transpose
        # back to (batch..., n, n) for the diagonal extract and the column
        # reorder. The inverse of the init perm [rank-1, rank-2]+[0..rank-3]
        # is [2..rank-1] + [1, 0] (batch dims in order, MATRIX DIMS
        # REVERSED — both are involutions on the matrix block). A's
        # back-transpose is only needed for batch dims (rank 2: a stayed
        # (n, n) throughout); V's is real at every rank (rank 2: [1, 0] —
        # the same formula).
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
        sv, si = self._emit_stable_argsort(w, dtype, w_shape, len(w_shape) - 1, lines)
        vs = self._emit_reorder_columns(v, si, dtype, x_shape, n, lines)
        self._names[id(op.results[0])] = sv
        self._names[id(op.results[1])] = vs
        return "\n".join(lines)

    def _emit_jacobi_rotation_loop_body(self, a_name, v_name, p_name, q_name,
                                        n, batch, dtype, x_shape, iota_row,
                                        iota_col, c_two, c_one, c_neg_one,
                                        c_zero, lines):
        """One while-body cyclic-Jacobi rotation zeroing ``A[p, q]``
        (``p < q``) with ``J = [[c, s], [-s, c]]``: ``B = A·J`` (column
        updates), ``A' = J^T·B`` (row updates), ``V = V·J`` (column
        updates). The carried matrices are in the TRANSPOSED layout
        ``(n, n, batch...)`` (``t_shape``; see ``_emit_eigh``), the (p, q)
        pair arrives as RUNTIME rank-0 i64 loop scalars, and all indexing
        goes through ``stablehlo.gather`` (``stablehlo.slice`` requires
        static start indices). The batch dims are STATIC trailing offset
        dims of every gather, so the indices are batch-free ``(2,)``
        (scalars) / ``(1,)`` (rows/columns) tensors. In this layout the
        "column p of A" values ``A[b, i, p]`` sit at ``AT[i, p, b]`` (a
        dim-1 gather, applied to the ROW dim of the mask selects), and the
        "row p" values at ``BT[p, j, b]`` (a dim-0 gather, applied to the
        COLUMN dim). The arithmetic sequence is identical to the
        pre-while unrolled rotation (same op order, same formulas — only
        slice→gather, fresh-constant→hoisted-name, and the static layout
        transpose differ), so results are bit-identical to the old
        emission. Per-batch-element scalars ``c``/``s`` come from the
        textbook formulas ``tau = (a_qq - a_pp)/(2 a_pq)``,
        ``t = sign(tau)/(|tau| + sqrt(1 + tau^2))`` (``sign(0) = +1`` —
        the equal-diagonal case needs the 45° rotation, not no-op),
        ``c = 1/sqrt(1 + t^2)``, ``s = t c``, guarded by ``a_pq == 0``
        (``c = 1, s = 0``; also shields the ``0/0 -> NaN`` degenerate case
        — the guard select happens AFTER the formulas, so the NaN is never
        propagated). Returns ``(new_a, new_v)`` SSA names (still in the
        transposed layout)."""
        i64 = np.dtype("int64")
        k = len(batch)
        t_shape = (n, n) + batch
        vec_shape = (n,) + batch
        # --- per-iteration index tensors (batch-free — the batch dims are
        # static offset dims; the pair (p, q) is shared across the batch) --
        idx_pp = self._index_pair(p_name, p_name, lines)
        idx_qq = self._index_pair(q_name, q_name, lines)
        idx_pq = self._index_pair(p_name, q_name, lines)
        idx_p = self._index_single(p_name, lines)
        idx_q = self._index_single(q_name, lines)
        # --- per-batch-element scalars: apq, app, aqq ---------------------
        apq = self._gather_scalar(a_name, idx_pq, dtype, n, batch, lines)
        app = self._gather_scalar(a_name, idx_pp, dtype, n, batch, lines)
        aqq = self._gather_scalar(a_name, idx_qq, dtype, n, batch, lines)
        # --- c, s from the textbook formulas (hoisted constants) ----------
        t1 = self._ew("subtract", aqq, app, dtype, batch, lines)
        t2 = self._ew("multiply", c_two, apq, dtype, batch, lines)
        tau = self._ew("divide", t1, t2, dtype, batch, lines)
        tau2 = self._ew("multiply", tau, tau, dtype, batch, lines)
        oneptau2 = self._ew("add", c_one, tau2, dtype, batch, lines)
        sq = self._ew("sqrt", oneptau2, None, dtype, batch, lines)
        abst = self._ew("abs", tau, None, dtype, batch, lines)
        denom = self._ew("add", abst, sq, dtype, batch, lines)
        ge0 = self._cmp(tau, c_zero, "GE", dtype, batch, lines)
        sgn = self._sel(ge0, c_one, c_neg_one, dtype, batch, lines)
        t = self._ew("divide", sgn, denom, dtype, batch, lines)
        t2 = self._ew("multiply", t, t, dtype, batch, lines)
        onept2 = self._ew("add", c_one, t2, dtype, batch, lines)
        sqt = self._ew("sqrt", onept2, None, dtype, batch, lines)
        c_calc = self._ew("divide", c_one, sqt, dtype, batch, lines)
        s_calc = self._ew("multiply", t, c_calc, dtype, batch, lines)
        apq_ne0 = self._cmp(apq, c_zero, "NE", dtype, batch, lines)
        c = self._sel(apq_ne0, c_calc, c_one, dtype, batch, lines)
        s = self._sel(apq_ne0, s_calc, c_zero, dtype, batch, lines)
        # Broadcasts with EXPLICIT dimension mappings (the transposed
        # layout (n, n, batch...) — numpy trailing alignment does not
        # apply, the batch dims stay trailing):
        #   c/s (batch,) → (n, batch...):      batch dims → 1..k
        #   col vec (n, batch...) → (n,n,batch...): n → ROW dim (0)
        #   row vec (n, batch...) → (n,n,batch...): n → COLUMN dim (1)
        c_b = self._bcast(c, dtype, batch, vec_shape, list(range(1, k + 1)), lines)
        s_b = self._bcast(s, dtype, batch, vec_shape, list(range(1, k + 1)), lines)
        col_dims = [0] + list(range(2, k + 2))
        row_dims = [1] + list(range(2, k + 2))
        # --- rotation masks (iota vs rank-0 broadcast of the pair) --------
        p_x = self._bcast(p_name, i64, (), t_shape, [], lines)
        q_x = self._bcast(q_name, i64, (), t_shape, [], lines)
        row_p = self._cmp(iota_row, p_x, "EQ", i64, t_shape, lines)
        row_q = self._cmp(iota_row, q_x, "EQ", i64, t_shape, lines)
        col_p = self._cmp(iota_col, p_x, "EQ", i64, t_shape, lines)
        col_q = self._cmp(iota_col, q_x, "EQ", i64, t_shape, lines)
        # --- B = A·J: column updates (dim-1 gathers of the transposed A) --
        acp = self._gather_vec(a_name, idx_p, "col", dtype, n, batch, lines)
        acq = self._gather_vec(a_name, idx_q, "col", dtype, n, batch, lines)
        ncp = self._ew("subtract", self._ew("multiply", c_b, acp, dtype, vec_shape, lines),
                       self._ew("multiply", s_b, acq, dtype, vec_shape, lines),
                       dtype, vec_shape, lines)
        ncq = self._ew("add", self._ew("multiply", s_b, acp, dtype, vec_shape, lines),
                       self._ew("multiply", c_b, acq, dtype, vec_shape, lines),
                       dtype, vec_shape, lines)
        b = self._sel(col_p, self._bcast(ncp, dtype, vec_shape, t_shape, col_dims, lines), a_name, dtype, t_shape, lines)
        b = self._sel(col_q, self._bcast(ncq, dtype, vec_shape, t_shape, col_dims, lines), b, dtype, t_shape, lines)
        # --- A' = J^T·B: row updates (dim-0 gathers of the updated B) -----
        brp = self._gather_vec(b, idx_p, "row", dtype, n, batch, lines)
        brq = self._gather_vec(b, idx_q, "row", dtype, n, batch, lines)
        nrp = self._ew("subtract", self._ew("multiply", c_b, brp, dtype, vec_shape, lines),
                       self._ew("multiply", s_b, brq, dtype, vec_shape, lines),
                       dtype, vec_shape, lines)
        nrq = self._ew("add", self._ew("multiply", s_b, brp, dtype, vec_shape, lines),
                       self._ew("multiply", c_b, brq, dtype, vec_shape, lines),
                       dtype, vec_shape, lines)
        a2 = self._sel(row_p, self._bcast(nrp, dtype, vec_shape, t_shape, row_dims, lines), b, dtype, t_shape, lines)
        a2 = self._sel(row_q, self._bcast(nrq, dtype, vec_shape, t_shape, row_dims, lines), a2, dtype, t_shape, lines)
        # --- V = V·J: column updates (dim-0 gathers of the transposed V) --
        vcp = self._gather_vec(v_name, idx_p, "row", dtype, n, batch, lines)
        vcq = self._gather_vec(v_name, idx_q, "row", dtype, n, batch, lines)
        nvp = self._ew("subtract", self._ew("multiply", c_b, vcp, dtype, vec_shape, lines),
                       self._ew("multiply", s_b, vcq, dtype, vec_shape, lines),
                       dtype, vec_shape, lines)
        nvq = self._ew("add", self._ew("multiply", s_b, vcp, dtype, vec_shape, lines),
                       self._ew("multiply", c_b, vcq, dtype, vec_shape, lines),
                       dtype, vec_shape, lines)
        v2 = self._sel(row_p, self._bcast(nvp, dtype, vec_shape, t_shape, row_dims, lines), v_name, dtype, t_shape, lines)
        v2 = self._sel(row_q, self._bcast(nvq, dtype, vec_shape, t_shape, row_dims, lines), v2, dtype, t_shape, lines)
        return a2, v2

    def _emit_eigh_sweep_check(self, body_lines, a2, done, k2, dtype, batch,
                               t_shape, mask, tol_sq, z_zero, b_true,
                               i_rps, i_zero, i1_scalar) -> str:
        """Emit the sweep-boundary convergence check for the ``eigh``
        early exit (``eigh_early_exit`` exporter option; called from
        ``_emit_eigh``) and return the NEW ``done`` carry SSA name.

        The check runs ONLY at sweep boundaries — inside a nested
        ``stablehlo.if`` guarded by ``sweep_end = (k2 % rps == 0)``, where
        ``rps = n(n-1)/2`` is the hoisted rotations-per-sweep constant and
        ``k2 = k+1 >= 1`` (so the positive-multiple test alone excludes
        the k == 0 first iteration) — keeping the per-rotation cost at one
        scalar predicate + one scalar if pass-through (the false branch
        returns the carried ``done`` unchanged), while the O(n^2) work
        executes once per executed sweep (~1 extra dispatch per sweep vs
        the 45/1225 rotations per sweep). The condition is SCALE-AWARE and
        relative: ``done = done OR sq_offdiag <= tol_sq * sq_total`` with
        ``sq_offdiag`` the squared Frobenius energy of the off-diagonal
        part (``a*a`` masked by the hoisted 0/1 i != j matrix), ``sq_total``
        the full squared energy, and ``tol_sq`` the hoisted squared dtype
        tolerance (f32 tol 3e-5, f64 tol 1e-13 — f32 CALIBRATED in WS-C
        task 2 step B: parity suite 9/9 green with margin, dim50 cov-like
        fires sweep ~4 for 11-21% savings; 1e-4 rejected — see the tol
        comment in ``_emit_eigh``). No sqrt: the comparison is done on
        the squared energies. ``converged`` is evaluated PER MATRIX (the
        matrix dims 0,1 of the transposed layout are reduced first, giving
        a (batch,) result) and AND-reduced across the batch — the loop
        carries ONE shared (p, q) pair for the whole batch, so the exit
        must wait for the SLOWEST batch element (a whole-tensor sum could
        stop while a low-energy batch element is still unconverged). For
        batch = () this reduces exactly to the scalar comparison. The if
        regions reference only body/captured values (never constants
        defined inside a region — iree constant-hoisting breaks).

        Lines are appended to ``body_lines`` with their region indentation
        baked relative to the while-body base indent (the ``_emit_eigh``
        body join adds the base 4 spaces per element): the if line at +0,
        region headers at +2, region ops at +4.
        """
        i64 = np.dtype("int64")
        # Sweep boundary: k2 (post-increment) is a positive multiple of rps.
        rem = self._ew("remainder", k2, i_rps, i64, (), body_lines)
        sweep_end = self._cmp(rem, i_zero, "EQ", i64, (), body_lines)
        # The nested if's TRUE-region ops (flat single lines — assembled
        # below at +4 relative to the if line).
        ops: list[str] = []
        a2sq = self._ew("multiply", a2, a2, dtype, t_shape, ops)
        masked = self._ew("multiply", a2sq, mask, dtype, t_shape, ops)
        b_shape = tuple(batch)
        sq_off = self._emit_eigh_reduce(ops, masked, dtype, t_shape,
                                        b_shape, [0, 1], z_zero)
        sq_tot = self._emit_eigh_reduce(ops, a2sq, dtype, t_shape,
                                        b_shape, [0, 1], z_zero)
        thresh = self._ew("multiply", sq_tot, tol_sq, dtype, b_shape, ops)
        conv_e = self._cmp(sq_off, thresh, "LE", dtype, b_shape, ops)
        conv = conv_e
        if b_shape:  # AND-reduce the per-matrix flags across the batch
            conv = self._emit_eigh_reduce(
                ops, conv_e, np.dtype("bool"), b_shape, (),
                list(range(len(b_shape))), b_true,
            )
        done_or = self._ew("or", done, conv, np.dtype("bool"), (), ops)
        # The if RESULT is a distinct fresh name (the or inside the true
        # region already holds ``done_or`` — reusing the same name across
        # the region boundary is legal shadowing but confusing).
        done2 = self._new_name()
        # Assemble the nested if.
        body_lines.append(f'{done2} = "stablehlo.if"({sweep_end}) ({{')
        body_lines.append("  ^bb0:")
        body_lines.extend("    " + line for line in ops)
        body_lines.append(f"    stablehlo.return {done_or} : {i1_scalar}")
        body_lines.append("  }, {")
        body_lines.append("  ^bb0:")
        body_lines.append(f"    stablehlo.return {done} : {i1_scalar}")
        body_lines.append(f"  }}) : ({i1_scalar}) -> {i1_scalar}")
        return done2

    def _emit_eigh_reduce(self, lines, operand, dtype, src_shape, out_shape,
                          dims, init_name) -> str:
        """One ``stablehlo.reduce`` for the eigh convergence check,
        emitting every physical line separately (no embedded newlines — the
        caller's uniform per-element indentation then applies to the whole
        op). ``dtype`` selects the reducer: add (float energies) or and
        (i1 batch flags). Returns the result SSA name."""
        elem_t = self._elem_type(dtype)
        r = self._new_name()
        a1, a2 = self._new_name(), self._new_name()
        mnemonic = "add" if np.dtype(dtype) != np.dtype("bool") else "and"
        lines.append(f'{r} = "stablehlo.reduce"({operand}, {init_name}) ({{')
        lines.append(f"  ^bb0({a1}: {elem_t}, {a2}: {elem_t}):")
        s = self._new_name()
        lines.append(f"    {s} = stablehlo.{mnemonic} {a1}, {a2} : {elem_t}")
        lines.append(f"    stablehlo.return {s} : {elem_t}")
        lines.append(
            f"  }}) {{dimensions = {self._i64_array(dims)}}} : "
            f"({self._type_str(dtype, src_shape)}, {elem_t}) -> "
            f"{self._type_str(dtype, out_shape)}"
        )
        return r

    def _index_single(self, idx_name, lines) -> str:
        """Rank-0 i64 scalar → a ``(1,)`` i64 index tensor — the index
        vector (``index_vector_dim = 0``) for a single-axis gather on the
        transposed-layout matrix. Batch-free: the batch dims of the operand
        are static trailing offset dims, so the index carries only the
        runtime (p, q)."""
        i64 = np.dtype("int64")
        b = self._new_name()
        lines.append(
            f"{b} = stablehlo.reshape {idx_name} : "
            f"({self._elem_type(i64)}) -> tensor<1xi64>"
        )
        return b

    def _index_pair(self, p_name, q_name, lines) -> str:
        """Two rank-0 i64 scalars → a ``(2,)`` i64 index tensor (two
        single-index reshapes concatenated along dim 0) — the index vector
        for a 2-axis (row, col) scalar gather on the transposed-layout
        matrix."""
        i64 = np.dtype("int64")
        p1 = self._index_single(p_name, lines)
        q1 = self._index_single(q_name, lines)
        idx = self._new_name()
        lines.append(
            f'{idx} = "stablehlo.concatenate"({p1}, {q1}) '
            f"{{dimension = 0 : i64}} : "
            f"(tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>"
        )
        return idx

    def _gather_scalar(self, src, idx_name, dtype, n, batch, lines) -> str:
        """``A[..., p, q]`` with RUNTIME (p, q) → ``(batch...,)``: a
        two-axis gather collapsing both matrix dims of the transposed
        ``(n, n, batch...)`` operand. The indices are a batch-free ``(2,)``
        pair tensor with ``index_vector_dim = 0``; the batch dims come
        back as trailing offset dims (the ``_emit_diagonal_extract``
        pattern — ``stablehlo.slice`` needs static start indices, so
        loop-carried pairs must gather)."""
        k = len(batch)
        g = self._new_name()
        lines.append(
            f'{g} = "stablehlo.gather"({src}, {idx_name}) '
            f"{{dimension_numbers = #stablehlo.gather<offset_dims = "
            f"[{self._int_list(range(k))}], "
            f"collapsed_slice_dims = [0, 1], start_index_map = [0, 1], "
            f"index_vector_dim = 0>, "
            f"slice_sizes = {self._i64_array([1, 1] + list(batch))}}} : "
            f"({self._type_str(dtype, (n, n) + batch)}, "
            f"tensor<2xi64>) -> {self._type_str(dtype, batch)}"
        )
        return g

    def _gather_vec(self, src, idx_name, which, dtype, n, batch, lines) -> str:
        """A single axis of the transposed-layout matrix with a RUNTIME
        index → ``(n, batch...)``: ``which="row"`` gathers ``[p, :, b...]``
        (collapses dim 0, the A-row/B-row index), ``which="col"`` gathers
        ``[:, p, b...]`` (collapses dim 1, the A-column index). The
        indices are a batch-free ``(1,)`` tensor with
        ``index_vector_dim = 0``; the n dim is the leading offset dim and
        the batch dims follow as trailing offset dims."""
        k = len(batch)
        if which == "row":
            collapsed, mapped = 0, 0
            slice_sizes = [1, n] + list(batch)
        else:
            collapsed, mapped = 1, 1
            slice_sizes = [n, 1] + list(batch)
        g = self._new_name()
        lines.append(
            f'{g} = "stablehlo.gather"({src}, {idx_name}) '
            f"{{dimension_numbers = #stablehlo.gather<offset_dims = "
            f"[{self._int_list(range(k + 1))}], "
            f"collapsed_slice_dims = [{collapsed}], "
            f"start_index_map = [{mapped}], "
            f"index_vector_dim = 0>, "
            f"slice_sizes = {self._i64_array(slice_sizes)}}} : "
            f"({self._type_str(dtype, (n, n) + batch)}, "
            f"tensor<1xi64>) -> {self._type_str(dtype, (n,) + batch)}"
        )
        return g

    def _bcast(self, name, dtype, src_shape, target_shape, dims, lines) -> str:
        """``stablehlo.broadcast_in_dim`` with an explicit dimension mapping
        (the numpy-trailing-alignment helper ``_emit_broadcast_static`` does
        NOT apply to the Jacobi updates, whose batch dims stay leading)."""
        b = self._new_name()
        lines.append(
            f'{b} = "stablehlo.broadcast_in_dim"({name}) '
            f"{{broadcast_dimensions = {self._i64_array(dims)}}} : "
            f"({self._type_str(dtype, src_shape)}) -> "
            f"{self._type_str(dtype, target_shape)}"
        )
        return b

    def _ew(self, mnemonic, a, b, dtype, shape, lines) -> str:
        """Elementwise op line on ``shape`` (unary when ``b is None``)."""
        name = self._new_name()
        if b is None:
            lines.append(f"{name} = stablehlo.{mnemonic} {a} : {self._type_str(dtype, shape)}")
        else:
            lines.append(f"{name} = stablehlo.{mnemonic} {a}, {b} : {self._type_str(dtype, shape)}")
        return name

    def _cmp(self, a, b, direction, dtype, shape, lines) -> str:
        """``stablehlo.compare`` with an explicit direction; result i1."""
        name = self._new_name()
        lines.append(
            f'{name} = "stablehlo.compare"({a}, {b}) '
            f"{{comparison_direction = #stablehlo<comparison_direction {direction}>}}"
            f" : ({self._type_str(dtype, shape)}, "
            f"{self._type_str(dtype, shape)}) -> "
            f"{self._type_str(np.dtype('bool'), shape)}"
        )
        return name

    def _sel(self, pred, on_true, on_false, dtype, shape, lines) -> str:
        """``stablehlo.select`` on ``shape`` (pred is the i1 mask)."""
        name = self._new_name()
        lines.append(
            f"{name} = stablehlo.select {pred}, {on_true}, {on_false} : "
            f"({self._type_str(np.dtype('bool'), shape)}, "
            f"{self._type_str(dtype, shape)}, "
            f"{self._type_str(dtype, shape)}) -> {self._type_str(dtype, shape)}"
        )
        return name

    def _emit_diagonal_extract(self, a_name, dtype, x_shape, n, lines) -> str:
        """Main diagonal of a square matrix (batch..., n, n) → (batch..., n):
        transpose so the matrix dims are at the front, a 2-index gather with
        the constant indices ``[[0,0],[1,1],...]`` (all slice dims
        collapsed; batch dims come back as offset dims), then transpose the
        batch dims to the front again. For rank 2 the two transposes are
        the identity and are skipped."""
        rank = len(x_shape)
        batch = x_shape[:-2]
        i64 = np.dtype("int64")
        idx2 = np.stack([np.arange(n, dtype=np.int64)] * 2, axis=1)
        iname = self._new_name()
        lines.append(
            f"{iname} = stablehlo.constant {self._constant_text(idx2)} : "
            f"{self._type_str(i64, (n, 2))}"
        )
        if rank == 2:
            src, src_shape = a_name, x_shape
            offset_dims: list = []
        else:
            src = self._new_name()
            src_shape = (n, n) + batch
            perm = [rank - 2, rank - 1] + list(range(rank - 2))
            lines.append(
                f'{src} = "stablehlo.transpose"({a_name}) '
                f"{{permutation = {self._i64_array(perm)}}} : "
                f"({self._type_str(dtype, x_shape)}) -> "
                f"{self._type_str(dtype, src_shape)}"
            )
            # The gather result is (index batch dims ++ offset dims): the
            # single index batch dim (the matrix index) occupies result
            # position 0, so the operand's batch offset dims land at result
            # positions 1..rank-2.
            offset_dims = list(range(1, rank - 1))
        g = self._new_name()
        lines.append(
            f'{g} = "stablehlo.gather"({src}, {iname}) '
            f"{{dimension_numbers = #stablehlo.gather<offset_dims = "
            f"[{self._int_list(offset_dims)}], collapsed_slice_dims = [0, 1], "
            f"start_index_map = [0, 1], index_vector_dim = 1>, "
            f"slice_sizes = {self._i64_array([1, 1] + list(batch))}}} : "
            f"({self._type_str(dtype, src_shape)}, {self._type_str(i64, (n, 2))})"
            f" -> {self._type_str(dtype, (n,) + batch)}"
        )
        if rank == 2:
            return g
        out = self._new_name()
        lines.append(
            f'{out} = "stablehlo.transpose"({g}) '
            f"{{permutation = {self._i64_array(list(range(1, rank - 1)) + [0])}}} : "
            f"({self._type_str(dtype, (n,) + batch)}) -> "
            f"{self._type_str(dtype, batch + (n,))}"
        )
        return out

    def _emit_reorder_columns(self, v_name, perm_name, dtype, x_shape, n, lines) -> str:
        """``V[..., perm, :]`` (numpy advanced indexing on the second-to-last
        axis): the operand is transposed so the gathered axis is 0 and its
        rows are the V columns (``(n, batch..., n)``), then gathered with
        the ``(batch..., n)`` permutation indices (offset dim = the trailing
        ``n``), and transposed back (the gather places the sorted column at
        result row i; swapping the last two dims restores the column layout)
        → ``(batch..., n, n)``."""
        rank = len(x_shape)
        batch = x_shape[:-2]
        i64 = np.dtype("int64")
        pshape = batch + (n,)
        iv_shape = pshape + (1,)
        iv = self._new_name()
        lines.append(
            f"{iv} = stablehlo.reshape {perm_name} : "
            f"({self._type_str(i64, pshape)}) -> {self._type_str(i64, iv_shape)}"
        )
        if rank == 2:
            src = self._new_name()
            lines.append(
                f'{src} = "stablehlo.transpose"({v_name}) '
                f"{{permutation = {self._i64_array([1, 0])}}} : "
                f"({self._type_str(dtype, x_shape)}) -> "
                f"{self._type_str(dtype, x_shape)}"
            )
            g = self._new_name()
            lines.append(
                f'{g} = "stablehlo.gather"({src}, {iv}) '
                f"{{dimension_numbers = #stablehlo.gather<offset_dims = [1], "
                f"collapsed_slice_dims = [0], start_index_map = [0], "
                f"index_vector_dim = 1>, "
                f"slice_sizes = {self._i64_array([1, n])}}} : "
                f"({self._type_str(dtype, x_shape)}, "
                f"{self._type_str(i64, iv_shape)}) -> "
                f"{self._type_str(dtype, x_shape)}"
            )
            out = self._new_name()
            lines.append(
                f'{out} = "stablehlo.transpose"({g}) '
                f"{{permutation = {self._i64_array([1, 0])}}} : "
                f"({self._type_str(dtype, x_shape)}) -> "
                f"{self._type_str(dtype, x_shape)}"
            )
            return out
        src = self._new_name()
        src_shape = (n,) + batch + (n,)
        perm = [rank - 1] + list(range(rank - 2)) + [rank - 2]
        lines.append(
            f'{src} = "stablehlo.transpose"({v_name}) '
            f"{{permutation = {self._i64_array(perm)}}} : "
            f"({self._type_str(dtype, x_shape)}) -> "
            f"{self._type_str(dtype, src_shape)}"
        )
        g = self._new_name()
        # Batched indices: the operand's batch dims (1..rank-2) pair with the
        # index batch dims via operand_batching_dims / start_indices_batching_dims
        # (iree 3.11 gather); the offset dim is the trailing n, the gathered
        # axis is dim 0 (collapsed). Result = index batch dims + offset dims =
        # (batch..., n, n) with the sorted column at result row i — the last
        # transpose restores the column layout.
        batching = list(range(1, rank - 1))
        lines.append(
            f'{g} = "stablehlo.gather"({src}, {iv}) '
            f"{{dimension_numbers = #stablehlo.gather<offset_dims = "
            f"[{rank - 1}], collapsed_slice_dims = [0], "
            f"operand_batching_dims = [{self._int_list(batching)}], "
            f"start_indices_batching_dims = "
            f"[{self._int_list(list(range(len(batch))))}], "
            f"start_index_map = [0], index_vector_dim = {rank - 1}>, "
            f"slice_sizes = {self._i64_array([1] + [1] * len(batch) + [n])}}} : "
            f"({self._type_str(dtype, src_shape)}, "
            f"{self._type_str(i64, iv_shape)}) -> "
            f"{self._type_str(dtype, pshape + (n,))}"
        )
        out = self._new_name()
        lines.append(
            f'{out} = "stablehlo.transpose"({g}) '
            f"{{permutation = "
            f"{self._i64_array(list(range(rank - 2)) + [rank - 1, rank - 2])}}} : "
            f"({self._type_str(dtype, pshape + (n,))}) -> "
            f"{self._type_str(dtype, x_shape)}"
        )
        return out

    def _emit_diag(self, op: Op) -> str:
        """``diag`` → numpy ``diag`` semantics composed from v1 ops (no
        StableHLO diag op exists): rank-1 ``(n,)`` → the ``(n, n)``
        diagonal matrix via an iota-EQ mask + select over a broadcast of
        the input (dtype preserved, incl. complex — the mask compares
        iotas, never data); rank-2 ``(m, n)`` → the main diagonal
        ``(min(m, n),)`` via a flatten reshape + a constant-index
        single-axis gather (indices ``i*(n+1)``). Dynamic dims → explicit
        BackendError (the constant index offsets need static shapes)."""
        x = op.operands[0]
        x_shape = tuple(x.type.shape)
        self._reject_dynamic_dims(op, x_shape, "operand shape")
        dtype = np.dtype(x.type.dtype)
        lines = []
        result_name = self._bind_results(op)[0]
        if len(x_shape) == 1:
            (n,) = x_shape
            i0 = self._emit_iota((n, n), 0, lines)
            i1 = self._emit_iota((n, n), 1, lines)
            eq = self._cmp(i0, i1, "EQ", np.dtype("int64"), (n, n), lines)
            xb = self._new_name()
            lines.append(
                f'{xb} = "stablehlo.broadcast_in_dim"({self._name(x)}) '
                f"{{broadcast_dimensions = {self._i64_array([1])}}} : "
                f"({self._type_str(dtype, (n,))}) -> "
                f"{self._type_str(dtype, (n, n))}"
            )
            zero, extra = self._scalar_constant_for(dtype, 0, (n, n))
            lines.extend(extra)
            lines.append(
                f"{result_name} = stablehlo.select {eq}, {xb}, {zero} : "
                f"({self._type_str(np.dtype('bool'), (n, n))}, "
                f"{self._type_str(dtype, (n, n))}, "
                f"{self._type_str(dtype, (n, n))}) -> "
                f"{self._type_str(dtype, (n, n))}"
            )
            return "\n".join(lines)
        m, n = x_shape
        k = min(m, n)
        i64 = np.dtype("int64")
        flat = self._new_name()
        lines.append(
            f"{flat} = stablehlo.reshape {self._name(x)} : "
            f"({self._type_str(dtype, (m, n))}) -> "
            f"{self._type_str(dtype, (m * n,))}"
        )
        idx = np.arange(k, dtype=np.int64) * (n + 1)
        iname = self._new_name()
        lines.append(
            f"{iname} = stablehlo.constant {self._constant_text(idx)} : "
            f"{self._type_str(i64, (k,))}"
        )
        iv = self._new_name()
        lines.append(
            f"{iv} = stablehlo.reshape {iname} : "
            f"({self._type_str(i64, (k,))}) -> {self._type_str(i64, (k, 1))}"
        )
        lines.append(
            f'{result_name} = "stablehlo.gather"({flat}, {iv}) '
            f"{{dimension_numbers = #stablehlo.gather<offset_dims = [], "
            f"collapsed_slice_dims = [0], start_index_map = [0], "
            f"index_vector_dim = 1>, "
            f"slice_sizes = {self._i64_array([1])}}} : "
            f"({self._type_str(dtype, (m * n,))}, "
            f"{self._type_str(i64, (k, 1))}) -> {self._type_str(dtype, (k,))}"
        )
        return "\n".join(lines)

    def _emit_arg_reduce(self, op: Op) -> str:
        """``argmax``/``argmin`` → a two-operand ``stablehlo.reduce`` over
        (value, full-rank iota) with the comparator ``(value_gt|lt) OR
        (value_eq AND idx_lt)`` — the first occurrence wins on ties,
        matching ``np.argmax``/``np.argmin`` (index init 0 keeps the
        reduce consistent: an element equal to the init never beats index
        0). axis=None flattens first (numpy); keepdims reshapes after.
        Float NaN: numpy returns the FIRST NaN position; StableHLO
        comparisons with NaN are false, so NaN never beats a finite value
        — divergence documented (evox fitnesses are finite)."""
        x = op.operands[0]
        kind = "max" if op.name == "argmax" else "min"
        axis_attr = op.attributes.get("axis")
        keepdims = bool(op.attributes.get("keepdims", False))
        x_shape = tuple(x.type.shape)
        self._reject_dynamic_dims(op, x_shape, "operand shape")
        lines = []
        if axis_attr is None:
            flat = (int(np.prod(x_shape)),)
            xf = self._new_name()
            lines.append(
                f"{xf} = stablehlo.reshape {self._name(x)} : "
                f"({self._vt(x.type)}) -> {self._type_str(x.type.dtype, flat)}"
            )
            red_shape_in = flat
            axis = 0
        else:
            rank = x.type.rank
            axis = self._normalize_axis(axis_attr, rank, f"{op.name}.axis")
            xf = self._name(x)
            red_shape_in = x_shape
        keys_dtype = np.dtype(x.type.dtype)
        key_name = xf
        if keys_dtype.kind == "b":
            t = self._new_name()
            lines.append(
                f"{t} = stablehlo.convert {key_name} : "
                f"({self._type_str(keys_dtype, red_shape_in)}) -> "
                f"{self._type_str(np.dtype('int8'), red_shape_in)}"
            )
            key_name, keys_dtype = t, np.dtype("int8")
        iota = self._emit_iota(red_shape_in, axis, lines)
        i64 = np.dtype("int64")
        v_init = self._reduce_init(kind, keys_dtype)
        vi, vi_lines = self._scalar_constant_for(keys_dtype, v_init, ())
        lines.extend(vi_lines)
        ii = self._new_name()
        lines.append(
            f"{ii} = stablehlo.constant {self._constant_text(np.asarray(0, dtype=i64))}"
            f" : tensor<i64>"
        )
        direction = "GT" if kind == "max" else "LT"
        key_elem = self._elem_type(keys_dtype)
        a, b, c, d = (self._new_name() for _ in range(4))
        c1, c2, c3, c4, c5 = (self._new_name() for _ in range(5))
        sel_v, sel_i = self._new_name(), self._new_name()
        region = (
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
        red_shape_out = tuple(
            d for i, d in enumerate(red_shape_in) if i != axis
        )
        vr, ir = self._new_name(), self._new_name()
        lines.append(
            f'{vr}, {ir} = "stablehlo.reduce"({key_name}, {iota}, {vi}, {ii}) '
            f"{region} {{dimensions = {self._i64_array([axis])}}} : "
            f"({self._type_str(keys_dtype, red_shape_in)}, "
            f"{self._type_str(i64, red_shape_in)}, {key_elem}, tensor<i64>) -> "
            f"({self._type_str(keys_dtype, red_shape_out)}, "
            f"{self._type_str(i64, red_shape_out)})"
        )
        cur = ir
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

    def _emit_tile(self, op: Op) -> str:
        """``tile`` → reshape + broadcast_in_dim + reshape decomposition
        matching numpy ``tile`` element-for-element (``out[i] =
        x[i mod s]``): the operand is promoted with leading size-1 dims
        when len(reps) > rank (reps right-aligned with leading 1s when
        shorter — NEVER a mix), then reshaped to interleave a size-1 dim
        BEFORE each dim, broadcast with the IDENTITY broadcast_dimensions
        (one entry per operand rank — the size-1 dims expand to the reps;
        element order then matches numpy's repeat loop exactly), and
        reshaped to the declared output — exact for static shapes."""
        x = op.operands[0]
        reps = tuple(int(r) for r in op.attributes["reps"])
        x_shape = tuple(x.type.shape)
        self._reject_dynamic_dims(op, x_shape, "operand shape")
        if not reps:
            self._names[id(op.result)] = self._name(x)
            return ""
        k = max(x.type.rank, len(reps))
        padded_shape = (1,) * (k - x.type.rank) + x_shape
        reps_k = (1,) * (k - len(reps)) + reps
        lines = []
        cur = self._name(x)
        if k != x.type.rank:
            t = self._new_name()
            lines.append(
                f"{t} = stablehlo.reshape {cur} : "
                f"({self._type_str(x.type.dtype, x_shape)}) -> "
                f"{self._type_str(x.type.dtype, padded_shape)}"
            )
            cur = t
        interleaved = tuple(d for s_i in padded_shape for d in (1, s_i))
        it = self._new_name()
        lines.append(
            f"{it} = stablehlo.reshape {cur} : "
            f"({self._type_str(x.type.dtype, padded_shape)}) -> "
            f"{self._type_str(x.type.dtype, interleaved)}"
        )
        bcast_shape = tuple(
            dim for r_i, s_i in zip(reps_k, padded_shape) for dim in (r_i, s_i)
        )
        bt = self._new_name()
        lines.append(
            f'{bt} = "stablehlo.broadcast_in_dim"({it}) '
            f"{{broadcast_dimensions = {self._i64_array(range(2 * k))}}} : "
            f"({self._type_str(x.type.dtype, interleaved)}) -> "
            f"{self._type_str(x.type.dtype, bcast_shape)}"
        )
        result_shape = tuple(op.result.type.shape)
        self._reject_dynamic_dims(op, result_shape, "result shape", op_name="tile")
        rt = self._new_name()
        lines.append(
            f"{rt} = stablehlo.reshape {bt} : "
            f"({self._type_str(x.type.dtype, bcast_shape)}) -> "
            f"{self._type_str(x.type.dtype, result_shape)}"
        )
        self._names[id(op.result)] = rt
        return "\n".join(lines)

    def _elem_dtype_str(self, dtype) -> str:
        return mapping.mlir_dtype(np.dtype(dtype))

    # --- Linear algebra ---

    def _dot_batch_tuple(self, rank: int, shape: tuple) -> tuple:
        """Batch dims of a dot operand under numpy matmul promotion: a
        rank-1 operand is promoted to a matrix by prepending (lhs) or
        appending (rhs) a 1, so it has NO batch dims of its own."""
        return () if rank == 1 else tuple(shape[:-2])

    def _dot_batch_dim_equal(self, d1, d2) -> bool:
        """True when two aligned batch dims are provably equal at runtime:
        equal statics, structurally equal symbolic dims, ``None`` pairs,
        or same-named ``Dim``\\ s (core semantics: same name unifies)."""
        if d1 == d2:
            return True
        return (
            isinstance(d1, Dim)
            and isinstance(d2, Dim)
            and d1.name == d2.name
        )

    def _dot_batch_all_one(self, batch: tuple) -> bool:
        """True for a non-empty batch tuple of provable size-1 dims (the
        tensor is identical across every batch position)."""
        return bool(batch) and all(d == 1 for d in batch)

    def _dot_broadcast_dim(self, da, db, op: Op):
        """Broadcast one aligned batch-dim pair (numpy rule, mirroring
        ``ir.inference._broadcast_dim``): 1 yields the other side; equal
        dims pass through; two unequal concrete ints raise BackendError
        (trace-time ShapeError already rejected them — defensive here);
        ``None`` (unchecked) yields ``None``; otherwise a
        ``DimExpr("max", ...)`` defers the check to runtime (the caller
        rejects those dims when a broadcast would be needed)."""
        if da == 1:
            return db
        if db == 1:
            return da
        if self._dot_batch_dim_equal(da, db):
            return da
        if da is None or db is None:
            return None
        if isinstance(da, int) and isinstance(db, int):
            raise BackendError(
                f"stablehlo export: op 'dot'{self._loc(op)} cannot "
                f"broadcast incompatible static batch dims {da!r} and "
                f"{db!r}"
            )
        return DimExpr("max", da, db)

    def _dot_broadcast_batch(self, ba: tuple, bb: tuple, op: Op) -> tuple:
        """Matmul batch broadcast of two dot-operand batch tuples: numpy
        RIGHT-aligned rule, matching ``ir.inference.infer_dot`` exactly
        (rank-1 operands are already reduced to an empty batch by
        ``_dot_batch_tuple``)."""
        rank = max(len(ba), len(bb))
        out = []
        for i in range(rank):
            da = 1 if i < rank - len(ba) else ba[i - (rank - len(ba))]
            db = 1 if i < rank - len(bb) else bb[i - (rank - len(bb))]
            out.append(self._dot_broadcast_dim(da, db, op))
        return tuple(out)

    def _emit_dot(self, op: Op) -> str:
        a, b = op.operands
        la, lb = a.type.rank, b.type.rank
        a_batch = self._dot_batch_tuple(la, tuple(a.type.shape))
        b_batch = self._dot_batch_tuple(lb, tuple(b.type.shape))
        target = self._dot_broadcast_batch(a_batch, b_batch, op)
        n = len(target)
        # etl dot = numpy matmul: contracting dims are the last of `a` /
        # second-to-last of `b`; batch dims are the leading ones.
        lines: list[str] = []
        a_name, b_name = self._name(a), self._name(b)
        a_shape = tuple(a.type.shape)
        b_shape = tuple(b.type.shape)
        if la == 1 or lb == 1:
            # Rank-1 operand (defensive — ``etl.dot`` requires rank >= 2,
            # but IR can be built directly): numpy promotes it to a matrix
            # with an empty batch tuple, so emit a NON-batched dot_general
            # whose free dims reproduce the promotion exactly — v·v:
            # contracting [0]x[0] (scalar result); M·v:
            # lhs [la-1] x rhs [0]; v·M: lhs [0] x rhs [lb-2].
            lhs_batch = rhs_batch = ()
            if la == 1:
                lhs_contract = (0,)
                rhs_contract = (0,) if lb == 1 else (lb - 2,)
            else:
                lhs_contract = (la - 1,)
                rhs_contract = (0,)
        elif len(a_batch) == len(b_batch) and all(
            self._dot_batch_dim_equal(x, y)
            for x, y in zip(a_batch, b_batch)
        ):
            # Matched batch structure — direct batched dot_general (the
            # original v1 behavior, byte-identical for existing cases).
            lhs_batch = rhs_batch = tuple(range(n))
            lhs_contract, rhs_contract = (la - 1,), (lb - 2,)
        elif not b_batch and _shape_is_static(
            tuple(a.type.shape)
        ) and _shape_is_static(tuple(b.type.shape)):
            # rhs is a plain matrix: emit a NON-batched dot_general whose
            # lhs free dims ARE a's batch dims — the result
            # (a_batch, m, n) matches infer_dot exactly with NO broadcast
            # (avoids materializing the batch expansion). Only for fully
            # static shapes: the iree llvm-cpu pipeline of this generation
            # cannot legalize the dynamic_reshape its own import inserts
            # for a non-batched dot_general with dynamic dims, so dynamic
            # shapes fall through to the (batched, dynamic-safe) broadcast
            # path below.
            lhs_batch = rhs_batch = ()
            lhs_contract, rhs_contract = (la - 1,), (lb - 2,)
        else:
            # Batch broadcasting required: expand each operand's batch
            # dims up to the target batch shape (numpy right-aligned
            # rule), then emit a matched batched dot_general. A rhs whose
            # batch is provably all size-1 contributes no result dims —
            # reshape it to a plain matrix and use the non-batched form
            # instead of materializing the broadcast. Only for fully
            # static shapes (same gate as the plain-matrix fast path
            # above): the iree llvm-cpu pipeline of this generation
            # cannot legalize the dynamic_reshape its own import inserts
            # for a non-batched dot_general with dynamic dims (the lhs
            # free dims or the squeeze itself), so dynamic shapes fall
            # through to the batched dynamic-broadcast path below, which
            # legalizes fine.
            for d in target:
                if isinstance(d, DimExpr):
                    raise BackendError(
                        f"stablehlo export: op 'dot'{self._loc(op)} cannot "
                        f"broadcast batch dims {a_batch!r} and {b_batch!r} "
                        f"(shapes {tuple(a.type.shape)!r} and "
                        f"{tuple(b.type.shape)!r}) — symbolic dims that "
                        "cannot be proven equal or size-1 at compile time "
                        "(dynamic batch broadcast)"
                    )
            if (
                self._dot_batch_all_one(b_batch)
                and len(b_batch) <= len(a_batch)
                and _shape_is_static(tuple(a.type.shape))
                and _shape_is_static(tuple(b.type.shape))
            ):
                b_name, extra, b_shape = self._dot_squeeze(op, b)
                lines.extend(extra)
                lhs_batch = rhs_batch = ()
                lhs_contract, rhs_contract = (la - 1,), (0,)
            else:
                a_name, extra, a_shape = self._dot_broadcast_operand(
                    op, a_name, a.type.dtype, tuple(a.type.shape),
                    b, target, n,
                )
                lines.extend(extra)
                b_name, extra, b_shape = self._dot_broadcast_operand(
                    op, b_name, b.type.dtype, tuple(b.type.shape),
                    a, target, n,
                )
                lines.extend(extra)
                lhs_batch = rhs_batch = tuple(range(n))
                # Contracting dims are relative to the BROADCAST shapes
                # (batch..., m, k) / (batch..., k, n): last / second-last.
                lhs_contract, rhs_contract = (n + 1,), (n,)
        dnums = (
            "#stablehlo.dot<"
            f"lhs_batching_dimensions = [{self._int_list(lhs_batch)}], "
            f"rhs_batching_dimensions = [{self._int_list(rhs_batch)}], "
            f"lhs_contracting_dimensions = [{self._int_list(lhs_contract)}], "
            f"rhs_contracting_dimensions = [{self._int_list(rhs_contract)}]>"
        )
        elem_dtype = np.dtype(a.type.dtype)
        dot_type = self._type_str(elem_dtype, op.result.type.shape)
        dot_name = self._new_name()
        lines.append(
            f'{dot_name} = "stablehlo.dot_general"({a_name}, '
            f"{b_name}) {{dot_dimension_numbers = {dnums}}} : "
            f"({self._type_str(a.type.dtype, a_shape)}, "
            f"{self._type_str(b.type.dtype, b_shape)}) -> "
            f"{dot_type}"
        )
        cur = dot_name
        # dot_general does not promote (XLA keeps the operand element type);
        # etl dot promotes dtypes per numpy.
        if np.dtype(op.result.type.dtype) != elem_dtype:
            converted = self._new_name()
            lines.append(
                f"{converted} = stablehlo.convert {cur} : "
                f"({dot_type}) -> {self._vt(op.result.type)}"
            )
            cur = converted
        self._names[id(op.result)] = cur
        return "\n".join(lines)

    def _dot_squeeze(self, op: Op, operand: Value) -> tuple[str, list, tuple]:
        """Reshape an operand whose batch dims are provably all size-1 to
        its plain matrix form — the tensor is identical across every batch
        position, so dropping the batch dims lets the dot emit a
        NON-batched dot_general (the size-1 batch dims must not appear as
        free dims in the result). Returns ``(name, lines, matrix_shape)``."""
        shape = tuple(operand.type.shape)
        matrix = shape[-2:]
        name = self._new_name()
        line = (
            f"{name} = stablehlo.reshape {self._name(operand)} : "
            f"({self._vt(operand.type)}) -> "
            f"{self._type_str(operand.type.dtype, matrix)}"
        )
        return name, [line], matrix

    def _dot_broadcast_operand(
        self,
        op: Op,
        name: str,
        dtype,
        shape: tuple,
        other: Value,
        target: tuple,
        n: int,
    ) -> tuple[str, list, tuple]:
        """SSA name of a dot operand broadcast from its batch dims up to
        the dot's target batch shape ``target`` (n = len(target)), plus
        prepend lines and the broadcast result shape. Right-aligned numpy
        mapping: the operand's batch dims land in the LAST
        ``len(shape) - 2`` target positions and its two matrix dims in
        positions n, n+1. ``other`` supplies the runtime size of
        pure-broadcast dynamic dims (its aligned batch dims). Static
        target shapes emit ``stablehlo.broadcast_in_dim``; dynamic targets
        emit ``stablehlo.dynamic_broadcast_in_dim`` with the runtime
        ``output_dimensions`` built from per-dim pieces (see
        ``_dot_dynamic_sources``)."""
        k = len(shape) - 2
        target_shape = target + shape[-2:]
        if shape == target_shape:
            return name, [], shape
        dims = list(range(n - k, n)) + [n, n + 1]
        if _shape_is_static(target_shape):
            bcast = self._new_name()
            line = (
                f'{bcast} = "stablehlo.broadcast_in_dim"({name}) '
                f"{{broadcast_dimensions = {self._i64_array(dims)}}} : "
                f"({self._type_str(dtype, shape)}) -> "
                f"{self._type_str(dtype, target_shape)}"
            )
            return bcast, [line], target_shape
        src_map = self._dot_dynamic_sources(op, name, shape, other, target, n)
        bcast, lines = self._dot_dynamic_bcast(
            name, dtype, shape, target_shape, dims, src_map
        )
        return bcast, lines, target_shape

    def _dot_dynamic_sources(
        self,
        op: Op,
        name: str,
        shape: tuple,
        other: Value,
        target: tuple,
        n: int,
    ) -> dict:
        """Per-result-dim runtime-size providers for the dynamic broadcast
        of ``(name, shape)`` up to ``target + shape[-2:]``:
        ``{result_dim_index: (provider_name, provider_type, provider_dim)}``
        for every dynamic dim, read via ``stablehlo.get_dimension_size``.
        A dim is sourced from the operand's own mapped dims when those are
        not provably 1, else from the other operand's aligned batch dim.
        Raises BackendError when no provider exists (unprovable merge —
        message names "dynamic")."""
        k = len(shape) - 2
        other_shape = tuple(other.type.shape)
        ko = len(other_shape) - 2
        other_name = self._name(other)
        other_type = self._type_str(other.type.dtype, other_shape)
        src_map: dict = {}
        for j, dim in enumerate(target):
            if _is_static_dim(dim):
                continue
            if isinstance(dim, DimExpr):
                raise BackendError(
                    f"stablehlo export: op 'dot'{self._loc(op)} cannot "
                    f"broadcast batch dims {tuple(shape[:-2])!r} and "
                    f"{other_shape[:-2]!r} (shapes {shape!r} and "
                    f"{other_shape!r}) — symbolic dims that cannot be "
                    "proven equal or size-1 at compile time (dynamic "
                    "batch broadcast)"
                )
            prov = None
            if (
                n - k <= j < n
                and not _is_static_one(shape[j - (n - k)])
            ):
                prov = (name, self._type_str(dtype, shape), j - (n - k))
            elif (
                n - ko <= j < n
                and not _is_static_one(other_shape[j - (n - ko)])
            ):
                prov = (other_name, other_type, j - (n - ko))
            if prov is None:
                raise BackendError(
                    f"stablehlo export: op 'dot'{self._loc(op)} cannot "
                    f"broadcast batch dims {tuple(shape[:-2])!r} and "
                    f"{other_shape[:-2]!r} (shapes {shape!r} and "
                    f"{other_shape!r}) — no runtime size source for the "
                    "dynamic batch dim (dynamic batch broadcast)"
                )
            src_map[j] = prov
        # Matrix dims are passed through unchanged: symbolic ones are
        # sourced from the operand itself.
        for j in (n, n + 1):
            if not _is_static_dim(shape[-2 + (j - n)]):
                src_map[j] = (name, self._type_str(dtype, shape), k + (j - n))
        return src_map

    def _dot_dynamic_bcast(
        self,
        value_name: str,
        dtype,
        operand_shape: tuple,
        result_shape: tuple,
        dims: list,
        src_map: dict,
    ) -> tuple[str, list]:
        """Emit ``stablehlo.dynamic_broadcast_in_dim`` for a dot operand
        whose target batch has dynamic dims. The runtime
        ``output_dimensions`` are built from per-dim pieces (mirroring
        ``_emit_dynamic_broadcast``): static dims via constants, dynamic
        dims via ``stablehlo.get_dimension_size`` (+ ``reshape`` to
        ``tensor<1xi32>``) on the provider from ``src_map``, joined by
        ``stablehlo.concatenate``. Returns ``(broadcast_name, lines)``."""
        lines: list[str] = []
        pieces: list[str] = []
        for i, dim in enumerate(result_shape):
            if _is_static_dim(dim):
                piece = self._new_name()
                lines.append(
                    f"{piece} = stablehlo.constant "
                    f"{self._constant_text(np.asarray([int(dim)], dtype=np.int32))}"
                    f" : tensor<1xi32>"
                )
            else:
                src_name, src_type, src_dim = src_map[i]
                size_name = self._new_name()
                lines.append(
                    f'{size_name} = "stablehlo.get_dimension_size"'
                    f"({src_name}) {{dimension = {int(src_dim)} : i64}} : "
                    f"({src_type}) -> tensor<i32>"
                )
                piece = self._new_name()
                lines.append(
                    f"{piece} = stablehlo.reshape {size_name} : "
                    f"(tensor<i32>) -> tensor<1xi32>"
                )
            pieces.append(piece)
        if len(pieces) == 1:
            dims_name = pieces[0]
        else:
            dims_name = self._new_name()
            piece_types = ", ".join("tensor<1xi32>" for _ in pieces)
            lines.append(
                f"{dims_name} = stablehlo.concatenate {', '.join(pieces)}, "
                f"dim = 0 : ({piece_types}) -> tensor<{len(pieces)}xi32>"
            )
        bcast_name = self._new_name()
        lines.append(
            f'{bcast_name} = "stablehlo.dynamic_broadcast_in_dim"'
            f"({value_name}, {dims_name}) "
            f"{{broadcast_dimensions = {self._i64_array(dims)}}} : "
            f"({self._type_str(dtype, operand_shape)}, "
            f"tensor<{len(pieces)}xi32>) -> "
            f"{self._type_str(dtype, result_shape)}"
        )
        return bcast_name, lines

    def _emit_conv(self, op: Op) -> str:
        x, w = op.operands
        # v1 uniformity: SAME padding already needed static dims; extend the
        # rejection to ALL conv shapes — a dynamic dim anywhere (operand or
        # result) fails here with a clear BackendError, never invalid MLIR.
        self._reject_dynamic_dims(op, tuple(x.type.shape), "operand shape")
        self._reject_dynamic_dims(op, tuple(w.type.shape), "operand shape")
        self._reject_dynamic_dims(op, tuple(op.result.type.shape), "result shape")
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
        # The StableHLO conv dnums custom syntax is positional with role
        # letters: `[b, f, S...]x[o, i, S...]->[b, f, S...]` — the digit
        # slots are the REMAINING (spatial) tensor positions, not dimension
        # indices. `[b, 0, f]` would mean NHWC (feature LAST) and produces
        # verifier failures ("input feature dimension / feature_group_count
        # = kernel input feature dimension") — never emit that.
        dnums = (
            "#stablehlo.conv<"
            f"[b, f, {spatial}]x[o, i, {spatial}]->[b, f, {spatial}]>"
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
        # window_strides / lhs_dilation / rhs_dilation / feature_group_count
        # / batch_group_count are REQUIRED attributes in the modern spec —
        # the verifier rejects conv ops that omit them (defaults are no
        # longer applied), so they are always emitted.
        attr_parts.append(f"window_strides = {self._i64_array(strides)}")
        attr_parts.append(f"lhs_dilation = {self._i64_array(in_dilation)}")
        attr_parts.append(f"rhs_dilation = {self._i64_array(k_dilation)}")
        attr_parts.append(f"feature_group_count = {int(feature_groups)} : i64")
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
                f"({conv_type}) -> {self._vt(op.result.type)}"
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

    # iree 3.11 mis-lowers a `stablehlo.while` nested inside a
    # `stablehlo.if` region (upstream; garbage binding descriptor at
    # invoke — `IndexError OUT_OF_RANGE ... binding length=...` on BOTH
    # llvm-cpu and cuda at every opt level; proven with raw
    # `iree-compile` + `iree-run-module` on the identical mlir with and
    # without the if wrapper; minimal repro `probe_min_while_if.mlir`).
    # `_emit_eigh`'s cyclic-Jacobi composition contains a while, so any
    # `etl.cond` wrapping an eigh decomposition (the evox CMAES fused
    # tell) used to emit exactly that crash shape.
    #
    # WORKAROUND: `_emit_if` HOISTS every while-producing subgraph out of
    # the branch regions — the branch-block prefix up to the last
    # while-producing op (``eigh`` / ``while`` / an ``if`` whose regions
    # recursively contain one) is emitted BEFORE the ``stablehlo.if`` and
    # the branch regions reference the hoisted values lexically
    # (StableHLO regions capture enclosing values). Safe because every op
    # that reaches the exporter is pure (effectful ops — collectives,
    # runtime_call/external_call — defer with BackendError, so nothing
    # with side effects can be moved). The hoisted prefix may compute
    # unconditionally what a branch computed conditionally — pure
    # semantics are preserved. The eigh composition stays a single while
    # (O(1) emitted text — no compile blowup at dim 50).
    #
    # Known residual: an eigh/while nested inside a cond inside a cond
    # (the inner if is inside the outer REGION, not the outer block
    # prefix) still emits while-in-if — upstream; restructure or
    # precompute. eigh inside a while BODY is not hoisted (semantics:
    # per-iteration) — while-in-while is a separate upstream question.

    def _emit_if(self, op: Op) -> str:
        pred = op.operands[0]
        # Bind the branch block args to the if operands BEFORE any hoisting
        # so the hoisted prefix resolves branch-local references to the
        # enclosing operand names (they dominate the hoist point — the if's
        # operands are emitted by the enclosing block before this op).
        for region in op.regions:
            for block in region.blocks:
                for arg, operand in zip(block.arguments, op.operands):
                    self._names[id(arg)] = self._name(operand)
        lines = []
        self._hoist_if_branch_whiles(op, lines)
        result_names = self._bind_results(op)
        lhs = f"{', '.join(result_names)} = " if result_names else ""
        lines.append(f'{lhs}"stablehlo.if"({self._name(pred)}) ({{')
        lines.append(self._emit_if_region(op.regions[0], op))
        lines.append("  }, {")
        lines.append(self._emit_if_region(op.regions[1], op))
        lines.append(
            f"  }}) : ({self._vt(pred.type)}) -> "
            f"{self._result_types_str(op.results)}"
        )
        return "\n".join(lines)

    def _hoist_if_branch_whiles(self, op: Op, lines: list) -> None:
        """Hoist every while-producing subgraph out of an ``if``'s branch
        regions (the iree 3.11 while-in-if workaround — see the comment
        above). For each branch block, the prefix up to and including the
        LAST while-producing op is emitted into ``lines`` (the enclosing
        block, just before the ``if``) and marked in ``self._hoisted_ops``
        so ``_block_body_lines`` skips it when the region is emitted."""
        for region in op.regions:
            for block in region.blocks:
                ops = list(block.ops)
                last = None
                for i, bop in enumerate(ops):
                    if bop.name == "return":
                        break
                    if self._is_while_producing(bop):
                        last = i
                if last is None:
                    continue
                for bop in ops[: last + 1]:
                    self._hoisted_ops.add(id(bop))
                    text = self._emit_op(bop)
                    if text:
                        lines.append(text)

    def _is_while_producing(self, bop: Op) -> bool:
        """True when emitting ``bop`` produces a ``stablehlo.while``:
        the ``eigh`` composition (one while), a direct ``while`` op, or an
        ``if`` whose regions recursively contain one."""
        if bop.name in ("eigh", "while"):
            return True
        if bop.name == "if":
            return any(
                self._region_has_while_producing(region)
                for region in bop.regions
            )
        return False

    def _region_has_while_producing(self, region) -> bool:
        return any(
            self._is_while_producing(bop)
            for block in region.blocks
            for bop in block.ops
            if bop.name != "return"
        )

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
        lines = []
        operand_names = [
            self._rewrite_while_init(op, v, lines) for v in op.operands
        ]
        result_names = self._bind_results(op)
        lhs = f"{', '.join(result_names)} = " if result_names else ""
        types = ", ".join(self._vt(v.type) for v in op.operands)
        lines.append(f'{lhs}"stablehlo.while"({", ".join(operand_names)}) ({{')
        lines.append(self._emit_while_region(op.regions[0]))
        lines.append("  }, {")
        lines.append(self._emit_while_region(op.regions[1]))
        lines.append(f"  }}) : ({types}) -> ({types})")
        return "\n".join(lines)

    # --- while-init constant rewrite (iree AffinityAnalysis SEGV) ---
    #
    # iree 3.11.0 SEGVs (SIGSEGV, "Error code: -11", during
    # VerifyLoweringToAsyncPass) on dense rank>=1 `stablehlo.constant`
    # values used as while-INIT operands — upstream
    # `IREE::Stream::AffinityAnalysis::run()` infinite recursion via
    # walkTransitiveUses ⇄ ValueConsumerAffinityPVS::updateValue (the real
    # evox NSGA2 `non_dominate_rank` while_loop: ~9/10 llvm-cpu, ~3/5
    # cuda; every pop/dim/probe fails; --mlir-disable-threading does not
    # help). The validated workaround (10/10 clean compiles + bit-exact
    # runs on llvm-cpu AND cuda) replaces an all-zero rank>=1 constant
    # init with COMPUTED zeros: `not(x)`/`and(x, not(x))` over a
    # same-shaped NON-constant pre-loop value + a dtype-changing `convert`
    # (+ `reduce` when the source is one axis larger) — the "P15" pattern.
    # Never `x - x` (subtract-derived) nor broadcast-of-scalar-0: iree's
    # canonicalizer folds both back into constants.
    #
    # SAFETY ORDERING (measured, iree 3.11): the SEGV is a use-graph-
    # topology effect, not a property of the zero expression. The REDUCE
    # path (source one axis larger, input-derived, bool/int) is the ONLY
    # class validated safe at real-graph scale (10/10 llvm-cpu AND cuda,
    # n=32/n=128/full NSGA2 tell). Exact-shape sources and the `iota`
    # fallback are small-graph-safe only — on the real non_dominate_rank
    # loop an exact-shape rank-1-derived source SEGVs 0/10 deterministically
    # and the iota fallback 0/10 (the exact-shape preference over the
    # reduce path was the bug in the first writer iteration; see
    # _find_while_zero_source). Non-zero constants and rank-0 (scalar)
    # inits are left untouched (scalars are unaffected; non-zero constants
    # have no valid replacement — such loops remain broken on iree, use
    # the unrolled fixed-point workaround; see stablehlo/CONTEXT.md Known
    # Issues).

    def _rewrite_while_init(self, op: Op, value: Value, lines: list) -> str:
        """Return the SSA name to use for one `while` init operand,
        emitting the computed-zeros rewrite into `lines` when applicable
        (see the class comment above for the trigger and the pattern)."""
        if not self._while_init_rewrite:
            return self._name(value)
        def_op = value.defining_op
        if def_op is None or def_op.name != "constant":
            return self._name(value)  # not a literal constant
        data = np.asarray(def_op.attributes["value"])
        target_shape = tuple(value.type.shape)
        if data.ndim < 1 or data.size == 0:
            return self._name(value)
        if not all(_is_static_dim(d) for d in target_shape):
            return self._name(value)
        if not bool(np.all(data == 0)):
            return self._name(value)  # non-zero constants: no valid rewrite
        target_dtype = np.dtype(value.type.dtype)
        src = self._find_while_zero_source(op, target_shape)
        # No usable source → the iota-derived fallback (always available
        # for a static target shape). NOTE: validated safe on SMALL graphs
        # only — at real-graph scale it also SEGVs (see the class comment).
        return self._emit_while_zero_init(src, target_shape, target_dtype, lines)

    def _find_while_zero_source(self, op: Op, target_shape: tuple):
        """Find a NON-constant pre-loop value (same block, emitted before
        the `while`) from which all-zero values of ``target_shape`` can be
        derived, with the classes ordered by validated safety on iree 3.11
        (the AffinityAnalysis SEGV is a use-graph-topology effect, NOT a
        property of the zero expression — the ordering below is the
        measured one):

        1. REDUCE path (validated-safe class, 10/10 clean compiles + bit
           exact runs on llvm-cpu AND cuda at real-graph scale): a value
           ONE axis larger than the target (input-derived, kind bool/int)
           reduced along the extra axis — the P15 dominance-matrix case:
           (n, n) i1 → not/and → convert i1→i32 → reduce dim 0 → (n,) init.
        2. EXACT-shape source (block args / non-constant derived values,
           kind bool/int): not/and over the same-shaped value (+ convert).
           Validated-safe on small graphs only — on the REAL
           non_dominate_rank loop an exact-shape rank-1-derived source
           SEGVs 0/10 deterministically (preferring it over the reduce
           path was the bug in the first writer iteration).
        3. IOTA fallback (in ``_emit_while_zero_init`` when no source):
           small-graph-safe only; at real-graph scale it also SEGVs.

        Returns ``(name, src_shape, src_dtype, reduce_axis|None)`` or None
        (no usable source — the caller keeps the constant init)."""
        block = op.parent
        candidates = []
        if block is not None:
            # Block args first (function inputs / enclosing while carries
            # in nested regions) — they are non-constant by construction.
            candidates.extend(block.arguments)
            for bop in block.ops:
                if bop is op:
                    break
                if bop.name != "constant":
                    candidates.extend(bop.results)
        # 1. reduce path first — the only class validated safe at scale.
        for cand in candidates:
            c_shape = tuple(cand.type.shape)
            dtype = np.dtype(cand.type.dtype)
            if dtype.kind not in "biu":
                continue
            if len(c_shape) == len(target_shape) + 1 and all(
                _is_static_dim(d) for d in c_shape
            ):
                for a in range(len(c_shape)):
                    if c_shape[:a] + c_shape[a + 1 :] == target_shape:
                        return (self._name(cand), c_shape, dtype, a)
        # 2. exact-shape source (small-graph-safe only — see the docstring).
        for cand in candidates:
            c_shape = tuple(cand.type.shape)
            if c_shape == target_shape:
                dtype = np.dtype(cand.type.dtype)
                if dtype.kind in "biu":
                    return (self._name(cand), c_shape, dtype, None)
        return None

    def _emit_while_zero_init(self, src, target_shape: tuple, target_dtype,
                              lines: list) -> str:
        """Emit the computed-zeros pattern for a while-init rewrite and
        return the result name. ``src`` is ``(name, shape, dtype,
        reduce_axis|None)``; when the source kind is not bool/int (or no
        source was found — caller passes None), an ``iota`` of the target
        shape (int32) is derived instead."""
        if src is None:
            iota_name = self._new_name()
            lines.append(
                f"{iota_name} = stablehlo.iota dim = 0 : "
                f"{self._type_str(np.dtype('int32'), target_shape)}"
            )
            src = (iota_name, target_shape, np.dtype("int32"), None)
        src_name, src_shape, src_dtype, axis = src
        n = self._new_name()
        lines.append(
            f"{n} = stablehlo.not {src_name} : "
            f"{self._type_str(src_dtype, src_shape)}"
        )
        z = self._new_name()
        lines.append(
            f"{z} = stablehlo.and {src_name}, {n} : "
            f"{self._type_str(src_dtype, src_shape)}"
        )
        cur = z
        if src_dtype != target_dtype:
            t = self._new_name()
            lines.append(
                f"{t} = stablehlo.convert {cur} : "
                f"({self._type_str(src_dtype, src_shape)}) -> "
                f"{self._type_str(target_dtype, src_shape)}"
            )
            cur = t
        if axis is not None:
            elem_t = self._elem_type(target_dtype)
            zero = self._new_name()
            lines.append(
                f"{zero} = stablehlo.constant "
                f"{self._constant_text(np.asarray(0, dtype=target_dtype))} : "
                f"{elem_t}"
            )
            a1, a2 = self._new_name(), self._new_name()
            s = self._new_name()
            region = (
                "({\n"
                f"  ^bb0({a1}: {elem_t}, {a2}: {elem_t}):\n"
                f"    {s} = stablehlo.add {a1}, {a2} : {elem_t}\n"
                f"    stablehlo.return {s} : {elem_t}\n"
                "  })"
            )
            r = self._new_name()
            lines.append(
                f'{r} = "stablehlo.reduce"({cur}, {zero}) {region} '
                f"{{dimensions = {self._i64_array([axis])}}} : "
                f"({self._type_str(target_dtype, src_shape)}, {elem_t}) -> "
                f"{self._type_str(target_dtype, target_shape)}"
            )
            cur = r
        return cur

    def _emit_while_region(self, region) -> str:
        """One `while` region (cond/body): block args bound positionally to
        the loop-carried operands. Args use FRESH counter names (never
        `%argN` — MLIR SSA names are function-scoped and would collide
        with the enclosing function's entry arguments / nested regions)."""
        out = []
        for i, block in enumerate(region.blocks):
            args = []
            for arg in block.arguments:
                name = self._new_name()
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
        result_type = self._vt(op.result.type)
        if name in ("all_gather", "reduce_scatter"):
            # etl scales the axis dim by world_size (symbolic => `?`), but
            # the emitted replica_groups describe a CONCRETE group of `n`
            # ranks, so the result type must be consistent with the emitted
            # program (iree-compile rejects `tensor<?x...>` result types:
            # "tensor.empty op incorrect number of dynamic sizes").
            result_type = self._collective_result_type(op, x, name, group_size)
        signature = (
            f"({self._vt(x.type)}) -> {result_type}"
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

    def _collective_result_type(self, op: Op, x: Value, name: str, group_size) -> str:
        """Result tensor type of all_gather/reduce_scatter, consistent with
        the emitted ``replica_groups`` (an n-rank group — n = 1 for the
        unknown world group, matching the v1 single-rank simulation).

        The IR result dim at the axis is ``operand_dim * world_size`` /
        ``// world_size`` — symbolic in general. When the operand axis dim
        is concrete, substitute the emitted program's concrete count
        (operand_dim * n / operand_dim // n); otherwise keep the symbolic
        dim (the compiler rejects dynamic shapes at input time — surfaced
        there, never silently re-specialized here).
        """
        axis = int(op.attributes.get("axis", 0))
        n = 1
        if (
            isinstance(group_size, int)
            and not isinstance(group_size, bool)
            and group_size > 0
        ):
            n = group_size
        shape = list(op.result.type.shape)
        operand_dim = x.type.shape[axis]
        if isinstance(operand_dim, int) and not isinstance(operand_dim, bool):
            result_dim = operand_dim * n if name == "all_gather" else operand_dim // n
            shape[axis] = result_dim
        rendered = self._type_str(op.result.type.dtype, shape)
        # Record the override so the enclosing func signature / return
        # terminator render the SAME concrete type as this op.
        self._type_overrides[id(op.result)] = rendered
        return rendered

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
            result_name = self._bind_results(op)[0]
            # The literal zero is a scalar constant; StableHLO elementwise
            # ops require equal shapes, so broadcast it when x is
            # non-scalar (a scalar x uses the constant directly —
            # broadcast_in_dim with empty dims is invalid on scalars). A
            # dynamic x shape sources the runtime output dimensions.
            x_shape = tuple(x.type.shape)
            zero_name, zero_lines = self._scalar_constant_for(
                dtype,
                0,
                x_shape,
                shape_source=(self._name(x), x.type.dtype, x_shape),
            )
            return "\n".join(
                [
                    *zero_lines,
                    f"{result_name} = stablehlo.maximum {self._name(x)}, "
                    f"{zero_name} : {self._vt(op.result.type)}",
                ]
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
                    types = ", ".join(self._vt_value(v) for v in op.operands)
                    lines.append(f"{terminator} {values} : {types}")
                else:
                    lines.append(terminator)
                break
            if id(op) in self._hoisted_ops:
                continue  # already emitted before the enclosing `if`
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

    def _vt_value(self, value: Value) -> str:
        """Rendered tensor type of `value`, honoring per-value overrides
        (see `_type_overrides`)."""
        return self._type_overrides.get(id(value), self._vt(value.type))

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
    def _list_attr(values) -> str:
        """Plain ``[a, b]`` list text for CUSTOM-form attributes (`dims =
        [...]` on ``stablehlo.broadcast_in_dim``) — the custom form takes a
        DenseI64ArrayAttr printed as a plain list, NOT the ``array<i64: ...>``
        spelling (iree-compile rejects the latter in custom form)."""
        return "[" + ", ".join(str(int(v)) for v in values) + "]"

    @staticmethod
    def _i64_array(values) -> str:
        """Render an index list as an i64 dense-array attribute.

        Modern StableHLO (20241104-era spec) declares index-list attributes
        as DenseI64ArrayAttr and REJECTS the legacy ``dense<[...]> :
        tensor<Nxi64>`` spelling ("'dimensions' failed to satisfy
        constraint: i64 dense array attribute"). The valid syntax is
        ``array<i64: 1, 2>``; the empty array prints as ``array<i64>``
        (``array<i64: >`` does NOT parse)."""
        inner = ", ".join(str(int(v)) for v in values)
        if not inner:
            return "array<i64>"
        return f"array<i64: {inner}>"

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
