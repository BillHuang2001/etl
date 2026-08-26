"""StableHLO MLIR text emission (Writer skeleton — architecture phase).

The Writer walks a verified `etl.ir.Module` and emits StableHLO MLIR text.
Mapping data lives in `./ops.py`; this module contains NO mapping tables —
it only consumes them. All behavioral bodies raise NotImplementedError until
the implementation phase (delegated to subagent_manager).

Import rules (binding, from `../../CONTEXT.md`): top-level imports are
restricted to `etl.core` and `etl.ir`; `etl.pipeline` is never imported.
The Writer receives an already-unwrapped `etl.ir.Module` from `export()`,
so `etl.trace` is not needed (Graph handling stays in `__init__.py`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import ops as mapping

if TYPE_CHECKING:
    from etl.ir import Function, Module, Op, Value


class Writer:
    """Emit StableHLO MLIR text for a verified `etl.ir.Module`.

    Emission approach (binding for the implementation):

    - **Input**: `export()` (in `./__init__.py`) unwraps a `trace.Graph` to
      its `.module`, calls `module.verify()`, and passes the verified module
      here. The Writer itself assumes a verified `etl.ir.Module`.
    - **Structure**: walk `module.functions` in declaration order. The
      output is wrapped in `module { ... }`. Function signatures render
      tensor-typed args/results via `mlir_type`.
    - **Emission order**: emit ops in block order (program order IS effect
      order — this preserves write/read/collective/callback semantics).
    - **SSA names**: assign `%0`, `%1`, ... to each `ir.Value` on first
      use; results are named by the op that produces them. Block arguments
      are named per function.
    - **Dispatch** (`write_op`) on the op name via `mapping.status`:
      * `"v1"` — emit `mapping.lookup_mapping(name)` with operand/result
        refs and attributes (`render_attr`). Comparisons additionally emit
        the `comparison_direction` attribute from `mapping.COMPARISON_MAP`.
      * `"decompose"` — emit the expansion described by
        `mapping.DECOMPOSITIONS` as ordinary sub-ops.
      * `"deferred"` or unknown — raise `core.BackendError` NAMING THE OP
        with a message suggesting decomposition or a future adapter. Never
        silently skip an op or emit partial output.
    - **Symbolic dims**: int dims render literally; `Dim`/`DimExpr`/`None`
      dims render as `?` (e.g. `tensor<?xNxf32>`). Rank is always concrete
      in etl, so the emitted rank is always correct.
    - **Constants**: `constant` ops render via `render_constant` as
      `stablehlo.constant` with a `dense<[...]>` elements attribute.
    - **Collectives**: the op name `broadcast` collides between the
      data-movement op (`stablehlo.broadcast_in_dim`) and the dist
      collective (`stablehlo.collective_broadcast`) — disambiguate by the
      op's effect kind (collective effect ⇒ collective mnemonic).
    """

    def __init__(self, module: Module) -> None:
        """Store the verified module and initialize per-write state.

        State: SSA name counter, per-value name map, function index.
        Raises TypeError if `module` is not an `etl.ir.Module` (export()
        pre-validates; this is a defensive check only).
        """
        raise NotImplementedError(
            "Writer.__init__ is an architecture-phase stub; implementation is delegated to subagent_manager"
        )

    def write(self) -> str:
        """Return the complete MLIR text for the module.

        Emits the `module { ... }` wrapper, then `write_function` for each
        function in `module.functions`, in order.
        """
        raise NotImplementedError(
            "Writer.write is an architecture-phase stub; implementation is delegated to subagent_manager"
        )

    def write_function(self, fn: Function) -> str:
        """Emit one `func.func` definition as MLIR text.

        Signature: function name + tensor-typed args/results via
        `mlir_type` (dynamic dims as `?`). Body: the entry block's ops in
        program order via `write_op`, followed by a `func.return` of the
        terminator's operands. Nested regions (cond/while_loop bodies) are
        emitted inline by `write_op`.
        """
        raise NotImplementedError(
            "Writer.write_function is an architecture-phase stub; implementation is delegated to subagent_manager"
        )

    def write_op(self, op: Op) -> str:
        """Dispatch a single op to its StableHLO emission.

        Uses `mapping.status(op.name)` / `mapping.lookup_mapping(op.name)`
        per the class docstring. Deferred or unknown ops raise
        `core.BackendError` naming the op. Region-carrying ops (`cond`,
        `while_loop`) emit their region blocks recursively.
        """
        raise NotImplementedError(
            "Writer.write_op is an architecture-phase stub; implementation is delegated to subagent_manager"
        )

    def mlir_type(self, dtype, shape) -> str:
        """Map a (numpy dtype, etl shape) pair to a StableHLO tensor type.

        dtype → `mapping.DTYPE_MAP` via `mapping.mlir_dtype` (normalized
        with `numpy.dtype`; unknown dtype ⇒ `core.BackendError` naming the
        dtype). Int dims render literally; symbolic dims (`Dim`/`DimExpr`)
        and `None` render as `?`. Examples: `(float32, (None, 256))` →
        `tensor<?x256xf32>`; `(bool, (2, 3))` → `tensor<2x3xi1>`.
        """
        raise NotImplementedError(
            "Writer.mlir_type is an architecture-phase stub; implementation is delegated to subagent_manager"
        )

    def render_attr(self, value) -> str:
        """Render a Python value as MLIR attribute syntax.

        Covers int/float/bool/str (bare literals or `"..."`), containers
        (`dense<[...]>`-style arrays, comma-separated lists, dicts as named
        attrs), and op-specific attribute layouts (e.g. `comparison_direction`
        for comparisons, window/convolution dims for conv, axis lists for
        reductions). Used by `write_op` to fill attribute slots.
        """
        raise NotImplementedError(
            "Writer.render_attr is an architecture-phase stub; implementation is delegated to subagent_manager"
        )

    def render_constant(self, value: Value) -> str:
        """Render a `constant` op's data as `stablehlo.constant`.

        Emits a `dense<[...]>` elements attribute (row-major, dtype-typed
        via `mapping.DTYPE_MAP`) followed by the op's result SSA name.
        """
        raise NotImplementedError(
            "Writer.render_constant is an architecture-phase stub; implementation is delegated to subagent_manager"
        )
