"""The numpy interpreter execution engine: ``Interpreter`` + ``KernelContext``.

Execution model (binding, parent CONTEXT.md):

- **Execution order = block op order.** This IS the effect ordering
  (write/read/collective/callback ops anchor order); pure ops keep program
  order for determinism.
- **Shape inference: reuse, don't duplicate.** Runtime shapes are computed by
  evaluating the IR result types (already inferred by ops-level inference at
  trace time — see ``ir/inference.py`` and the Builder) against name->int
  symbolic-dim bindings (``shapes.py``). The backend carries NO second copy
  of shape rules.
- **Control flow** (``if``/``while``/``scan``) is interpreted by recursively
  running region blocks through ``KernelContext.run_region`` — genuinely
  dynamic runtime control flow, never specialized per iteration.
- **Kernel dispatch** is the ``kernels`` table (``kernels.dispatch(op_name)``);
  the ``return`` terminator is special-cased in the loop, NOT dispatched.
- **Output validation:** every kernel result is checked against the op's
  declared result types — dtype must match exactly (``BackendError`` naming
  the op); shape is validated ELEMENTWISE (``ShapeError`` naming op/index/
  dim): rank must match exactly, an expected ``None`` (runtime-dynamic) dim
  is unchecked, and every other dim must evaluate to the concrete runtime
  dim. Kernels never silently coerce.

``KernelContext`` (the ``ctx`` of the kernel contract in
``kernels/__init__.py``) carries per-execution state:

- ``bindings: dict[str, int]`` — symbolic-dim bindings for the current
  execution (extended positionally by ``run_region`` from region-arg shapes).
- ``rank_context: RankContext`` — the effective execution context (from
  ``exec_context.get_rank_context()``, honoring the per-``run`` override).
- ``module: ir.Module`` — the module being executed.
- ``run_region(region, arg_tensors) -> list[Tensor]`` — binds the region's
  entry-block arguments to tensors, extends ``bindings``, and executes the
  block in op order.
- ``compute_output_shapes(op, input_shapes, input_dtypes) -> list[tuple[int|None, ...]]``
  — evaluates ``op.results[i].type.shape`` against ``bindings``; ``None``
  dims stay ``None`` (unchecked).
- ``resolve_callback(callback_id) -> callable`` — runtime_call callback
  lookup via ``etl.ops.constant._get_callback``.
- ``evaluate_shape(shape) -> tuple[int, ...]`` — convenience wrapper over
  ``shapes.evaluate_shape(shape, self.bindings)``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from etl import core
from etl import ir

from . import kernels
from . import shapes
from .exec_context import get_rank_context, set_rank_context

__all__ = ["KernelContext", "Interpreter", "entry_function"]


def entry_function(module: ir.Module) -> ir.Function:
    """The module's entry function: ``"main"``, else the single function.

    Raises:
        core.BackendError: The module has no ``"main"`` function and not
            exactly one function.
    """
    try:
        return module.get_function("main")
    except KeyError:
        if len(module.functions) == 1:
            return module.main
        raise core.BackendError(
            f"module '{module.name}' has {len(module.functions)} functions "
            "and no 'main' entry function"
        ) from None


# ---------------------------------------------------------------------------
# Symbolic-dim binding (shared by Interpreter.run and KernelContext.run_region)
# ---------------------------------------------------------------------------


def _bind_expr_leaves(bindings: Dict[str, int], expr: core.DimExpr, where: str) -> None:
    """Recurse into a DimExpr binding every resolvable ``Dim`` leaf.

    An already-bound leaf is left alone (the enclosing expression evaluation
    will surface inconsistencies); an unbound leaf with a known ``size`` is
    bound from that size. Unbound leaves without a known size stay unresolved
    — evaluating the expression then raises ``ShapeError`` naming the dim
    (the interpreter never guesses a dimension).
    """
    for leaf in (expr.left, expr.right):
        if isinstance(leaf, core.DimExpr):
            _bind_expr_leaves(bindings, leaf, where)
        elif isinstance(leaf, core.Dim):
            if leaf.name not in bindings and leaf.size is not None:
                bindings[leaf.name] = leaf.size


def _bind_dim(bindings: Dict[str, int], dim: Any, size: int, where: str) -> None:
    """Bind one declared shape entry against a concrete runtime dimension.

    - ``None`` -> unchecked (runtime-dynamic).
    - ``int`` -> must equal ``size`` (``ShapeError`` otherwise).
    - ``Dim`` -> bind name->size; an existing conflicting binding, or a known
      ``Dim.size`` disagreeing with ``size``, raises ``ShapeError``.
    - ``DimExpr`` -> bind resolvable ``Dim`` leaves, then evaluate the
      expression against ``bindings`` and compare to ``size`` (mismatch or
      unresolved dims raise ``ShapeError``).
    """
    if dim is None:
        return
    if isinstance(dim, int) and not isinstance(dim, bool):
        if dim != size:
            raise core.ShapeError(
                f"{where}: static dimension {dim} does not match runtime "
                f"dimension {size}"
            )
        return
    if isinstance(dim, core.Dim):
        existing = bindings.get(dim.name)
        if existing is not None:
            if existing != size:
                raise core.ShapeError(
                    f"{where}: dimension {dim.name!r} is already bound to "
                    f"{existing}, got {size}"
                )
        elif dim.size is not None:
            if dim.size != size:
                raise core.ShapeError(
                    f"{where}: dimension {dim.name!r} has known size "
                    f"{dim.size}, got runtime size {size}"
                )
            bindings[dim.name] = size
        else:
            bindings[dim.name] = size
        return
    if isinstance(dim, core.DimExpr):
        _bind_expr_leaves(bindings, dim, where)
        try:
            evaluated = shapes.evaluate_dim_expr(dim, bindings)
        except core.ShapeError as exc:
            raise core.ShapeError(
                f"{where}: cannot evaluate symbolic dimension {dim!r}: {exc}"
            ) from exc
        if evaluated != size:
            raise core.ShapeError(
                f"{where}: symbolic dimension {dim!r} evaluates to "
                f"{evaluated}, got runtime size {size}"
            )
        return
    raise core.ShapeError(
        f"{where}: unsupported shape entry {dim!r} of type "
        f"{type(dim).__name__}"
    )


def _bind_shape(
    bindings: Dict[str, int], declared: Sequence[Any], actual: Sequence[int], where: str
) -> None:
    """Walk a declared shape against a concrete shape pairwise, binding dims.

    Rank must match (``ShapeError`` otherwise); per-entry rules in
    :func:`_bind_dim`.
    """
    declared = tuple(declared)
    actual = tuple(actual)
    if len(declared) != len(actual):
        raise core.ShapeError(
            f"{where}: rank mismatch — declared rank {len(declared)} vs "
            f"runtime rank {len(actual)}"
        )
    for i, (dim, size) in enumerate(zip(declared, actual)):
        _bind_dim(bindings, dim, size, where=f"{where}, dim {i}")


def _shape_with_unchecked_nones(
    shape: Sequence[Any], bindings: Dict[str, int]
) -> Tuple[Optional[int], ...]:
    """Evaluate a symbolic shape, keeping ``None`` dims unchecked (as None)."""
    return tuple(
        None if dim is None else shapes.evaluate_dim_expr(dim, bindings)
        for dim in shape
    )


def _validate_result(
    ctx: "KernelContext", op: ir.Op, index: int, tensor: core.Tensor, value: ir.Value
) -> None:
    """Validate one kernel result against its declared IR result type.

    dtype must match exactly (``BackendError`` naming the op — kernels never
    silently coerce). Shape validation is elementwise (``ShapeError`` naming
    the op, result index, and offending dim):
    - rank must match exactly;
    - an expected ``None`` (runtime-dynamic) dim is UNCHECKED — skipped;
    - every other expected dim is evaluated against the symbolic-dim
      bindings and must equal the concrete runtime dim.
    """
    value_type = value.type
    if tensor.dtype != value_type.dtype:
        raise core.BackendError(
            f"kernel for op '{op.name}' produced result {index} with dtype "
            f"{tensor.dtype}, expected {value_type.dtype} — kernels must "
            "never silently coerce dtypes"
        )
    expected = _shape_with_unchecked_nones(value_type.shape, ctx.bindings)
    actual = tuple(tensor.shape)
    if len(actual) != len(expected):
        raise core.ShapeError(
            f"kernel for op '{op.name}' produced result {index} with shape "
            f"{actual}, expected {expected} — rank mismatch"
        )
    for dim, (want, got) in enumerate(zip(expected, actual)):
        if want is None:
            continue  # runtime-dynamic dim: unchecked by design
        if want != got:
            raise core.ShapeError(
                f"kernel for op '{op.name}' produced result {index} with "
                f"shape {actual}, expected {expected} — dim {dim}: got "
                f"{got}, expected {want}"
            )


class KernelContext:
    """Per-execution state handed to every kernel (see module docstring).

    Attribute contract (binding for all kernel modules):

    - ``bindings``: dim name -> concrete int (symbolic-dim bindings).
    - ``rank_context``: the effective ``RankContext`` (rank/world_size
      scalars resolve from it).
    - ``module``: the ``ir.Module`` being executed.
    - ``run_region(region, arg_tensors)``: execute a nested region.
    - ``compute_output_shapes(op, input_shapes, input_dtypes)``: evaluate the
      op's declared result shapes (None dims unchecked).
    - ``resolve_callback(callback_id)``: runtime_call callback lookup.
    - ``evaluate_shape(shape)``: evaluate a symbolic shape to concrete ints.
    """

    def __init__(
        self,
        bindings: Dict[str, int],
        rank_context: Any,
        module: ir.Module,
        interpreter: "Interpreter",
    ) -> None:
        self.bindings = bindings
        self.rank_context = rank_context
        self.module = module
        self._interpreter = interpreter

    def run_region(self, region: ir.Region, arg_tensors: List[core.Tensor]) -> List[core.Tensor]:
        """Bind the region's entry-block arguments and execute the block.

        Argument count must match the entry block's argument count
        (``BackendError`` otherwise). Each region-arg declared shape is
        walked pairwise against the concrete tensor shape, EXTENDING
        ``bindings`` (conflicting bindings => ``ShapeError``). The block's
        ops then execute in op order through the same dispatch loop, and the
        ``return`` terminator's operand tensors are returned.
        """
        block = region.entry
        if len(block.arguments) != len(arg_tensors):
            raise core.BackendError(
                f"region entry block expects {len(block.arguments)} "
                f"argument(s), got {len(arg_tensors)}"
            )
        for i, (argument, tensor) in enumerate(zip(block.arguments, arg_tensors)):
            _bind_shape(
                self.bindings,
                argument.type.shape,
                tensor.shape,
                where=f"region argument {i}",
            )
        return self._interpreter._run_block(block, arg_tensors)

    def compute_output_shapes(
        self, op: ir.Op, input_shapes: Any, input_dtypes: Any
    ) -> List[Tuple[Optional[int], ...]]:
        """Evaluate the op's declared result shapes against ``bindings``.

        SHAPE-RULE REUSE (mandated): the IR result types were ALREADY
        inferred by ops-level inference at trace time (the Builder ran the
        op's ``shape_fn``); this method only EVALUATES those symbolic shapes
        against the concrete dim bindings — the backend never re-runs
        ``shape_fn`` and carries no second copy of shape rules. ``None`` dims
        stay ``None`` (runtime-dynamic, unchecked).

        ``input_shapes`` / ``input_dtypes`` are accepted for kernel API
        stability and are unused here (inputs were already consumed by
        trace-time inference).
        """
        return [
            _shape_with_unchecked_nones(value.type.shape, self.bindings)
            for value in op.results
        ]

    def resolve_callback(self, callback_id: str) -> Callable[..., Any]:
        """Resolve a ``runtime_call`` callback identifier to its callable.

        Lazy import (import acyclicity): ``etl.ops.constant._get_callback``
        is the callback registry's internal lookup. Unknown ids raise
        ``core.BackendError`` naming the id (artifacts with ``runtime_call``
        require the same callback registrations at load time).
        """
        from etl.ops.constant import _get_callback

        callback = _get_callback(callback_id)
        if callback is None:
            raise core.BackendError(
                f"runtime_call: no callback registered under id "
                f"{callback_id!r} — the process must register the callback "
                "before loading/executing this program"
            )
        return callback

    def evaluate_shape(self, shape: Sequence[Any]) -> Tuple[int, ...]:
        """Evaluate a symbolic shape to concrete ints via ``self.bindings``."""
        return shapes.evaluate_shape(shape, self.bindings)


class Interpreter:
    """The numpy execution engine for one module (see module docstring).

    Attributes:
        module: the ``ir.Module`` to execute.
        signature: optional ``backends.Signature`` for input/output
            validation (count + dtype + symbolic-dim binding on inputs;
            count + dtype on outputs). ``None`` falls back to the entry
            function's block-arg types.
        kernels_registered: True once ``kernels.register_all()`` has
            populated the dispatch table (ensured in ``__init__``).
    """

    def __init__(
        self,
        module: ir.Module,
        signature: Any = None,
        kernels_registered: bool = False,
    ) -> None:
        self.module = module
        self.signature = signature
        if not kernels_registered:
            kernels.register_all()  # idempotent
        self.kernels_registered = True
        self._ctx: Optional[KernelContext] = None

    # ------------------------------------------------------------------ run

    def run(
        self, flat_input_tensors: List[core.Tensor], rank_context: Any = None
    ) -> List[core.Tensor]:
        """Execute the entry function on flat input tensors.

        Steps: (a) resolve the entry function; (b) validate the input count
        against the signature specs (or the entry block-arg count);
        (c) validate per-input dtype against the spec dtype
        (``DTypeError``); (d) build the symbolic-dim bindings by walking each
        spec shape against the concrete shape (conflicts => ``ShapeError``);
        (e) honor the optional ``rank_context`` override (thread-local, with
        restore in ``finally``); (f) execute the entry region; (g) validate
        output count + dtype against the signature output specs.
        """
        function = entry_function(self.module)
        if self.signature is not None:
            input_specs = tuple(self.signature.input_specs)
        else:
            input_specs = tuple(
                value.type for value in function.entry_block.arguments
            )
        if len(flat_input_tensors) != len(input_specs):
            raise core.BackendError(
                f"program expects {len(input_specs)} input tensor(s), got "
                f"{len(flat_input_tensors)}"
            )
        for i, tensor in enumerate(flat_input_tensors):
            if not isinstance(tensor, core.Tensor):
                raise core.BackendError(
                    f"input {i} must be a core.Tensor, got "
                    f"{type(tensor).__name__}"
                )
            if tensor.dtype != input_specs[i].dtype:
                raise core.DTypeError(
                    f"input {i}: expected dtype {input_specs[i].dtype}, got "
                    f"{tensor.dtype}"
                )
        bindings: Dict[str, int] = {}
        for i, (tensor, spec) in enumerate(zip(flat_input_tensors, input_specs)):
            _bind_shape(
                bindings, spec.shape, tensor.shape, where=f"input {i}"
            )

        previous: Any = None
        if rank_context is not None:
            previous = get_rank_context()
            set_rank_context(rank_context)
        try:
            ctx = KernelContext(
                bindings=bindings,
                rank_context=get_rank_context(),
                module=self.module,
                interpreter=self,
            )
            self._ctx = ctx
            outputs = self._run_block(function.entry_block, flat_input_tensors)
        finally:
            self._ctx = None
            if previous is not None:
                set_rank_context(previous)

        if self.signature is not None:
            output_specs = tuple(self.signature.output_specs)
            if len(outputs) != len(output_specs):
                raise core.BackendError(
                    f"program produced {len(outputs)} output tensor(s), "
                    f"expected {len(output_specs)}"
                )
            for i, (tensor, spec) in enumerate(zip(outputs, output_specs)):
                if tensor.dtype != spec.dtype:
                    raise core.BackendError(
                        f"output {i}: expected dtype {spec.dtype}, got "
                        f"{tensor.dtype}"
                    )
        return outputs

    # ----------------------------------------------------------- op dispatch

    def _run_block(
        self, block: ir.Block, arg_tensors: List[core.Tensor]
    ) -> List[core.Tensor]:
        """Execute one block's ops in order; return the terminator's tensors.

        The value environment maps ``value.id`` (module-unique ints) to
        computed ``core.Tensor``s. The ``return`` terminator is special-cased
        in the loop (NOT dispatched). Every other op dispatches through
        ``kernels.dispatch(op_name)``; results are normalized to a tuple and
        validated against the op's declared result types (dtype exactly,
        shape with ``None`` dims unchecked). Nested regions reuse the SAME
        loop via ``KernelContext.run_region``.
        """
        ctx = self._ctx
        assert ctx is not None, "Interpreter._run_block requires an active KernelContext"
        env: Dict[int, core.Tensor] = {
            argument.id: tensor
            for argument, tensor in zip(block.arguments, arg_tensors)
        }
        for op in block.ops:
            if op.name == "return":
                return [env[value.id] for value in op.operands]
            kernel = kernels.dispatch(op.name)
            operands = tuple(env[value.id] for value in op.operands)
            result = kernel(ctx, op, operands)
            results = (result,) if isinstance(result, core.Tensor) else tuple(result)
            if len(results) != len(op.results):
                raise core.BackendError(
                    f"kernel for op '{op.name}' produced {len(results)} "
                    f"result(s), expected {len(op.results)}"
                )
            for i, (tensor, value) in enumerate(zip(results, op.results)):
                _validate_result(ctx, op, i, tensor, value)
                env[value.id] = tensor
        raise core.BackendError(
            "block executed to completion without a 'return' terminator — "
            "the module is invalid (verify should have rejected it)"
        )
