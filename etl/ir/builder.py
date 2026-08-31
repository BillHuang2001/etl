"""The Builder — EvoXIR's op-construction API.

A Builder emits ops into a ``Module`` via an insertion-point stack. The
frontend ``ops`` module obtains the active builder from
``trace.current_builder()`` and calls ``emit``/``create``; ALL IR mutation
funnels through this class so invariants (id assignment, parent wiring,
operand use bookkeeping, result-type inference) are enforced in exactly one
place.

Contracts enforced here (cheap, eager checks; deep structural validation is
still ``verify``'s job):

* unknown op names raise ``KeyError`` (from the registry lookup);
* arity, region-count, and attribute-schema violations raise
  ``VerificationError`` before any IR is mutated;
* op/value ids are assigned from the owning module's counters;
* result types resolve in order: explicit ``result_types`` → the ``OpDef``'s
  ``shape_fn`` → op-specific resolution (``constant``/``call``/``if``/
  ``runtime_call``/``block_call`` — see ``inference.py``), else
  ``VerificationError`` demanding explicit types;
* a ``ShapeError`` raised by a ``shape_fn`` propagates with the op's source
  location appended to its message (when a real location is known) — it is a
  precise inference failure, not a structural violation.

Attribute-validation details (binding for ``verify``'s agreement checks):

* ``attributes`` may be ``None`` (treated as empty); a copy with declared
  defaults applied is stored on the op.
* ``ATTR_DTYPE`` values accept anything ``etl.core.dtype`` accepts and are
  normalized to the numpy dtype *name* string (the serialization format).
* ``ATTR_INT``/``ATTR_INTS`` accept ``None`` only for attributes whose
  ``AttrSpec`` declares ``default=None`` (the documented nullable attrs, e.g.
  ``argmax.axis``, ``transpose.permutation``).
* ``ATTR_FLOAT`` accepts ints and floats (bools rejected); the sequence tags
  (``INTS``/``FLOATS``/``STRS``/``NESTED_INTS``/``SHAPE``) accept tuples or
  lists and are normalized to tuples; ``ATTR_SHAPE`` entries must be
  ``int | Dim | DimExpr | None``.
* ``result_specs`` (``runtime_call``/``block_call``/``external_call``) is a
  sequence of ``ValueType`` | ``TensorSpec`` | ``{"dtype": ..., "shape": ...}``
  entries, converted to ``ValueType``s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from etl.core import (
    DTypeError,
    Dim,
    DimExpr,
    ShapeError,
    TensorSpec,
    VerificationError,
    dtype,
)

from .block import Block
from .function import Function
from .location import Location
from .module import Module
from .op import Op
from .op_defs import (
    ATTR_ANY,
    ATTR_BOOL,
    ATTR_DTYPE,
    ATTR_FLOAT,
    ATTR_FLOATS,
    ATTR_INT,
    ATTR_INTS,
    ATTR_NDARRAY,
    ATTR_NESTED_INTS,
    ATTR_SHAPE,
    ATTR_STR,
    ATTR_STRS,
    OpDef,
    opdef,
)
from .region import Region
from .types import ValueType
from .value import Use, Value


def _is_int(value: Any) -> bool:
    """True for plain ints (bools excluded — numpy-style tag checks)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    """True for ints/floats (bools excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass
class InsertionPoint:
    """Internal: where the builder currently appends ops.

    Attributes:
        block: The target block.
        position: Index into ``block.ops`` where new ops are inserted.
    """

    block: Block
    position: int


class Builder:
    """Constructs IR inside a ``Module``.

    Contract for every emission method: resolve the op's ``OpDef`` (unknown
    name raises immediately); check arity, attribute schema, and region count
    (cheap failures raise ``VerificationError`` early — full validation is
    still ``verify``'s job); assign module-unique ids from the owning module's
    counters; infer result types via ``OpDef.shape_fn`` unless explicit
    ``result_types`` are given (``shape_fn=None`` means op-specific resolution
    or mandatory explicit types — see ``inference.py``); create result
    ``Value``s, wire operand ``Use``s, set parent pointers.
    """

    def __init__(self, module: Optional[Module] = None) -> None:
        """Create a builder; ``module`` may be None until ``build_module``."""
        self.module = module
        self._insertion_stack: list[InsertionPoint] = []

    # --- insertion-point management ------------------------------------------

    def _current_insertion_point(self) -> InsertionPoint:
        """The top of the insertion-point stack.

        Raises:
            RuntimeError: If no insertion point has been set.
        """
        if not self._insertion_stack:
            raise RuntimeError(
                "Builder has no insertion point: call build_function or "
                "set_insertion_point first"
            )
        return self._insertion_stack[-1]

    @property
    def current_block(self) -> Block:
        """The block new ops are inserted into.

        Raises:
            RuntimeError: If no insertion point has been set.
        """
        return self._current_insertion_point().block

    @property
    def current_region(self) -> Region:
        """The region containing the current insertion point.

        Raises:
            RuntimeError: If no insertion point has been set, or the current
                block is detached (no parent region).
        """
        region = self.current_block.parent
        if region is None:
            raise RuntimeError("the current block is detached (no parent region)")
        return region

    def set_insertion_point(self, target: Block | Region) -> None:
        """Set the insertion point: Block -> its start; Region -> its entry
        block's start. Replaces the current insertion point (the stack must
        already have at least one entry).

        Raises:
            RuntimeError: If no insertion point has been set yet.
            TypeError: If ``target`` is neither a ``Block`` nor a ``Region``.
        """
        if isinstance(target, Region):
            block = target.entry
        elif isinstance(target, Block):
            block = target
        else:
            raise TypeError(
                "set_insertion_point expects a Block or Region, got "
                f"{type(target).__name__}"
            )
        self._current_insertion_point()  # stack must have >= 1 entry
        self._insertion_stack[-1] = InsertionPoint(block, 0)

    def push_region(self, region: Region) -> None:
        """Push ``region``'s entry block (at its start) onto the
        insertion-point stack (for building nested if/while bodies);
        ``pop_region`` restores."""
        self._insertion_stack.append(InsertionPoint(region.entry, 0))

    def pop_region(self) -> Region:
        """Pop the insertion-point stack, returning the region left behind.

        Raises:
            RuntimeError: If the insertion-point stack is empty, or the popped
                block is detached (no parent region).
        """
        if not self._insertion_stack:
            raise RuntimeError("pop_region: the insertion-point stack is empty")
        point = self._insertion_stack.pop()
        region = point.block.parent
        if region is None:
            raise RuntimeError(
                "pop_region: the popped block is detached (no parent region)"
            )
        return region

    # --- module/function/region construction -----------------------------------

    def _require_module(self) -> Module:
        """The attached module.

        Raises:
            RuntimeError: If no module is attached to this builder.
        """
        if self.module is None:
            raise RuntimeError("Builder has no module: call build_module first")
        return self.module

    @staticmethod
    def _check_input_types(input_types: Any) -> tuple[ValueType, ...]:
        """Normalize/validate a sequence of input ``ValueType``s."""
        types = tuple(input_types)
        for value_type in types:
            if not isinstance(value_type, ValueType):
                raise TypeError(
                    "input_types must be ValueType instances, got "
                    f"{value_type!r}"
                )
        return types

    def _block_arguments(
        self, block: Block, input_types: tuple[ValueType, ...]
    ) -> tuple[Value, ...]:
        """Fresh block-argument Values (ids from the module's counters)."""
        module = self._require_module()
        return tuple(
            Value(id=module.new_value_id(), type=value_type, owner=block, index=i)
            for i, value_type in enumerate(input_types)
        )

    def build_module(self, name: str = "main", metadata: dict[str, Any] | None = None) -> Module:
        """Create a new ``Module``, attach it to this builder, reset the
        insertion-point stack, and return it."""
        self.module = Module(name=name, metadata={} if metadata is None else metadata)
        self._insertion_stack = []
        return self.module

    def build_function(
        self,
        name: str,
        input_types: tuple[ValueType, ...],
        metadata: dict[str, Any] | None = None,
    ) -> Function:
        """Create a function with a single-block region whose block arguments
        are fresh ``Value``s of ``input_types``; set the insertion point to its
        block. The caller must emit a terminator.

        Raises:
            RuntimeError: If no module is attached to this builder.
        """
        module = self._require_module()
        types = self._check_input_types(input_types)
        region = Region(blocks=[])
        block = Block()
        block.arguments = self._block_arguments(block, types)
        region.append_block(block)  # wires block.parent = region
        function = Function(
            name=name,
            input_types=types,
            region=region,
            metadata={} if metadata is None else metadata,
        )  # Function.__post_init__ wires region.parent = function
        module.add_function(function)
        point = InsertionPoint(block, 0)
        if self._insertion_stack:
            self._insertion_stack[-1] = point
        else:
            self._insertion_stack.append(point)
        return function

    def build_region(self, input_types: tuple[ValueType, ...] = ()) -> Region:
        """Create a detached single-block region whose entry-block arguments
        are fresh ``Value``s of ``input_types`` — the body of an upcoming
        `if`/`while` op (pass it via ``create(..., regions=(...))``).

        Raises:
            RuntimeError: If no module is attached to this builder.
        """
        self._require_module()
        types = self._check_input_types(input_types)
        region = Region(blocks=[])  # detached: parent stays None
        block = Block()
        block.arguments = self._block_arguments(block, types)
        region.append_block(block)
        return region

    def insert_block(self, region: Region, position: int | None = None) -> Block:
        """Create a new empty block in ``region`` at ``position`` (None =
        append) and return it. Does not change the insertion point."""
        block = Block()
        if position is None:
            region.append_block(block)
        else:
            region.insert_block(position, block)
        return block

    # --- attribute validation ---------------------------------------------------

    def _validate_attributes(
        self, definition: OpDef, attributes: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Validate/normalize ``attributes`` against the op's schema.

        Returns a copy with declared defaults applied and values normalized
        per their ``ATTR_*`` tags.

        Raises:
            VerificationError: Not a dict, unknown keys, missing required
                attributes, or values not matching their declared tag.
        """
        if attributes is None:
            raw: dict[str, Any] = {}
        elif isinstance(attributes, dict):
            raw = attributes
        else:
            raise VerificationError(
                f"op '{definition.name}': attributes must be a dict, got "
                f"{type(attributes).__name__}"
            )
        schema = {spec.name: spec for spec in definition.attributes}
        unknown = sorted(set(raw) - set(schema))
        if unknown:
            raise VerificationError(
                f"op '{definition.name}': unknown attribute(s) {unknown}; "
                f"declared: {sorted(schema)}"
            )
        checked: dict[str, Any] = {}
        for spec in definition.attributes:
            if spec.name in raw:
                value = raw[spec.name]
            elif spec.required:
                raise VerificationError(
                    f"op '{definition.name}': missing required attribute "
                    f"'{spec.name}'"
                )
            else:
                checked[spec.name] = spec.default
                continue
            checked[spec.name] = self._check_attr_value(definition.name, spec, value)
        return checked

    def _check_attr_value(self, op_name: str, spec: Any, value: Any) -> Any:
        """Check one attribute value against its ``AttrSpec`` tag."""
        tag = spec.type
        where = f"attribute '{spec.name}' of op '{op_name}'"
        if tag == ATTR_BOOL:
            if not isinstance(value, bool):
                raise VerificationError(f"{where}: expected a bool, got {value!r}")
            return value
        if tag == ATTR_INT:
            if not _is_int(value) and not (value is None and spec.default is None):
                raise VerificationError(f"{where}: expected an int, got {value!r}")
            return value
        if tag == ATTR_FLOAT:
            if not _is_number(value):
                raise VerificationError(f"{where}: expected a float, got {value!r}")
            return value
        if tag == ATTR_STR:
            if not isinstance(value, str):
                raise VerificationError(f"{where}: expected a str, got {value!r}")
            return value
        if tag == ATTR_DTYPE:
            try:
                return dtype(value).name  # normalize to the name string
            except DTypeError as exc:
                raise VerificationError(
                    f"{where}: expected a dtype (numpy dtype name), got {value!r}"
                ) from exc
        if tag == ATTR_INTS:
            return self._check_sequence(where, value, _is_int, "int",
                                        allow_none=spec.default is None)
        if tag == ATTR_FLOATS:
            return self._check_sequence(where, value, _is_number, "float")
        if tag == ATTR_STRS:
            return self._check_sequence(where, value, lambda v: isinstance(v, str), "str")
        if tag == ATTR_NESTED_INTS:
            if not isinstance(value, (tuple, list)):
                raise VerificationError(
                    f"{where}: expected a sequence of int tuples, got {value!r}"
                )
            pairs = []
            for pair in value:
                if not isinstance(pair, (tuple, list)) or not pair or not all(
                    _is_int(entry) for entry in pair
                ):
                    raise VerificationError(
                        f"{where}: expected int tuples, got entry {pair!r}"
                    )
                pairs.append(tuple(pair))
            return tuple(pairs)
        if tag == ATTR_SHAPE:
            if not isinstance(value, (tuple, list)):
                raise VerificationError(
                    f"{where}: expected a tuple of shape dims, got {value!r}"
                )
            return self._check_shape_dims(where, tuple(value))
        if tag == ATTR_NDARRAY:
            if not isinstance(value, np.ndarray):
                raise VerificationError(
                    f"{where}: expected a numpy array, got {type(value).__name__}"
                )
            return value
        if tag == ATTR_ANY:
            return value  # JSON-ability is documented per op
        raise VerificationError(f"{where}: unknown attribute tag {tag!r}")

    @staticmethod
    def _check_sequence(
        where: str, value: Any, predicate: Any, expected: str, allow_none: bool = False
    ) -> tuple | None:
        """Validate a tuple/list of entries matching ``predicate``.

        Returns a tuple (or ``None`` when ``allow_none`` and the value is
        None).
        """
        if value is None and allow_none:
            return None
        if not isinstance(value, (tuple, list)):
            raise VerificationError(
                f"{where}: expected a sequence of {expected}s, got {value!r}"
            )
        seq = tuple(value)
        for entry in seq:
            if not predicate(entry):
                raise VerificationError(
                    f"{where}: expected {expected}s, got entry {entry!r}"
                )
        return seq

    @staticmethod
    def _check_shape_dims(where: str, dims: tuple) -> tuple:
        """Validate shape entries: ``int | Dim | DimExpr | None``."""
        checked = []
        for dim in dims:
            if _is_int(dim) or isinstance(dim, (Dim, DimExpr)):
                checked.append(dim)
            elif dim is None:
                checked.append(None)
            else:
                raise VerificationError(f"{where}: invalid shape dim {dim!r}")
        return tuple(checked)

    # --- result-type resolution --------------------------------------------------

    def _resolve_result_types(
        self,
        definition: OpDef,
        op_name: str,
        operands: tuple[Value, ...],
        attrs: dict[str, Any],
        regions: tuple[Region, ...],
        explicit: tuple[ValueType, ...] | None,
        location: Location | None,
    ) -> tuple[ValueType, ...]:
        """Resolve result types: explicit → shape_fn → op-specific → error.

        A ``ShapeError`` raised by a ``shape_fn`` propagates with the op's
        source location appended to its message (when a real location is
        known — never for ``Location.unknown()``).
        """
        if explicit is not None:
            types = tuple(explicit)
            if not all(isinstance(t, ValueType) for t in types):
                raise VerificationError(
                    f"op '{op_name}': result_types must be ValueType "
                    f"instances, got {types!r}"
                )
            return types
        if definition.shape_fn is not None:
            input_types = tuple(operand.type for operand in operands)
            try:
                return tuple(definition.shape_fn(input_types, attrs))
            except ShapeError as exc:
                if (
                    location is not None
                    and location.line > 0
                    and location.file != "<unknown>"
                ):
                    suffix = f" (at {location})"
                    if suffix not in exc.args[0]:
                        exc.args = (f"{exc.args[0]}{suffix}",) + exc.args[1:]
                raise
        if op_name == "constant":
            return (self._constant_result_type(attrs),)
        if op_name == "call":
            return self._call_result_types(attrs)
        if op_name == "if":
            return self._if_result_types(regions)
        if op_name in ("runtime_call", "block_call", "external_call"):
            return self._specs_result_types(op_name, attrs["result_specs"])
        raise VerificationError(
            f"op '{op_name}': no shape_fn and no op-specific result-type rule; "
            "pass explicit result_types"
        )

    @staticmethod
    def _constant_result_type(attrs: dict[str, Any]) -> ValueType:
        """Result type of ``constant``: dtype/shape from the payload array."""
        payload = attrs["value"]  # schema guarantees a required ndarray
        return ValueType(np.dtype(payload.dtype), tuple(int(d) for d in payload.shape))

    def _call_result_types(self, attrs: dict[str, Any]) -> tuple[ValueType, ...]:
        """Result types of ``call``: the callee's output signature.

        Raises:
            VerificationError: Unknown callee, or the callee has no
                terminator (pass explicit ``result_types`` in that case).
        """
        module = self._require_module()
        callee_name = attrs["callee"]
        try:
            callee = module.get_function(callee_name)
        except KeyError as exc:
            raise VerificationError(
                f"op 'call': no function named '{callee_name}' in module "
                f"'{module.name}'"
            ) from exc
        try:
            return tuple(callee.output_types)
        except ValueError as exc:
            raise VerificationError(
                f"op 'call': callee '{callee_name}' has no terminator; "
                "pass explicit result_types"
            ) from exc

    def _if_result_types(self, regions: tuple[Region, ...]) -> tuple[ValueType, ...]:
        """Result types of ``if``: the operand types of each region's
        ``return`` terminator; both branches must agree.

        Raises:
            VerificationError: Wrong region count, a missing entry block or
                terminator, or mismatching branch result types.
        """
        if len(regions) != 2:
            raise VerificationError(
                "op 'if' requires exactly 2 regions (true, false)"
            )
        true_types = self._branch_result_types("true", regions[0])
        false_types = self._branch_result_types("false", regions[1])
        if true_types != false_types:
            raise VerificationError(
                "op 'if': branch result types differ — "
                f"true: {true_types}, false: {false_types}"
            )
        return true_types

    def _branch_result_types(self, branch: str, region: Region) -> tuple[ValueType, ...]:
        """The yielded operand types of one if-branch's ``return``."""
        try:
            block = region.entry
        except ValueError:
            raise VerificationError(
                f"op 'if': {branch} region has no entry block; "
                "pass explicit result_types"
            ) from None
        terminator = block.terminator
        if terminator is None:
            raise VerificationError(
                f"op 'if': {branch} region has no 'return' terminator; "
                "pass explicit result_types"
            )
        return tuple(value.type for value in terminator.operands)

    def _specs_result_types(
        self, op_name: str, specs: Any
    ) -> tuple[ValueType, ...]:
        """Convert declared ``result_specs`` entries into ``ValueType``s."""
        if not isinstance(specs, (tuple, list)):
            raise VerificationError(
                f"op '{op_name}': 'result_specs' must be a sequence of result "
                f"specs, got {specs!r}"
            )
        return tuple(self._spec_to_type(op_name, spec) for spec in specs)

    def _spec_to_type(self, op_name: str, spec: Any) -> ValueType:
        """Convert one result spec: ``ValueType`` | ``TensorSpec`` | dict."""
        if isinstance(spec, ValueType):
            return spec
        if isinstance(spec, TensorSpec):
            return ValueType(spec.dtype, tuple(spec.shape))
        if isinstance(spec, dict):
            try:
                value_dtype = dtype(spec["dtype"])
                shape = tuple(spec["shape"])
            except (KeyError, TypeError, DTypeError) as exc:
                raise VerificationError(
                    f"op '{op_name}': result spec must map 'dtype'/'shape', "
                    f"got {spec!r}"
                ) from exc
            return ValueType(
                value_dtype,
                self._check_shape_dims(f"op '{op_name}' result spec", shape),
            )
        raise VerificationError(
            f"op '{op_name}': cannot interpret result spec {spec!r} — expected "
            "a ValueType, a TensorSpec, or a {'dtype': ..., 'shape': ...} dict"
        )

    # --- op emission -----------------------------------------------------------

    def _prepare_op(
        self,
        definition: OpDef,
        op_name: str,
        operands: tuple[Value, ...],
        attributes: dict[str, Any] | None,
        result_types: tuple[ValueType, ...] | None,
        regions: tuple[Region, ...],
        location: Location | None,
    ) -> Op:
        """Run all cheap checks and build an (not yet inserted) ``Op``.

        Arity, operand kind, region count, attribute schema, and result-type
        resolution are validated; ids are assigned from the module's counters;
        operand ``Use``s and region parent pointers are wired. The caller
        inserts the op into a block (which wires ``op.parent``).
        """
        ops = tuple(operands)
        regs = tuple(regions)
        if not definition.check_arity(len(ops)):
            raise VerificationError(
                f"op '{op_name}' expects arity {definition.arity}, got "
                f"{len(ops)} operand(s)"
            )
        for operand in ops:
            if not isinstance(operand, Value):
                raise VerificationError(
                    f"op '{op_name}': operand {operand!r} is not an SSA Value"
                )
        if len(regs) != definition.regions:
            raise VerificationError(
                f"op '{op_name}' declares {definition.regions} region(s), "
                f"got {len(regs)}"
            )
        attrs = self._validate_attributes(definition, attributes)
        resolved = self._resolve_result_types(
            definition, op_name, ops, attrs, regs, result_types, location
        )
        if (
            isinstance(definition.result_count, int)
            and len(resolved) != definition.result_count
        ):
            raise VerificationError(
                f"op '{op_name}' declares {definition.result_count} result(s), "
                f"got {len(resolved)}"
            )
        module = self._require_module()
        op = Op(
            name=op_name,
            id=module.new_op_id(),
            operands=ops,
            attributes=attrs,
            regions=regs,
            location=location,
        )
        op.results = tuple(
            Value(id=module.new_value_id(), type=result_type, owner=op, index=i)
            for i, result_type in enumerate(resolved)
        )
        for region in regs:
            region.parent = op
        for i, operand in enumerate(ops):
            operand.add_use(Use(op, i))
        return op

    def create(
        self,
        op_name: str,
        operands: tuple[Value, ...] = (),
        attributes: dict[str, Any] | None = None,
        result_types: tuple[ValueType, ...] | None = None,
        location: Location | None = None,
        regions: tuple[Region, ...] = (),
    ) -> Op:
        """Create an op at the current insertion point and return it.

        Result types: ``result_types`` if given, else ``OpDef.shape_fn``; if
        neither can resolve them (``shape_fn=None``, op-specific resolution
        unavailable) raise ``VerificationError`` demanding explicit types.
        Result ``Value``s get fresh ids from the module counters and their
        ``owner``/``index`` set to this op. Operand ``Use``s and region parent
        pointers are wired. ``attributes`` is validated against the op's
        attribute schema (required keys present, types tagged correctly).

        Raises:
            RuntimeError: If no insertion point has been set.
        """
        definition = opdef(op_name)  # KeyError propagates for unknown names
        op = self._prepare_op(
            definition, op_name, operands, attributes, result_types, regions, location
        )
        point = self._current_insertion_point()
        point.block.insert(point.position, op)  # wires op.parent
        point.position += 1
        return op

    def emit(
        self,
        op_name: str,
        operands: tuple[Value, ...] = (),
        attributes: dict[str, Any] | None = None,
        result_type: ValueType | None = None,
        location: Location | None = None,
        regions: tuple[Region, ...] = (),
    ) -> Value:
        """Single-result convenience: ``create(...)`` then ``op.result``.

        Raises:
            ValueError: If the op does not have exactly one result.
        """
        result_types = None if result_type is None else (result_type,)
        op = self.create(
            op_name,
            operands=operands,
            attributes=attributes,
            result_types=result_types,
            location=location,
            regions=regions,
        )
        return op.result

    def set_terminator(
        self,
        block: Block,
        op_name: str,
        operands: tuple[Value, ...] = (),
        attributes: dict[str, Any] | None = None,
        location: Location | None = None,
    ) -> Op:
        """Append a terminator op (``return``) to ``block``.

        The op is appended (never inserted at the current insertion point), so
        it is last by construction; the insertion-point stack is untouched.

        Raises:
            VerificationError: If ``op_name`` is not a terminator, ``block``
                already has one, or the terminator is not last.
        """
        definition = opdef(op_name)  # KeyError propagates for unknown names
        if not definition.is_terminator:
            raise VerificationError(f"op '{op_name}' is not a terminator")
        if any(op.is_terminator for op in block.ops):
            raise VerificationError("block already has a terminator")
        op = self._prepare_op(
            definition, op_name, operands, attributes, (), (), location
        )
        block.append(op)  # appended → last by construction; wires op.parent
        return op
