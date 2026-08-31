"""Module verification.

``verify(module)`` is the IR contract checker: it validates structural
invariants, type agreement, attribute schemas, SSA well-formedness, and v1
restrictions, raising ``VerificationError`` (owned by ``etl.core``,
re-exported by this package) with a source-location-annotated message on the
FIRST violation (no silent recovery, no partial reporting in v1).

Attribute-validation policy (binding; identical tag-for-tag to the Builder's
acceptance rules — see ``builder.py``, "Attribute-validation details"):

* ``ATTR_DTYPE`` values accept anything ``etl.core.dtype`` accepts (numpy
  dtype objects, name strings, scalar types, objects exposing ``.dtype``; the
  Builder normalizes them to the name string).
* ``ATTR_INT``/``ATTR_INTS`` accept ``None`` only for attributes whose
  ``AttrSpec`` declares ``default=None`` (the documented nullable attrs, e.g.
  ``argmax.axis``, ``transpose.permutation``, ``slice.strides``,
  ``conv.strides``/``input_dilation``/``kernel_dilation``).
* ``ATTR_FLOAT`` accepts ints and floats (bools rejected); the sequence tags
  (``INTS``/``FLOATS``/``STRS``/``NESTED_INTS``/``SHAPE``) accept tuples or
  lists; ``ATTR_NESTED_INTS`` entries must be NON-EMPTY int sequences;
  ``ATTR_SHAPE`` entries must be ``int | Dim | DimExpr | None``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from etl.core import DTypeError, Dim, DimExpr, ShapeError, VerificationError, dtype

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
    AttrSpec,
    OpDef,
    opdef,
)
from .region import Region
from .types import ValueType
from .value import Use, Value
from .version import IR_FORMAT_VERSION

__all__ = ["verify", "VerificationError"]


def verify(module: Module) -> None:
    """Validate ``module`` against all EvoXIR invariants.

    Raises:
        VerificationError: On the first violation, with source location when
            one exists.

    Checked invariants (binding):

    *Module level*
    - ``module.version`` == ``IR_FORMAT_VERSION``.
    - At least one function; function names unique; each function's ``parent``
      is the module.

    *Function level*
    - The function region holds exactly one block (v1 restriction;
      multi-block regions are reserved for future versions).
    - Entry-block argument count/types match ``function.input_types`` exactly.
    - The entry block ends with a terminator, which is the ``return`` op and
      is the LAST op (nothing after a terminator).
    - ``function.output_types`` (return operands) is consistent.

    *Region/block level (function regions AND nested op regions)*
    - Every region has at least one block; each block's ``parent`` is its
      region; each op's ``parent`` is its block.
    - Nested regions of an op: count matches the op's ``OpDef.regions``;
      entry-block argument types match the op's operand types (v1 binding
      convention).

    *Op level*
    - ``op.name`` is registered (unknown name fails).
    - Operand count within the ``OpDef`` arity; region count matches; result
      count within the declared ``result_count`` (when not None).
    - Attributes match the schema: no unknown keys, all required keys present,
      each value's type matches its ``AttrSpec`` tag — acceptance policy
      identical to the Builder's (see module docstring; in particular
      ``ATTR_INT``/``ATTR_INTS`` values may be ``None`` only when the
      ``AttrSpec`` declares ``default=None``).
    - Results are ``Value``s owned by this op, with module-unique ids.
    - Result types agree with ``OpDef.shape_fn(input_types, attributes)`` when
      ``shape_fn`` is not None (ops with ``shape_fn=None`` must record
      consistent types from op-specific resolution).

    *Value/SSA level*
    - Value ids unique across the module.
    - Operands are defined before use: block arguments of an enclosing block,
      or results of ops earlier in the same or an enclosing block
      (SSA dominance). No use of a value from an unrelated region.
    - Use bookkeeping is consistent: every ``Use`` recorded on a value
      actually refers to that value at that operand index, and every operand
      of every op has a matching recorded ``Use``.

    *Effect ordering*
    - Verification does NOT reorder anything; it only checks the structural
      invariants above. Effectful-op ordering is positional by construction
      (see CONTEXT.md, "Effect model").
    """
    if not isinstance(module, Module):
        raise TypeError(
            f"verify() expects an ir.Module, got {type(module).__name__}"
        )
    _verify_module(module)
    for function in module.functions:
        _verify_function(module, function)
    values, ops = _collect_ids(module)
    _verify_uses(values, ops)


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def _fail(message: str, location: Location | None = None) -> None:
    """Raise ``VerificationError``, annotating with the source location."""
    if location is not None:
        raise VerificationError(f"{message} [at {location}]")
    raise VerificationError(message)


def _value_desc(value: Any) -> str:
    """Short human-readable description of an attribute value's type."""
    if isinstance(value, (tuple, list)):
        return f"{type(value).__name__} with {len(value)} entries"
    return type(value).__name__


# ---------------------------------------------------------------------------
# Module level
# ---------------------------------------------------------------------------

def _verify_module(module: Module) -> None:
    if module.version != IR_FORMAT_VERSION:
        _fail(
            f"module '{module.name}': version {module.version} does not match "
            f"the IR format version {IR_FORMAT_VERSION}"
        )
    if not module.functions:
        _fail(f"module '{module.name}': must contain at least one function")
    seen: set[str] = set()
    for function in module.functions:
        if not isinstance(function, Function):
            _fail(
                f"module '{module.name}': functions must be Function "
                f"instances, got {type(function).__name__}"
            )
        if function.name in seen:
            _fail(f"module '{module.name}': duplicate function name '{function.name}'")
        seen.add(function.name)
        if function.parent is not module:
            _fail(
                f"function '{function.name}': parent is not module "
                f"'{module.name}'"
            )


# ---------------------------------------------------------------------------
# Function level
# ---------------------------------------------------------------------------

def _verify_function(module: Module, function: Function) -> None:
    fname = function.name
    region = function.region
    if not isinstance(region, Region):
        _fail(
            f"function '{fname}': region must be a Region, got "
            f"{type(region).__name__}"
        )
    if len(region.blocks) != 1:
        if not region.blocks:
            _fail(f"function '{fname}': region has no blocks")
        _fail(
            f"function '{fname}': region has {len(region.blocks)} blocks; "
            "v1 requires exactly one"
        )
    if region.parent is not function:
        _fail(f"function '{fname}': region.parent is not the function")
    block = region.blocks[0]
    if not isinstance(block, Block):
        _fail(
            f"function '{fname}': entry block must be a Block, got "
            f"{type(block).__name__}"
        )
    if block.parent is not region:
        _fail(f"function '{fname}': entry block.parent is not the function region")
    if len(block.arguments) != len(function.input_types):
        _fail(
            f"function '{fname}': entry block has {len(block.arguments)} "
            f"arguments, expected {len(function.input_types)}"
        )
    for i, (arg, expected) in enumerate(zip(block.arguments, function.input_types)):
        if not isinstance(arg, Value):
            _fail(
                f"function '{fname}': entry argument {i} must be a Value, got "
                f"{type(arg).__name__}"
            )
        if not isinstance(expected, ValueType):
            _fail(
                f"function '{fname}': input type {i} must be a ValueType, got "
                f"{type(expected).__name__}"
            )
        if arg.type != expected:
            _fail(
                f"function '{fname}': entry argument {i} type {arg.type} does "
                f"not match input type {expected}"
            )
    _verify_block_terminator(block, f"function '{fname}'")
    # output_types consistency: recompute from the return terminator and
    # compare with the Function property (trivially consistent by
    # construction, but the property must not raise).
    terminator = block.ops[-1]
    for i, operand in enumerate(terminator.operands):
        if not isinstance(operand, Value):
            _fail(
                f"function '{fname}': return operand {i} must be a Value, got "
                f"{type(operand).__name__}"
            )
    expected_outputs = tuple(value.type for value in terminator.operands)
    try:
        actual = function.output_types
    except ValueError as error:
        _fail(f"function '{fname}': output_types unavailable: {error}")
    if actual != expected_outputs:
        _fail(
            f"function '{fname}': output_types {actual} disagree with the "
            f"return operands' types {expected_outputs}"
        )
    _verify_block_ops(module, block, frozenset())


# ---------------------------------------------------------------------------
# Region/block level (depth-first walk)
# ---------------------------------------------------------------------------

def _verify_block_terminator(block: Block, context: str) -> None:
    """Every block must end with the ``return`` terminator, and nothing may
    follow a terminator."""
    ops = block.ops
    if not ops:
        _fail(f"{context}: block has no ops (missing 'return' terminator)")
    terminator_indexes: list[int] = []
    for index, op in enumerate(ops):
        if not isinstance(op, Op):
            _fail(
                f"{context}: block op {index} must be an Op, got "
                f"{type(op).__name__}"
            )
        try:
            opdef_obj = opdef(op.name)
        except KeyError:
            continue  # unknown names are reported by the op-level check
        if opdef_obj.is_terminator:
            terminator_indexes.append(index)
    if not terminator_indexes:
        last = ops[-1]
        try:
            opdef(last.name)
        except KeyError:
            _fail(
                f"{context}: no terminator (last op '{last.name}' is not a "
                "registered op)"
            )
        _fail(f"{context}: no 'return' terminator")
    for index in terminator_indexes:
        if index != len(ops) - 1:
            _fail(
                f"{context}: terminator '{ops[index].name}' is not the last op"
            )
    if ops[-1].name != "return":
        _fail(f"{context}: terminator must be the 'return' op, got '{ops[-1].name}'")


def _verify_block_ops(module: Module, block: Block, scope: frozenset[int]) -> None:
    """Check every op of ``block`` in order, then recurse into nested regions.

    ``scope`` holds the ids of all values defined before this block (block
    arguments of enclosing blocks and results of earlier ops in the same or
    an enclosing block) — the SSA in-scope set.
    """
    local: set[int] = set(scope)
    for index, arg in enumerate(block.arguments):
        if not isinstance(arg, Value):
            _fail(
                f"block argument {index} must be a Value, got "
                f"{type(arg).__name__}"
            )
        if arg.owner is not block or arg.index != index:
            _fail(
                f"block argument %{arg.id} has inconsistent owner/index "
                f"({type(arg.owner).__name__}, {arg.index}), expected "
                f"(this block, {index})"
            )
        local.add(arg.id)
    for op in block.ops:
        if not isinstance(op, Op):
            _fail(f"block op must be an Op, got {type(op).__name__}")
        _verify_op(module, op)
        if op.parent is not block:
            _fail(
                f"op '{op.name}' (id {op.id}): parent is not its block",
                op.location,
            )
        for operand in op.operands:
            if not isinstance(operand, Value):
                _fail(
                    f"op '{op.name}' (id {op.id}): operand must be a Value, "
                    f"got {type(operand).__name__}",
                    op.location,
                )
            if operand.id not in local:
                _fail(
                    f"op '{op.name}' (id {op.id}): operand %{operand.id} is "
                    "not defined before this use (SSA dominance)",
                    op.location,
                )
        for region in op.regions:
            _verify_nested_region(module, op, region, local)
        for result in op.results:
            local.add(result.id)


def _verify_nested_region(
    module: Module, op: Op, region: Region, scope: frozenset[int]
) -> None:
    if not isinstance(region, Region):
        _fail(
            f"op '{op.name}' (id {op.id}): nested region must be a Region, "
            f"got {type(region).__name__}",
            op.location,
        )
    if not region.blocks:
        _fail(f"op '{op.name}' (id {op.id}): nested region has no blocks", op.location)
    if region.parent is not op:
        _fail(
            f"op '{op.name}' (id {op.id}): nested region.parent is not the op",
            op.location,
        )
    entry = region.blocks[0]
    if not isinstance(entry, Block):
        _fail(
            f"op '{op.name}' (id {op.id}): region entry must be a Block, got "
            f"{type(entry).__name__}",
            op.location,
        )
    if entry.parent is not region:
        _fail(
            f"op '{op.name}' (id {op.id}): region entry block.parent is not "
            "its region",
            op.location,
        )
    # v1 binding convention: entry-block argument count/type == operand
    # count/type (for `if` ALL operands, including the predicate, bind).
    if len(entry.arguments) != len(op.operands):
        _fail(
            f"op '{op.name}' (id {op.id}): region entry block has "
            f"{len(entry.arguments)} arguments, expected {len(op.operands)} "
            "(one per operand)",
            op.location,
        )
    for i, (arg, operand) in enumerate(zip(entry.arguments, op.operands)):
        if not isinstance(arg, Value):
            _fail(
                f"op '{op.name}' (id {op.id}): region entry argument {i} must "
                f"be a Value, got {type(arg).__name__}",
                op.location,
            )
        if arg.type != operand.type:
            _fail(
                f"op '{op.name}' (id {op.id}): region entry argument {i} type "
                f"{arg.type} does not match operand type {operand.type}",
                op.location,
            )
    # The region's entry args dominate every block of the region; results of
    # ops in sibling blocks do not (conservative: only enclosing scope).
    base = frozenset(scope) | frozenset(arg.id for arg in entry.arguments)
    for block in region.blocks:
        if not isinstance(block, Block):
            _fail(
                f"op '{op.name}' (id {op.id}): region block must be a Block, "
                f"got {type(block).__name__}",
                op.location,
            )
        if block.parent is not region:
            _fail(
                f"op '{op.name}' (id {op.id}): region block.parent is not its "
                "region",
                op.location,
            )
        _verify_block_terminator(block, f"op '{op.name}' region")
        _verify_block_ops(module, block, base)


# ---------------------------------------------------------------------------
# Op level
# ---------------------------------------------------------------------------

def _verify_op(module: Module, op: Op) -> None:
    where = f"op '{op.name}' (id {op.id})"
    loc = op.location
    try:
        opdef_obj = opdef(op.name)
    except KeyError:
        _fail(f"unknown op '{op.name}' (not in the op registry)", loc)
        return
    if not opdef_obj.check_arity(len(op.operands)):
        _fail(
            f"{where}: operand count {len(op.operands)} violates declared "
            f"arity {opdef_obj.arity}",
            loc,
        )
    if len(op.regions) != opdef_obj.regions:
        _fail(
            f"{where}: has {len(op.regions)} regions, declared "
            f"{opdef_obj.regions}",
            loc,
        )
    result_count = opdef_obj.result_count
    if result_count is not None:
        n_results = len(op.results)
        if isinstance(result_count, int):
            if n_results != result_count:
                _fail(
                    f"{where}: result count {n_results} != declared "
                    f"{result_count}",
                    loc,
                )
        else:
            lo, hi = result_count
            if n_results < lo or (hi is not None and n_results > hi):
                _fail(
                    f"{where}: result count {n_results} outside declared "
                    f"range {result_count}",
                    loc,
                )
    _verify_attrs(opdef_obj, op, where, loc)
    for i, operand in enumerate(op.operands):
        if not isinstance(operand, Value):
            _fail(
                f"{where}: operand {i} must be a Value, got "
                f"{type(operand).__name__}",
                loc,
            )
    for i, result in enumerate(op.results):
        if not isinstance(result, Value):
            _fail(
                f"{where}: result {i} must be a Value, got "
                f"{type(result).__name__}",
                loc,
            )
        if result.owner is not op:
            _fail(f"{where}: result %{result.id} is not owned by this op", loc)
        if result.index != i:
            _fail(
                f"{where}: result %{result.id} has index {result.index}, "
                f"expected {i}",
                loc,
            )
    _verify_result_types(module, op, opdef_obj, loc)


def _verify_attrs(opdef_obj: OpDef, op: Op, where: str, loc: Location | None) -> None:
    attributes = op.attributes
    if not isinstance(attributes, dict):
        _fail(f"{where}: attributes must be a dict, got {type(attributes).__name__}", loc)
    schema = {spec.name: spec for spec in opdef_obj.attributes}
    for key in attributes:
        if key not in schema:
            _fail(f"{where}: unknown attribute '{key}'", loc)
    for spec in opdef_obj.attributes:
        if spec.required and spec.name not in attributes:
            _fail(f"{where}: missing required attribute '{spec.name}'", loc)
    for key, value in attributes.items():
        spec = schema[key]
        if not _attr_matches(spec, value):
            _fail(
                f"{where}: attribute '{key}' must be {spec.type}, got "
                f"{_value_desc(value)}",
                loc,
            )


def _attr_matches(spec: AttrSpec, value: Any) -> bool:
    """True if ``value`` matches ``spec`` (tag + nullable-default rule).

    Tag interpretation (binding; kept in sync with the Builder's
    ``_check_attr_value`` — the two must agree tag-for-tag):
    - ``bool``: Python bool.
    - ``int``: int or None (bool excluded); ``None`` accepted ONLY when
      ``spec.default is None`` (the documented nullable attrs, e.g.
      ``argmax.axis``).
    - ``float``: int or float (bool excluded).
    - ``str``: str.
    - ``dtype``: anything ``etl.core.dtype`` accepts (numpy dtype object,
      name string, scalar type, object exposing ``.dtype``).
    - ``ints``: tuple/list of ints (bool excluded); ``None`` accepted ONLY
      when ``spec.default is None`` (e.g. ``transpose.permutation``).
    - ``floats``/``strs``: tuple/list of the base kind.
    - ``nested_ints``: tuple/list of NON-EMPTY tuple/list of ints.
    - ``shape``: tuple/list of int | Dim | DimExpr | None.
    - ``ndarray``: numpy array.
    - ``any``: anything.
    """
    tag = spec.type
    if tag == ATTR_ANY:
        return True
    if tag == ATTR_BOOL:
        return isinstance(value, bool)
    if tag == ATTR_INT:
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            value is None and spec.default is None
        )
    if tag == ATTR_FLOAT:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if tag == ATTR_STR:
        return isinstance(value, str)
    if tag == ATTR_DTYPE:
        try:
            dtype(value)
        except DTypeError:
            return False
        return True
    if tag == ATTR_INTS:
        return (value is None and spec.default is None) or (
            isinstance(value, (tuple, list))
            and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
        )
    if tag == ATTR_FLOATS:
        return isinstance(value, (tuple, list)) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
        )
    if tag == ATTR_STRS:
        return isinstance(value, (tuple, list)) and all(
            isinstance(v, str) for v in value
        )
    if tag == ATTR_NESTED_INTS:
        return isinstance(value, (tuple, list)) and all(
            isinstance(v, (tuple, list))
            and len(v) > 0
            and all(isinstance(x, int) and not isinstance(x, bool) for x in v)
            for v in value
        )
    if tag == ATTR_SHAPE:
        return isinstance(value, (tuple, list)) and all(
            v is None
            or (isinstance(v, int) and not isinstance(v, bool))
            or isinstance(v, (Dim, DimExpr))
            for v in value
        )
    if tag == ATTR_NDARRAY:
        return isinstance(value, np.ndarray)
    return False


# ---------------------------------------------------------------------------
# Result-type agreement
# ---------------------------------------------------------------------------

def _verify_result_types(
    module: Module, op: Op, opdef_obj: OpDef, loc: Location | None
) -> None:
    where = f"op '{op.name}' (id {op.id})"
    if opdef_obj.shape_fn is not None:
        input_types = tuple(operand.type for operand in op.operands)
        try:
            expected = opdef_obj.shape_fn(input_types, dict(op.attributes))
        except (ShapeError, ValueError, TypeError) as error:
            _fail(f"{where}: result-type inference failed: {error}", loc)
            return
        if len(expected) != len(op.results):
            _fail(
                f"{where}: inference yields {len(expected)} results, recorded "
                f"{len(op.results)}",
                loc,
            )
        for i, (expected_type, result) in enumerate(zip(expected, op.results)):
            _check_type_agreement(op, i, expected_type, result.type, loc)
    else:
        _verify_op_specific_results(module, op, loc)


def _check_type_agreement(
    op: Op, index: int, expected: ValueType, actual: Any, loc: Location | None
) -> None:
    where = f"op '{op.name}' (id {op.id})"
    if not isinstance(actual, ValueType):
        _fail(
            f"{where}: result {index} type must be a ValueType, got "
            f"{type(actual).__name__}",
            loc,
        )
    if expected.dtype != actual.dtype:
        _fail(
            f"{where}: result {index} dtype mismatch: inferred {expected.dtype}, "
            f"recorded {actual.dtype}",
            loc,
        )
    if expected.shape != actual.shape:
        _fail(
            f"{where}: result {index} shape mismatch: inferred "
            f"{expected.shape!r}, recorded {actual.shape!r}",
            loc,
        )


def _region_return(block: Block) -> Op | None:
    """The block's last op if it is a ``return`` terminator, else None."""
    if not block.ops:
        return None
    last = block.ops[-1]
    if not isinstance(last, Op):
        return None
    try:
        opdef_obj = opdef(last.name)
    except KeyError:
        return None
    if not opdef_obj.is_terminator or last.name != "return":
        return None
    return last


def _verify_op_specific_results(module: Module, op: Op, loc: Location | None) -> None:
    """Result-type agreement for ``shape_fn=None`` ops (op-specific rules)."""
    where = f"op '{op.name}' (id {op.id})"
    name = op.name
    if name == "constant":
        payload = op.attributes["value"]
        expected = ValueType(np.dtype(payload.dtype), tuple(payload.shape))
        _check_type_agreement(op, 0, expected, op.results[0].type, loc)
    elif name == "if":
        result_types = tuple(value.type for value in op.results)
        for r_index, region in enumerate(op.regions):
            if not region.blocks:
                _fail(f"{where}: region {r_index} has no blocks", loc)
            terminator = _region_return(region.blocks[0])
            if terminator is None:
                _fail(f"{where}: region {r_index} has no 'return' terminator", loc)
            for i, operand in enumerate(terminator.operands):
                if not isinstance(operand, Value):
                    _fail(
                        f"{where}: region {r_index} return operand {i} must be "
                        f"a Value, got {type(operand).__name__}",
                        loc,
                    )
            branch_types = tuple(value.type for value in terminator.operands)
            if branch_types != result_types:
                _fail(
                    f"{where}: region {r_index} returns {branch_types}, op "
                    f"results are {result_types}",
                    loc,
                )
    elif name == "call":
        callee_name = op.attributes["callee"]
        try:
            callee = module.get_function(callee_name)
        except KeyError:
            _fail(
                f"{where}: no function named '{callee_name}' in module "
                f"'{module.name}'",
                loc,
            )
        try:
            expected = callee.output_types
        except (ValueError, AttributeError) as error:
            _fail(
                f"{where}: callee '{callee_name}' has no valid output "
                f"signature: {error}",
                loc,
            )
        result_types = tuple(value.type for value in op.results)
        if expected != result_types:
            _fail(
                f"{where}: callee '{callee_name}' returns {expected}, op "
                f"results are {result_types}",
                loc,
            )
    elif name in ("runtime_call", "block_call", "external_call"):
        specs = op.attributes["result_specs"]
        if not isinstance(specs, (tuple, list)) or not all(
            isinstance(spec, ValueType) for spec in specs
        ):
            _fail(
                f"{where}: attribute 'result_specs' must be a sequence of "
                f"ValueTypes, got {_value_desc(specs)}",
                loc,
            )
        result_types = tuple(value.type for value in op.results)
        if tuple(specs) != result_types:
            _fail(
                f"{where}: declared result_specs {tuple(specs)} do not match "
                f"op results {result_types}",
                loc,
            )
    elif name == "return":
        pass  # zero results — enforced by the declared result_count == 0
    else:
        _fail(
            f"{where}: has no shape-inference hook and no op-specific "
            "result-type rule",
            loc,
        )


# ---------------------------------------------------------------------------
# Global passes: id uniqueness + use bookkeeping
# ---------------------------------------------------------------------------

def _collect_ids(module: Module) -> tuple[dict[int, Value], dict[int, Op]]:
    """Collect every value and op of the module, checking id uniqueness."""
    values: dict[int, Value] = {}
    ops: dict[int, Op] = {}

    def collect_region(region: Region) -> None:
        for block in region.blocks:
            for arg in block.arguments:
                if not isinstance(arg, Value):
                    _fail(
                        f"block argument must be a Value, got "
                        f"{type(arg).__name__}"
                    )
                if arg.id in values:
                    _fail(f"duplicate value id {arg.id} in module '{module.name}'")
                values[arg.id] = arg
            for op in block.ops:
                if op.id in ops:
                    _fail(f"duplicate op id {op.id} in module '{module.name}'")
                ops[op.id] = op
                for result in op.results:
                    if not isinstance(result, Value):
                        _fail(
                            f"op '{op.name}' result must be a Value, got "
                            f"{type(result).__name__}"
                        )
                    if result.id in values:
                        _fail(
                            f"duplicate value id {result.id} in module "
                            f"'{module.name}'"
                        )
                    values[result.id] = result
                for nested in op.regions:
                    if not isinstance(nested, Region):
                        _fail(
                            f"op '{op.name}' nested region must be a Region, "
                            f"got {type(nested).__name__}"
                        )
                    collect_region(nested)

    for function in module.functions:
        collect_region(function.region)
    return values, ops


def _verify_uses(values: dict[int, Value], ops: dict[int, Op]) -> None:
    # Forward: every Use recorded on a value refers back to that value at the
    # recorded operand index, targets an op of this module, and appears once.
    for vid, value in values.items():
        seen: set[tuple[int, int]] = set()
        for use in value.uses:
            if not isinstance(use, Use):
                _fail(
                    f"value %{vid}: use record must be a Use, got "
                    f"{type(use).__name__}"
                )
            owner = use.owner
            if not isinstance(owner, Op):
                _fail(f"value %{vid}: use record has a non-Op owner")
            index = use.operand_index
            if not isinstance(index, int):
                _fail(
                    f"value %{vid}: use record operand_index must be an int, "
                    f"got {type(index).__name__}"
                )
            if not (0 <= index < len(owner.operands)) or owner.operands[index] is not value:
                _fail(
                    f"value %{vid}: use record does not refer to this value at "
                    "the recorded operand index"
                )
            if owner.id not in ops or ops[owner.id] is not owner:
                _fail(
                    f"value %{vid}: use record refers to an op outside the "
                    f"module '{owner.name}'"
                )
            key = (owner.id, index)
            if key in seen:
                _fail(
                    f"value %{vid}: duplicate use record for op '{owner.name}' "
                    f"operand {index}"
                )
            seen.add(key)
    # Reverse: every operand of every op has a matching recorded Use.
    for op in ops.values():
        for i, operand in enumerate(op.operands):
            if not any(
                use.owner is op and use.operand_index == i for use in operand.uses
            ):
                _fail(
                    f"op '{op.name}' (id {op.id}): operand {i} (%{operand.id}) "
                    "has no matching Use record"
                )
