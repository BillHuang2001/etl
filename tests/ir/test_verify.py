"""``etl.ir.verify(module)`` violation-class tests.

``verify`` is the EvoXIR contract checker: it validates module / function /
region / block / op / value / SSA invariants and raises ``VerificationError``
(owned by ``etl.core``) on the FIRST violation, with a source-location
annotation (``[at file:line:col]``) whenever the offending op carries one.

This file exercises every violation class documented in ``verify``'s
docstring — one test per class, parametrized where the class spans several
triggers.  The dominant pattern: build a VALID module, mutate ONE thing,
assert the specific message, and (where a restore makes sense) restore and
assert the module verifies again.  Modules that the Builder's eager checks
would refuse (bad attributes, wrong result types, wrong parent wiring) are
built valid and then mutated in place.

The ``./etl/`` package is read-only from here; if a test below exposes a real
bug in etl it stays as a failing test with a ``# BUG(etl)`` marker.
"""

import numpy as np
import pytest

from etl import ir
from etl.core import VerificationError

F32_2X3 = ir.ValueType(np.dtype("float32"), (2, 3))
F64_2X3 = ir.ValueType(np.dtype("float64"), (2, 3))
BOOL_SCALAR = ir.ValueType(np.dtype("bool"), ())


# ---------------------------------------------------------------------------
# Test-only opdef registrations — see the ``test_only_opdefs`` fixture below:
# tests that need ``_test_halt`` / ``_test_no_rule`` must request it, because
# the opdef registry is process-global (registering at import time would leak
# into every other test module in the same pytest run).
# ---------------------------------------------------------------------------


def _make_test_only_opdefs() -> tuple[ir.OpDef, ir.OpDef]:
    """Build the two test-only OpDefs used by the violation tests.

    - ``_test_halt``: a registered NON-return terminator.  `verify` requires
      the last op of every block to be the ``return`` op specifically, so
      hitting the "terminator must be the 'return' op" branch needs a second
      terminator in the registry (v1 ships only ``return``).
    - ``_test_no_rule``: a registered op with ``shape_fn=None`` and no
      op-specific result rule — `verify` rejects such ops explicitly.
    """
    halt = ir.OpDef(
        name="_test_halt",
        category="terminator",
        description="test-only terminator for verify() violation tests",
        arity=(0, None),
        result_count=0,
        effect=ir.EFFECT_PURE,
        is_terminator=True,
    )
    no_rule = ir.OpDef(
        name="_test_no_rule",
        category="control",
        description="test-only op with no shape_fn and no result rule",
        arity=0,
        result_count=1,
        effect=ir.EFFECT_PURE,
    )
    return halt, no_rule


@pytest.fixture
def test_only_opdefs():
    """Register the test-only OpDefs; restore the registry to its prior state.

    Restoring the registry exactly (including any OpDef that was already
    registered under those names before this fixture ran) keeps the global
    registry clean for sibling test modules such as ``test_registry.py``.
    """
    from etl.ir import op_defs as op_defs_mod

    opdefs = _make_test_only_opdefs()
    prior = {
        opdef.name: op_defs_mod._REGISTRY.pop(opdef.name, None) for opdef in opdefs
    }
    try:
        for opdef in opdefs:
            op_defs_mod.register_opdef(opdef)
        yield
    finally:
        for name, previous in prior.items():
            op_defs_mod._REGISTRY.pop(name, None)
            if previous is not None:
                op_defs_mod._REGISTRY[name] = previous


# ---------------------------------------------------------------------------
# Module builders — every helper returns a FRESH, VALID module.
# ---------------------------------------------------------------------------


def _new_module(name: str = "m") -> tuple[ir.Module, ir.Builder]:
    m = ir.Module(name=name)
    return m, ir.Builder(m)


def _valid_module() -> ir.Module:
    """``main(x: f32[2,3], y: f32[2,3]) { %2 = add(%0, %1); return %2 }``."""
    m, b = _new_module()
    f = b.build_function("main", (F32_2X3, F32_2X3))
    x, y = f.entry_block.arguments
    s = b.emit("add", (x, y))
    b.set_terminator(b.current_block, "return", (s,))
    return m


def _valid_if_module() -> ir.Module:
    """A module whose ``main`` holds an ``if``; both branches return f32[2,3]."""
    return _if_module_with_region_types((BOOL_SCALAR, F32_2X3, F32_2X3))


def _if_module_with_region_types(
    region_types: tuple[ir.ValueType, ...],
) -> ir.Module:
    """``main(pred, x, y)`` with an ``if`` whose regions have ``region_types``
    entry args; each branch returns its second entry argument (f32[2,3])."""
    m, b = _new_module()
    f = b.build_function("main", (BOOL_SCALAR, F32_2X3, F32_2X3))
    pred, x, y = f.entry_block.arguments
    regions = []
    for _ in range(2):
        region = b.build_region(region_types)
        b.push_region(region)
        b.set_terminator(b.current_block, "return", (region.entry.arguments[1],))
        b.pop_region()
        regions.append(region)
    if_op = b.create("if", operands=(pred, x, y), regions=tuple(regions))
    b.set_terminator(b.current_block, "return", (if_op.results[0],))
    return m


def _valid_control_module() -> ir.Module:
    """``main`` with ``constant`` + ``runtime_call`` + ``block_call`` whose
    ``result_specs`` are proper ``ValueType`` sequences."""
    m, b = _new_module()
    f = b.build_function("main", (F32_2X3, F32_2X3))
    x, y = f.entry_block.arguments
    b.emit("constant", attributes={"value": np.array([1.0, 2.0], dtype=np.float32)})
    r = b.create(
        "runtime_call",
        operands=(x,),
        attributes={"callback": "cb", "result_specs": (F32_2X3,)},
    )
    bl = b.create(
        "block_call",
        operands=(y,),
        attributes={"block_name": "my_block", "result_specs": (F32_2X3,)},
    )
    s = b.emit("add", (r.results[0], bl.results[0]))
    b.set_terminator(b.current_block, "return", (s,))
    return m


def _empty_function_module() -> ir.Module:
    """A function whose block has no ops at all (no terminator)."""
    m, b = _new_module()
    b.build_function("main", ())
    return m


def _terminator_not_last_module() -> ir.Module:
    """Valid module with a pure op appended AFTER the return op."""
    m, b = _new_module()
    f = b.build_function("main", (F32_2X3, F32_2X3))
    x, y = f.entry_block.arguments
    s = b.emit("add", (x, y))
    b.set_terminator(b.current_block, "return", (s,))
    extra = b.create("negate", (x,))  # insertion point sits right after `add`
    f.entry_block.append(extra)  # now AFTER the return op
    return m


def _reduce_module() -> ir.Module:
    m, b = _new_module()
    f = b.build_function("main", (F32_2X3,))
    (x,) = f.entry_block.arguments
    r = b.emit("reduce_sum", (x,), attributes={"axes": (0,), "reduce_op": "sum"})
    b.set_terminator(b.current_block, "return", (r,))
    return m


def _pad_module() -> ir.Module:
    m, b = _new_module()
    f = b.build_function("main", (F32_2X3,))
    (x,) = f.entry_block.arguments
    p = b.emit("pad", (x,), attributes={"padding_config": ((1, 1), (0, 0))})
    b.set_terminator(b.current_block, "return", (p,))
    return m


def _module_with_extra_op() -> ir.Module:
    """``main`` with ``add`` then an unused ``negate`` (for result stealing)."""
    m, b = _new_module()
    f = b.build_function("main", (F32_2X3, F32_2X3))
    x, y = f.entry_block.arguments
    s = b.emit("add", (x, y))
    b.emit("negate", (x,))
    b.set_terminator(b.current_block, "return", (s,))
    return m


def _explicit_result_module(result_type: ir.ValueType) -> ir.Module:
    """``add`` with explicit ``result_type`` (Builder accepts; verify checks)."""
    m, b = _new_module()
    f = b.build_function("main", (F32_2X3, F32_2X3))
    x, y = f.entry_block.arguments
    s = b.emit("add", (x, y), result_type=result_type)
    b.set_terminator(b.current_block, "return", (s,))
    return m


def _chain_module() -> ir.Module:
    """``negate(x)`` then ``add(negate_result, x)`` (operand produced earlier)."""
    m, b = _new_module()
    f = b.build_function("main", (F32_2X3,))
    (x,) = f.entry_block.arguments
    neg = b.emit("negate", (x,))
    s = b.emit("add", (neg, x))
    b.set_terminator(b.current_block, "return", (s,))
    return m


def _call_unknown_module() -> ir.Module:
    m, b = _new_module()
    f = b.build_function("main", ())
    call_op = b.create(
        "call", attributes={"callee": "ghost"}, result_types=(F32_2X3,)
    )
    b.set_terminator(b.current_block, "return", (call_op.results[0],))
    return m


def _call_mismatch_module() -> ir.Module:
    m, b = _new_module()
    callee = b.build_function("callee", (F32_2X3,))
    (z,) = callee.entry_block.arguments
    b.set_terminator(b.current_block, "return", (z,))
    b.build_function("main", ())
    call_op = b.create(
        "call", attributes={"callee": "callee"}, result_types=(F64_2X3,)
    )
    b.set_terminator(b.current_block, "return", (call_op.results[0],))
    return m


def _specs_module(op_name: str) -> ir.Module:
    """``runtime_call``/``block_call`` with DICT ``result_specs`` — the Builder
    accepts them (converting for result-type resolution only) but stores the
    attribute unrewritten, which verify rejects."""
    m, b = _new_module()
    f = b.build_function("main", ())
    extra = (
        {"callback": "cb"}
        if op_name == "runtime_call"
        else {"block_name": "blk"}
    )
    op = b.create(
        op_name,
        attributes={"result_specs": [{"dtype": "float32", "shape": (2, 3)}], **extra},
    )
    b.set_terminator(b.current_block, "return", (op.results[0],))
    return m


def _if_with_empty_region_module() -> ir.Module:
    m, b = _new_module()
    f = b.build_function("main", (BOOL_SCALAR, F32_2X3, F32_2X3))
    pred, x, y = f.entry_block.arguments
    good = b.build_region((BOOL_SCALAR, F32_2X3, F32_2X3))
    b.push_region(good)
    b.set_terminator(b.current_block, "return", (good.entry.arguments[1],))
    b.pop_region()
    if_op = b.create(
        "if",
        operands=(pred, x, y),
        regions=(ir.Region(), good),  # first region has no blocks
        result_types=(F32_2X3,),
    )
    b.set_terminator(b.current_block, "return", (if_op.results[0],))
    return m


def _no_rule_module() -> ir.Module:
    m, b = _new_module()
    f = b.build_function("main", ())
    op = b.create("_test_no_rule", result_types=(F32_2X3,))
    b.set_terminator(b.current_block, "return", (op.results[0],))
    return m


# ---------------------------------------------------------------------------
# Mutators (each changes ONE thing on an otherwise-valid module).
# ---------------------------------------------------------------------------


def _swap_first_two_ops(m: ir.Module) -> None:
    ops = m.main.entry_block.ops
    ops[0], ops[1] = ops[1], ops[0]


def _give_return_a_result(m: ir.Module) -> None:
    ret = m.main.entry_block.ops[-1]
    ret.results = (ir.Value(id=m.new_value_id(), type=F32_2X3, owner=ret, index=0),)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _assert_violation(build, mutate, pattern: str) -> None:
    """Build a fresh module, mutate ONE thing, expect the specific failure."""
    module = build()
    mutate(module)
    with pytest.raises(ir.VerificationError, match=pattern):
        ir.verify(module)


# ---------------------------------------------------------------------------
# 1. Valid modules
# ---------------------------------------------------------------------------


def test_verification_error_is_owned_by_core():
    assert ir.VerificationError is VerificationError


def test_valid_modules_pass():
    assert ir.verify(_valid_module()) is None
    assert ir.verify(_valid_if_module()) is None
    assert ir.verify(_valid_control_module()) is None
    # Explicit result types that AGREE with shape inference are fine.
    assert ir.verify(_explicit_result_module(F32_2X3)) is None


# ---------------------------------------------------------------------------
# 2. Module level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "mutate", "pattern"),
    [
        pytest.param(
            _valid_module,
            lambda m: setattr(m, "version", ir.IR_FORMAT_VERSION + 1),
            "does not match the IR format version",
            id="version-mismatch",
        ),
        pytest.param(
            lambda: ir.Module(name="m"),
            lambda m: None,
            "at least one function",
            id="no-functions",
        ),
        pytest.param(
            _valid_module,
            lambda m: m.functions.append(m.main),
            "duplicate function name",
            id="duplicate-function-name",
        ),
        pytest.param(
            _valid_module,
            lambda m: setattr(m.main, "parent", None),
            "parent is not module",
            id="function-parent-not-module",
        ),
        pytest.param(
            _valid_module,
            lambda m: m.functions.append("junk"),
            "must be Function",
            id="non-function-entry",
        ),
    ],
)
def test_module_level_violations(build, mutate, pattern):
    _assert_violation(build, mutate, pattern)


# ---------------------------------------------------------------------------
# 3. Function level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "mutate", "pattern"),
    [
        pytest.param(
            _valid_module,
            lambda m: m.main.region.append_block(ir.Block()),
            "v1 requires exactly one",
            id="two-blocks",
        ),
        pytest.param(
            _valid_module,
            lambda m: setattr(m.main.region, "parent", None),
            "region.parent is not the function",
            id="region-parent",
        ),
        pytest.param(
            _valid_module,
            lambda m: setattr(m.main.entry_block, "parent", None),
            "block.parent is not the function region",
            id="block-parent",
        ),
        pytest.param(
            _valid_module,
            lambda m: setattr(
                m.main.entry_block, "arguments", m.main.entry_block.arguments[:-1]
            ),
            "arguments, expected",
            id="entry-arg-count",
        ),
        pytest.param(
            _valid_module,
            lambda m: setattr(m.main.entry_block.arguments[0], "type", F64_2X3),
            "does not match input type",
            id="entry-arg-type",
        ),
        pytest.param(
            _empty_function_module,
            lambda m: None,
            "missing 'return' terminator",
            id="missing-terminator-empty-block",
        ),
        pytest.param(
            _valid_module,
            lambda m: setattr(m.main.entry_block.ops[-1], "name", "while"),
            "no 'return' terminator",
            id="last-op-not-return",
        ),
        pytest.param(
            _terminator_not_last_module,
            lambda m: None,
            "not the last op",
            id="terminator-not-last",
        ),
        pytest.param(
            _valid_module,
            lambda m: setattr(m.main.entry_block.ops[-1], "name", "_test_halt"),
            "must be the 'return' op",
            id="non-return-terminator",
        ),
        pytest.param(
            _valid_module,
            lambda m: setattr(
                m.main.entry_block.ops[-1],
                "operands",
                (m.main.entry_block.arguments[0], "junk"),
            ),
            "must be a Value",
            id="return-operand-not-value",
        ),
    ],
)
def test_function_level_violations(build, mutate, pattern, test_only_opdefs):
    _assert_violation(build, mutate, pattern)


# ---------------------------------------------------------------------------
# 4. Op level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "mutate", "pattern"),
    [
        pytest.param(
            _valid_module,
            lambda m: setattr(m.main.entry_block.ops[0], "name", "addd"),
            "unknown op",
            id="unknown-op-name",
        ),
        pytest.param(
            _valid_module,
            lambda m: setattr(
                m.main.entry_block.ops[0],
                "operands",
                (m.main.entry_block.arguments[0],),
            ),
            "violates declared arity",
            id="arity-violation",
        ),
        pytest.param(
            _valid_module,
            lambda m: setattr(m.main.entry_block.ops[0], "regions", (ir.Region(),)),
            "regions, declared",
            id="region-count-non-region-op",
        ),
        pytest.param(
            _valid_if_module,
            lambda m: setattr(
                m.main.entry_block.ops[0],
                "regions",
                m.main.entry_block.ops[0].regions[:1],
            ),
            "regions, declared",
            id="region-count-if-one-region",
        ),
        pytest.param(
            _valid_module,
            lambda m: setattr(m.main.entry_block.ops[0], "results", ()),
            "result count",
            id="result-count-mismatch",
        ),
        pytest.param(
            _valid_module,
            _give_return_a_result,
            "result count",
            id="return-result-count",
        ),
        pytest.param(
            _valid_module,
            lambda m: setattr(m.main.entry_block.ops[0], "attributes", "junk"),
            "attributes must be a dict",
            id="attributes-not-dict",
        ),
        pytest.param(
            _valid_module,
            lambda m: setattr(m.main.entry_block.ops[0], "attributes", {"axes": (0,)}),
            "unknown attribute",
            id="unknown-attribute",
        ),
        pytest.param(
            _reduce_module,
            lambda m: m.main.entry_block.ops[0].attributes.pop("axes"),
            "missing required attribute",
            id="missing-required-attribute",
        ),
        pytest.param(
            _pad_module,
            lambda m: m.main.entry_block.ops[0].attributes.__setitem__(
                "padding_config", (1, 1)
            ),
            "nested_ints",
            id="wrong-typed-attribute-nested-ints",
        ),
        pytest.param(
            _module_with_extra_op,
            lambda m: setattr(
                m.main.entry_block.ops[0],
                "results",
                (m.main.entry_block.ops[1].results[0],),
            ),
            "not owned by this op",
            id="result-not-owned",
        ),
        pytest.param(
            _valid_module,
            lambda m: setattr(m.main.entry_block.ops[0].results[0], "index", 5),
            "has index",
            id="result-index-wrong",
        ),
        pytest.param(
            _valid_module,
            lambda m: setattr(m.main.entry_block.ops[0].results[0], "type", "float32"),
            "must be a ValueType",
            id="result-type-not-valuetype",
        ),
    ],
)
def test_op_level_violations(build, mutate, pattern):
    _assert_violation(build, mutate, pattern)


# ---------------------------------------------------------------------------
# 5. shape_fn result-type agreement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result_type", "pattern"),
    [
        pytest.param(F64_2X3, "dtype mismatch", id="dtype-mismatch"),
        pytest.param(
            ir.ValueType(np.dtype("float32"), (3, 3)), "shape mismatch", id="shape-mismatch"
        ),
    ],
)
def test_shape_fn_agreement_violations(result_type, pattern):
    _assert_violation(lambda: _explicit_result_module(result_type), lambda m: None, pattern)


# ---------------------------------------------------------------------------
# 6. Op-specific result rules (shape_fn=None)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "mutate", "pattern"),
    [
        pytest.param(
            _valid_if_module,
            lambda m: setattr(m.main.entry_block.ops[0].results[0], "type", F64_2X3),
            "region 0 returns",
            id="if-results-mismatch",
        ),
        pytest.param(
            _call_unknown_module,
            lambda m: None,
            "no function named",
            id="call-unknown-callee",
        ),
        pytest.param(
            _call_mismatch_module,
            lambda m: None,
            "callee",
            id="call-result-mismatch",
        ),
        pytest.param(
            lambda: _specs_module("runtime_call"),
            lambda m: None,
            "must be a sequence of ValueTypes",
            id="runtime-call-dict-specs",
        ),
        pytest.param(
            lambda: _specs_module("block_call"),
            lambda m: None,
            "must be a sequence of ValueTypes",
            id="block-call-dict-specs",
        ),
        pytest.param(
            _no_rule_module,
            lambda m: None,
            "no shape-inference hook",
            id="no-result-rule",
        ),
    ],
)
def test_op_specific_result_rule_violations(build, mutate, pattern, test_only_opdefs):
    _assert_violation(build, mutate, pattern)


# ---------------------------------------------------------------------------
# 7. SSA / value level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "mutate", "pattern"),
    [
        pytest.param(
            _valid_module,
            lambda m: setattr(
                m.main.entry_block.ops[0].results[0],
                "id",
                m.main.entry_block.arguments[1].id,
            ),
            "duplicate value id",
            id="duplicate-value-id",
        ),
        pytest.param(
            _chain_module,
            _swap_first_two_ops,
            "not defined before this use",
            id="use-before-def",
        ),
        pytest.param(
            _valid_module,
            lambda m: setattr(
                m.main.entry_block.ops[0],
                "operands",
                (m.main.entry_block.arguments[0], "junk"),
            ),
            "operand 1 must be a Value",
            id="operand-not-value",
        ),
        pytest.param(
            _valid_module,
            lambda m: m.main.entry_block.arguments[0].uses.clear(),
            "no matching Use record",
            id="missing-use-record",
        ),
        pytest.param(
            _valid_module,
            lambda m: m.main.entry_block.arguments[0].uses.append(
                m.main.entry_block.arguments[0].uses[0]
            ),
            "duplicate use record",
            id="duplicate-use-record",
        ),
        pytest.param(
            _valid_module,
            lambda m: m.main.entry_block.arguments[0].uses.append(
                ir.Use(m.main.entry_block.ops[0], 1)
            ),
            "does not refer to this value",
            id="mismatched-use-record",
        ),
    ],
)
def test_ssa_value_level_violations(build, mutate, pattern):
    _assert_violation(build, mutate, pattern)


# ---------------------------------------------------------------------------
# 8. Nested regions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "mutate", "pattern"),
    [
        pytest.param(
            lambda: _if_module_with_region_types((BOOL_SCALAR, F32_2X3)),
            lambda m: None,
            "one per operand",
            id="entry-arg-count",
        ),
        pytest.param(
            lambda: _if_module_with_region_types((BOOL_SCALAR, F32_2X3, F64_2X3)),
            lambda m: None,
            "does not match operand type",
            id="entry-arg-type",
        ),
        pytest.param(
            _if_with_empty_region_module,
            lambda m: None,
            "no blocks",
            id="nested-region-no-blocks",
        ),
        pytest.param(
            _valid_if_module,
            lambda m: setattr(
                m.main.entry_block.ops[0].regions[0], "parent", None
            ),
            "nested region.parent is not the op",
            id="nested-region-parent",
        ),
    ],
)
def test_nested_region_violations(build, mutate, pattern):
    _assert_violation(build, mutate, pattern)


# ---------------------------------------------------------------------------
# 9. Source-location annotation
# ---------------------------------------------------------------------------


def test_location_annotation_in_violation_message():
    m, b = _new_module()
    f = b.build_function("main", (F32_2X3, F32_2X3))
    x, y = f.entry_block.arguments
    s = b.emit("add", (x, y), location=ir.Location("model.py", 83, 1))
    b.set_terminator(b.current_block, "return", (s,))
    add_op = f.entry_block.ops[0]
    add_op.parent = None  # ONE mutation: parent wiring broken
    with pytest.raises(ir.VerificationError, match=r"model\.py:83"):
        ir.verify(m)
    add_op.parent = f.entry_block  # restore: the module verifies again
    assert ir.verify(m) is None


# ---------------------------------------------------------------------------
# 10. Type contract
# ---------------------------------------------------------------------------


def test_verify_rejects_non_module():
    with pytest.raises(TypeError, match="expects an ir.Module"):
        ir.verify("not a module")
