"""Tests for SymbolicTensor purity and operator dispatch.

Covers the core value-model contract:
- SymbolicTensor is a pure graph value: no ``numpy``/``data_ptr``/
  ``__dlpack__``/``__array__``, no Python truthiness, unhashable.
- Construction validation (dtype normalization, shape entries, ir.Value
  duck-typing, frozen dataclass).
- Operator dunders dispatch through the handler registry
  (``register_operator_handlers``), including reflected dunders.
- Missing handlers raise a clear TraceError naming the kind.
- Integration: real etl.ops handlers build IR ops inside a trace.

NOTE: importing ``etl`` (this module does) auto-imports ``etl.ops``, which
registers all operator handlers and the constant builder. Tests for missing
handlers must therefore temporarily clear the registries via fixtures.
"""

import dataclasses

import numpy as np
import pytest

import etl
from etl.core import symbolic as symbolic_mod
from etl.core.symbolic import _OPERATOR_HANDLERS

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _FakeValue:
    """Duck-typed ir.Value: exposes the required ``id`` attribute."""

    id = "fake_id"


def make_st(dtype="float32", shape=(2, 2), location=None):
    return etl.SymbolicTensor(_FakeValue(), dtype, shape, location=location)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_handlers():
    """Replace ALL operator handlers with recording fakes; restore after.

    Each fake records its call args (per kind) and returns a fresh
    SymbolicTensor, so dispatch mechanics can be tested without a trace.
    """
    saved = dict(_OPERATOR_HANDLERS)
    records = {}

    def make_fake(kind):
        def fake(*args):
            records.setdefault(kind, []).append(args)
            return make_st()

        return fake

    _OPERATOR_HANDLERS.clear()
    for kind in symbolic_mod._OPERATOR_KINDS:
        _OPERATOR_HANDLERS[kind] = make_fake(kind)
    try:
        yield records
    finally:
        _OPERATOR_HANDLERS.clear()
        _OPERATOR_HANDLERS.update(saved)


@pytest.fixture
def cleared_handlers():
    """Temporarily clear the operator-handler registry; restore after."""
    saved = dict(_OPERATOR_HANDLERS)
    _OPERATOR_HANDLERS.clear()
    try:
        yield
    finally:
        _OPERATOR_HANDLERS.clear()
        _OPERATOR_HANDLERS.update(saved)


@pytest.fixture
def cleared_constant_builder():
    """Temporarily clear the constant-op builder hook; restore after."""
    saved = symbolic_mod._CONSTANT_BUILDER
    symbolic_mod._CONSTANT_BUILDER = None
    try:
        yield
    finally:
        symbolic_mod._CONSTANT_BUILDER = saved


# ---------------------------------------------------------------------------
# 1. purity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["numpy", "data_ptr", "__dlpack__", "__array__"])
def test_purity_no_concrete_accessors(name):
    st = make_st()
    # A symbolic tensor must never be confused with concrete data — neither
    # on the instance nor on the class.
    assert not hasattr(st, name)
    assert not hasattr(etl.SymbolicTensor, name)


def test_bool_raises_with_control_flow_hint():
    st = make_st()
    with pytest.raises(etl.TraceError, match="etl.cond") as excinfo:
        bool(st)
    message = str(excinfo.value)
    assert "etl.while_loop" in message
    assert "etl.scan" in message


def test_bool_message_includes_location():
    st = make_st(location="model.py:83")
    with pytest.raises(etl.TraceError) as excinfo:
        bool(st)
    assert "model.py:83" in str(excinfo.value)


def test_unhashable():
    st = make_st()
    with pytest.raises(TypeError, match="unhashable"):
        hash(st)
    assert etl.SymbolicTensor.__hash__ is None


# ---------------------------------------------------------------------------
# 2. construction / validation
# ---------------------------------------------------------------------------


def test_dtype_is_normalized():
    st = etl.SymbolicTensor(_FakeValue(), "float32", (2, 3))
    assert isinstance(st.dtype, np.dtype)
    assert st.dtype == np.dtype("float32")


def test_shape_is_tupleified_and_entries_accepted():
    st = etl.SymbolicTensor(
        _FakeValue(),
        etl.float32,
        [etl.Dim("n"), etl.Dim("k") + etl.Dim("m"), 2, None],
    )
    assert isinstance(st.shape, tuple)
    assert st.shape == (etl.Dim("n"), etl.Dim("k") + etl.Dim("m"), 2, None)


@pytest.mark.parametrize("bad_entry", [1.5, "two", (2, 2)])
def test_invalid_shape_entry_raises(bad_entry):
    with pytest.raises(TypeError, match="shape"):
        etl.SymbolicTensor(_FakeValue(), etl.float32, (bad_entry,))


def test_value_must_expose_id():
    with pytest.raises(TypeError, match="'id'"):
        etl.SymbolicTensor(object(), etl.float32, ())


def test_id_property_and_location_default():
    st = etl.SymbolicTensor(_FakeValue(), etl.float32, ())
    assert st.id == "fake_id"
    assert st.location is None


def test_frozen():
    st = make_st()
    with pytest.raises(dataclasses.FrozenInstanceError):
        st.dtype = etl.float64  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. dispatch mechanics (unit, fake handlers — no trace needed)
# ---------------------------------------------------------------------------

_BINARY_OPS = [
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
    ("matmul", lambda a, b: a @ b),
    ("truediv", lambda a, b: a / b),
    ("pow", lambda a, b: a**b),
    ("lt", lambda a, b: a < b),
    ("gt", lambda a, b: a > b),
    ("le", lambda a, b: a <= b),
    ("ge", lambda a, b: a >= b),
    ("eq", lambda a, b: a == b),
    ("getitem", lambda a, b: a[b]),
]


@pytest.mark.parametrize("kind,op", _BINARY_OPS, ids=[k for k, _ in _BINARY_OPS])
def test_binary_dispatch_routes_to_kind(fake_handlers, kind, op):
    records = fake_handlers
    st1, st2 = make_st(), make_st()
    result = op(st1, st2)
    # Results are fresh SymbolicTensors (compared by identity — `==` would
    # dispatch back through the eq handler).
    assert isinstance(result, etl.SymbolicTensor)
    assert len(records[kind]) == 1
    left, right = records[kind][0]
    assert left is st1
    assert right is st2


def test_getitem_passes_scalar_key(fake_handlers):
    records = fake_handlers
    st = make_st()
    result = st[0]
    assert isinstance(result, etl.SymbolicTensor)
    args = records["getitem"][0]
    assert args[0] is st
    assert args[1] == 0


def test_neg_dispatches_single_operand(fake_handlers):
    records = fake_handlers
    st = make_st()
    result = -st
    assert isinstance(result, etl.SymbolicTensor)
    (operand,) = records["neg"][0]
    assert operand is st


_REFLECTED_OPS = [
    ("add", lambda a: 2 + a),
    ("sub", lambda a: 2 - a),
    ("mul", lambda a: 2 * a),
    ("matmul", lambda a: 2 @ a),
    ("truediv", lambda a: 2 / a),
    ("pow", lambda a: 2**a),
]


@pytest.mark.parametrize(
    "kind,op", _REFLECTED_OPS, ids=[k for k, _ in _REFLECTED_OPS]
)
def test_reflected_dispatch_swaps_operands(fake_handlers, kind, op):
    records = fake_handlers
    st = make_st()
    result = op(st)
    assert isinstance(result, etl.SymbolicTensor)
    left, right = records[kind][0]
    assert left == 2
    assert right is st


def test_register_replaces_existing_handler(fake_handlers):
    records = fake_handlers

    def fake2(left, right):
        records.setdefault("add2", []).append((left, right))
        return make_st()

    etl.core.register_operator_handlers("add", fake2)
    st1, st2 = make_st(), make_st()
    st1 + st2
    # (identity checks only: tuple/list `==` would dispatch through the eq
    # handler, whose result is a SymbolicTensor, not a Python bool)
    assert len(records["add2"]) == 1
    assert records["add2"][0][0] is st1
    assert records["add2"][0][1] is st2


def test_register_non_callable_raises():
    with pytest.raises(TypeError, match="callable"):
        etl.core.register_operator_handlers("add", 42)


# ---------------------------------------------------------------------------
# 4. missing handlers / builders
# ---------------------------------------------------------------------------


def test_missing_handler_raises_naming_kind(cleared_handlers):
    st1, st2 = make_st(), make_st()
    with pytest.raises(etl.TraceError, match="kind 'add'") as excinfo:
        st1 + st2
    assert "etl.ops" in str(excinfo.value)


def test_constant_without_builder_raises(cleared_constant_builder):
    t = etl.tensor([1.0, 2.0])
    with pytest.raises(etl.TraceError, match="etl.ops"):
        etl.core.constant(t)


# ---------------------------------------------------------------------------
# 5. integration: real handlers build ops inside a trace
# ---------------------------------------------------------------------------


def test_operators_build_symbolic_results_inside_trace():
    seen = []

    def fn(a, b):
        seen.append(a + b)
        seen.append(a - b)
        seen.append(a * b)
        seen.append(a / b)
        seen.append(a @ b)
        seen.append(a**2)
        seen.append(-a)
        seen.append(a < b)
        seen.append(a <= b)
        seen.append(a > b)
        seen.append(a >= b)
        seen.append(a == b)
        seen.append(a[0])
        return a + b

    spec = etl.TensorSpec((2, 2), etl.float32)
    graph = etl.trace(fn, spec, spec)

    # 7 arithmetic ops (float32), 5 comparisons (bool), getitem (float32).
    expected_dtypes = [etl.float32] * 7 + [etl.bool_] * 5 + [etl.float32]
    assert len(seen) == len(expected_dtypes)
    for result, expected in zip(seen, expected_dtypes):
        assert isinstance(result, etl.SymbolicTensor)
        assert result.dtype == expected
    # getitem sliced the leading axis: (2, 2) -> (2,)
    assert seen[-1].shape == (2,)
    # every SSA value has a distinct identity
    assert len({r.id for r in seen}) == len(seen)
    assert isinstance(graph, etl.Graph)


# ---------------------------------------------------------------------------
# 6. constant with the registered builder (outside a trace -> TraceError)
# ---------------------------------------------------------------------------


def test_constant_registered_but_no_trace_raises():
    t = etl.tensor([1.0, 2.0])
    with pytest.raises(etl.TraceError):
        etl.core.constant(t)
