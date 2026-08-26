"""Tests for `@etl.defn` — the graph-definition marker decorator.

Contract under test (`etl/trace/defn.py` + `etl/trace/CONTEXT.md`):
- `@etl.defn` (bare or with options) returns an `etl.Defn` that stores the
  ORIGINAL function object — it does NOT wrap, so metadata is naturally
  preserved.
- A `Defn` is NEVER callable: direct calls always raise `etl.TraceError`
  directing to `etl.trace(defn, *specs)` / `etl.evaluate(defn, *args)`.
- Idempotent: `etl.defn(existing_defn)` returns the same object.
- `etl.trace` accepts a Defn and produces the same Graph as tracing the
  plain function.

Constants inside traced functions MUST be `etl.constant(etl.tensor(v,
dtype=d))` — never captured concrete Tensors (see spec-compliance tests).
"""

import inspect

import numpy as np
import pytest

import etl


# --- decorator forms --------------------------------------------------------


def test_bare_decorator_returns_defn_marking_both():
    @etl.defn
    def f(x):
        return etl.add(x, x)

    assert isinstance(f, etl.Defn)
    assert f.__etl_defn__
    assert f.fn.__etl_defn__


def test_defn_stores_the_original_function_object():
    def f(x):
        return etl.add(x, x)

    d = etl.defn(f)

    assert d.fn is f


def test_options_form_stores_options_and_bare_defaults_to_empty():
    def f(x):
        return etl.add(x, x)

    bare = etl.defn(f)
    assert bare.options == {}

    @etl.defn(compiler_hint="iree")
    def g(x):
        return etl.add(x, x)

    assert g.options == {"compiler_hint": "iree"}


# --- idempotence ------------------------------------------------------------


def test_defn_of_defn_is_idempotent():
    def f(x):
        return etl.add(x, x)

    d = etl.defn(f)
    # BUG(etl): `etl.defn(existing_defn)` builds a NEW Defn instead of
    # returning the same object, violating the contract "applying `defn`
    # to an existing `Defn` returns it unchanged" (defn.py docstring).
    again = etl.defn(d)

    assert again is d


def test_defn_fn_is_always_the_unwrapped_plain_function():
    def f(x):
        return etl.add(x, x)

    d = etl.defn(f)
    again = etl.defn(d)  # unwraps — never nests Defns

    assert d.fn is f
    assert again.fn is f
    assert not isinstance(d.fn, etl.Defn)
    assert not isinstance(again.fn, etl.Defn)


# --- metadata preservation (defn does NOT wrap) -----------------------------


def test_defn_preserves_function_metadata():
    def f(x):
        """Add one to every element."""
        return etl.add(x, etl.constant(etl.tensor(1.0, dtype=etl.float32)))

    d = etl.defn(f)

    assert d.fn.__name__ == f.__name__
    assert d.fn.__doc__ == f.__doc__
    assert inspect.signature(d.fn) == inspect.signature(f)


def test_defn_repr_contains_function_name():
    @etl.defn
    def add_one(x):
        return etl.add(x, etl.constant(etl.tensor(1.0, dtype=etl.float32)))

    assert "add_one" in repr(add_one)


# --- calling a Defn always raises (no implicit eager) -----------------------

_CALL_CASES = [
    pytest.param((), id="no-args"),
    pytest.param((etl.TensorSpec((3,), etl.float32),), id="tensor-spec"),
    pytest.param(
        (np.array([1.0, 2.0, 3.0], dtype=np.float32),), id="concrete-ndarray"
    ),
]


@pytest.mark.parametrize("args", _CALL_CASES)
@pytest.mark.parametrize("needle", ["etl.trace", "etl.evaluate"])
def test_calling_defn_always_raises_traceerror_directing_to_staging(args, needle):
    @etl.defn
    def f(x):
        return etl.add(x, x)

    with pytest.raises(etl.TraceError, match=needle):
        f(*args)


# --- tracing a Defn equals tracing the plain fn -----------------------------


def test_trace_defn_matches_trace_plain_fn():
    @etl.defn
    def f(x):
        return etl.add(x, etl.constant(etl.tensor(1.0, dtype=etl.float32)))

    spec = etl.TensorSpec((3,), etl.float32)
    graph_from_defn = etl.trace(f, spec)
    graph_from_fn = etl.trace(f.fn, spec)

    assert etl.ir.serialize_module(graph_from_defn.module) == etl.ir.serialize_module(
        graph_from_fn.module
    )
