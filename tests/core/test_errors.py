"""Tests for the etl error hierarchy (``etl.core.errors``).

Contract (see ``etl/core/CONTEXT.md`` and the root error strategy):

- ``ETLError`` is the single base class for all public etl errors, deriving
  directly from ``Exception``.
- The eight subclasses ``TraceError, ShapeError, TransformError, BackendError,
  PersistenceError, DeviceError, DTypeError, VerificationError`` all derive
  from ``ETLError`` (and hence from ``Exception``).
- All nine names are re-exported from ``etl.core`` and the package root ``etl``
  (identical objects, not copies).
- Errors are ordinary exceptions: instantiable with a message, message
  preserved via ``str(exc)``/``exc.args[0]``, catchable as ``ETLError``.
"""

import pytest

import etl
import etl.core
from etl.core import ETLError, errors

SUBCLASS_NAMES = [
    "TraceError",
    "ShapeError",
    "TransformError",
    "BackendError",
    "PersistenceError",
    "DeviceError",
    "DTypeError",
    "VerificationError",
]

ALL_NAMES = ["ETLError", *SUBCLASS_NAMES]


def test_etl_error_base_is_exactly_exception():
    # errors.py declares `class ETLError(Exception)` — assert exactly that.
    assert ETLError.__bases__ == (Exception,)


@pytest.mark.parametrize("name", SUBCLASS_NAMES)
def test_subclass_derives_from_etl_error_and_exception(name):
    cls = getattr(errors, name)
    assert issubclass(cls, ETLError)
    assert issubclass(cls, Exception)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_error_is_instantiable_and_message_preserved(name):
    cls = getattr(errors, name)
    msg = f"{name}: something went wrong"
    exc = cls(msg)
    assert str(exc) == msg
    assert exc.args[0] == msg


@pytest.mark.parametrize("name", ALL_NAMES)
def test_error_is_catchable_as_exception(name):
    cls = getattr(errors, name)
    msg = f"{name}: boom"
    with pytest.raises(Exception, match=rf"{name}: boom"):
        raise cls(msg)


@pytest.mark.parametrize("name", SUBCLASS_NAMES)
def test_subclass_is_catchable_as_etl_error(name):
    cls = getattr(errors, name)
    msg = f"{name}: caught by base"
    with pytest.raises(ETLError, match=rf"{name}: caught by base"):
        raise cls(msg)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_re_exported_from_etl_core(name):
    # Same object as in errors, not a shadow copy.
    assert getattr(etl.core, name) is getattr(errors, name)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_re_exported_from_etl_root(name):
    # Same object as in errors, not a shadow copy.
    assert getattr(etl, name) is getattr(errors, name)
