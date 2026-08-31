"""External-kernel registry — the RUN-TIME side of the external-kernel mechanism.

Kernels are registered by NAME in this process-global registry and resolved at
run time by the numpy backend when it executes an ``external_call`` op. The
mechanism is kernel-agnostic: the callable contract is plain numpy arrays in,
plain numpy arrays (or a ``core.Tensor`` / tuple of them) out — a triton
kernel written against this interface (in a DIFFERENT repo) will plug in via
the same ``register_external_kernel`` call. Compiler-backend host-dispatch
(round 2) consumes the same registry.

Binding rules (see ``etl/CONTEXT.md``, "External kernels"):

- Callable signature: ``fn(*np_arrays) -> ndarray | Tensor | tuple/list of
  them``. The number/dtype/shape of the returned arrays must match the specs
  declared by the ``external_call`` op that invoked the kernel — the numpy
  backend validates count/dtype/shape at run time (``BackendError`` /
  ``ShapeError`` — never a silent coercion).
- Registration is process-global and REPLACES any previous registration under
  the same name (last registration wins — documented overwrite semantics;
  this makes hot-reloading kernels and adapter re-registration safe).
- The registry is NOT serialized: graph artifacts carry only the kernel name;
  any process that runs a graph must re-register its kernels first
  (``BackendError`` naming the kernel otherwise).
- Errors are loud: registering a non-callable / bad name raises ``TypeError``;
  unregistering an unknown name raises ``KeyError``; resolving an unknown name
  returns ``None`` (the numpy backend turns that into ``BackendError``).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

__all__ = [
    "register_external_kernel",
    "unregister_external_kernel",
    "get_external_kernel",
]

#: name -> callable. Process-global; the numpy backend (and, in round 2, the
#: compiler-adapter host-dispatch) resolve through :func:`get_external_kernel`.
_REGISTRY: Dict[str, Callable] = {}


def register_external_kernel(name: str, callable_) -> None:
    """Register ``callable_`` as the external kernel named ``name``.

    Overwrites any previous registration under ``name`` (last wins —
    documented; there is no error on re-registration).

    Args:
        name: Non-empty kernel name (the same string passed to
            ``etl.external_call`` in traced graphs).
        callable_: ``fn(*np_arrays) -> ndarray | Tensor | tuple/list of them``.

    Raises:
        TypeError: ``name`` is not a non-empty str, or ``callable_`` is not
            callable.
    """
    if not isinstance(name, str) or not name:
        raise TypeError(
            f"register_external_kernel: name must be a non-empty str, got "
            f"{type(name).__name__}"
        )
    if not callable(callable_):
        raise TypeError(
            f"register_external_kernel: kernel for {name!r} must be "
            f"callable, got {type(callable_).__name__}"
        )
    _REGISTRY[name] = callable_


def unregister_external_kernel(name: str) -> None:
    """Remove the kernel registered under ``name``.

    Raises:
        KeyError: no kernel is registered under ``name`` (loud — never
            silent).
    """
    if not isinstance(name, str) or not name:
        raise TypeError(
            f"unregister_external_kernel: name must be a non-empty str, got "
            f"{type(name).__name__}"
        )
    try:
        del _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unregister_external_kernel: no external kernel registered "
            f"under {name!r}"
        ) from None


def get_external_kernel(name: str) -> Optional[Callable]:
    """The callable registered under ``name`` (``None`` if unknown).

    Internal-use contract: the numpy backend resolves the ``external_call``
    op's ``name`` attribute through this lookup at run time and turns
    ``None`` into ``core.BackendError`` naming the kernel. Public for
    diagnostics and for round-2 adapter host-dispatch.
    """
    return _REGISTRY.get(name)
