"""External-kernel registry — the RUN-TIME side of the external-kernel mechanism.

Kernels are registered by NAME in this process-global registry and resolved at
run time by the numpy backend when it executes an ``external_call`` op. The
mechanism is kernel-agnostic: the callable contract is plain numpy arrays in,
plain numpy arrays (or a ``core.Tensor`` / tuple of them) out — a triton
kernel written against this interface (in a DIFFERENT repo) will plug in via
the same ``register_external_kernel`` call.

The registry is PER-BACKEND: each name maps to a dict of backend slots
(``{backend_or_None: callable}``), where ``None`` is the DEFAULT slot.
Resolution at dispatch time is ``get_external_kernel(name, backend)``: the
exact backend slot first, then the default slot, then ``None``. A kernel
registered with the plain two-argument form lives in the default slot and is
used by every backend. A portable decomposition (a ``@etl.defn`` graph
function, registered via :func:`register_portable`) is the OPTIONAL graph-side
implementation: ``vmap``/``grad`` fall back to inlining it (pre-registered
fallback rules under ``external:<name>``) when no explicit transform rules
exist. ``register_external_kernel`` returns an :class:`ExternalKernel` handle
offering decorator-style registration of backend impls, the portable, and the
transform rules.

Binding rules (see ``etl/CONTEXT.md``, "External kernels"):

- Callable signature: ``fn(*np_arrays) -> ndarray | Tensor | tuple/list of
  them``. The number/dtype/shape of the returned arrays must match the specs
  declared by the ``external_call`` op that invoked the kernel — the numpy
  backend validates count/dtype/shape at run time (``BackendError`` /
  ``ShapeError`` — never a silent coercion).
- Registration is process-global and REPLACES any previous registration in
  the same backend slot (last registration wins — documented overwrite
  semantics; this makes hot-reloading kernels and adapter re-registration
  safe).
- The registry is NOT serialized: graph artifacts carry only the kernel name;
  any process that runs a graph must re-register its kernels first
  (``BackendError`` naming the kernel otherwise).
- Errors are loud: registering a non-callable / bad name / bad backend raises
  ``TypeError``; unregistering a name that was never registered (neither
  kernel slots nor a portable) raises ``KeyError``; resolving an unknown name
  returns ``None`` (the numpy backend turns that into ``BackendError``).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

__all__ = [
    "ExternalKernel",
    "get_external_kernel",
    "get_portable",
    "register_external_kernel",
    "register_portable",
    "unregister_external_kernel",
]

#: name -> {backend_or_None: callable}. Process-global; the numpy backend and
#: the compiler-adapter host-dispatch resolve through :func:`get_external_kernel`
#: with their backend name (``None`` = the default slot, used by every backend
#: when no exact backend slot is registered).
_REGISTRY: Dict[str, Dict[Optional[str], Callable]] = {}

#: name -> etl.defn function. The optional graph decomposition, traced LAZILY
#: by the fallback transform rules (and potentially by backends) when needed.
_PORTABLES: Dict[str, Callable] = {}


def register_external_kernel(
    name: str, callable_, backend: Optional[str] = None
) -> "ExternalKernel":
    """Register ``callable_`` as the external kernel named ``name``.

    Registers into the ``backend`` slot of the name's slot dict (``None`` =
    the DEFAULT slot, used by every backend that has no exact backend slot).
    Overwrites any previous registration in the same slot (last wins —
    documented; there is no error on re-registration).

    Args:
        name: Non-empty kernel name (the same string passed to
            ``etl.external_call`` in traced graphs).
        callable_: ``fn(*np_arrays) -> ndarray | Tensor | tuple/list of them``.
        backend: Optional backend name (``"numpy"``, ``"iree"``, ...) — any
            string is accepted (backend-neutral; there is no registry check).
            ``None`` registers the default slot.

    Returns:
        An :class:`ExternalKernel` handle for ``name`` (decorator-style
        registration of backend impls, the portable decomposition, and the
        transform rules).

    Raises:
        TypeError: ``name`` is not a non-empty str, ``backend`` is neither
            None nor a str, or ``callable_`` is not callable.
    """
    if not isinstance(name, str) or not name:
        raise TypeError(
            f"register_external_kernel: name must be a non-empty str, got "
            f"{type(name).__name__}"
        )
    if backend is not None and not isinstance(backend, str):
        raise TypeError(
            f"register_external_kernel: backend must be None or a str, got "
            f"{type(backend).__name__}"
        )
    if not callable(callable_):
        raise TypeError(
            f"register_external_kernel: kernel for {name!r} must be "
            f"callable, got {type(callable_).__name__}"
        )
    _REGISTRY.setdefault(name, {})[backend] = callable_
    return ExternalKernel(name)


def unregister_external_kernel(name: str) -> None:
    """Remove every kernel slot AND the registered portable for ``name``.

    Does NOT touch the ``etl.transforms`` rule registries: transform rules
    are graph-level, not run-time registrations, so they survive (re-register
    the portable to restore the fallback rules if you want them back).

    Raises:
        KeyError: the name was never registered (no kernel slots and no
            portable) — loud, never silent.
    """
    if not isinstance(name, str) or not name:
        raise TypeError(
            f"unregister_external_kernel: name must be a non-empty str, got "
            f"{type(name).__name__}"
        )
    has_kernel = name in _REGISTRY
    has_portable = name in _PORTABLES
    if not has_kernel and not has_portable:
        raise KeyError(
            f"unregister_external_kernel: no external kernel registered "
            f"under {name!r}"
        )
    if has_kernel:
        del _REGISTRY[name]
    if has_portable:
        del _PORTABLES[name]


def get_external_kernel(
    name: str, backend: Optional[str] = None
) -> Optional[Callable]:
    """The callable for ``name`` at ``backend`` (``None`` if unknown).

    Resolution: the exact ``backend`` slot first, then the default (``None``)
    slot, then ``None``. ``get_external_kernel(name)`` (no backend) returns
    the default slot — backward compatible with the original single-slot
    registry.

    Internal-use contract: backends resolve the ``external_call`` op's
    ``name`` attribute through this lookup at run time (numpy passes its
    backend name; the iree host-dispatch does the same) and turn ``None``
    into ``core.BackendError`` naming the kernel. Public for diagnostics.
    """
    slots = _REGISTRY.get(name)
    if slots is None:
        return None
    if backend in slots:
        return slots[backend]
    return slots.get(None)


def register_portable(name: str, fn) -> None:
    """Register the optional graph decomposition for kernel ``name``.

    ``fn`` must be an ``@etl.defn`` function (it carries the ``__etl_defn__``
    marker, like block portables); it is traced LAZILY when a fallback rule
    fires (or by a backend that consumes portables). Pre-registers the
    decomposition FALLBACK rules under ``external:<name>``: the vjp fallback
    ALWAYS, the batching fallback only when no explicit batching rule is
    already registered (explicit rules win; explicit rules registered later
    overwrite the fallbacks — last-wins, same decl-time pattern as
    ``etl/block/decl.py``).

    Args:
        name: Non-empty kernel name.
        fn: An ``@etl.defn`` function mapping the kernel's tensor operands to
            its declared result tensors.

    Raises:
        TypeError: ``name`` is not a non-empty str, or ``fn`` is not an
            ``@etl.defn`` function.
    """
    if not isinstance(name, str) or not name:
        raise TypeError(
            f"register_portable: name must be a non-empty str, got "
            f"{type(name).__name__}"
        )
    _validate_portable(name, fn)
    _PORTABLES[name] = fn
    # Fallback rules: diffing always falls back to the decomposition; the
    # batching fallback only fills a slot no explicit rule claimed yet.
    from .external_rules import (  # lazy: import acyclicity
        register_portable_batching_fallback,
        register_portable_diff_fallback,
    )
    from etl import transforms

    register_portable_diff_fallback(name)
    if f"external:{name}" not in transforms.batching_rules:
        register_portable_batching_fallback(name)


def get_portable(name: str) -> Optional[Callable]:
    """The registered portable (``@etl.defn``) implementation for ``name``."""
    return _PORTABLES.get(name)


def _validate_portable(name: str, fn: Any) -> None:
    """Pure validation of a portable implementation (no mutation).

    Portable implementations must be ``@etl.defn`` functions: they are traced
    lazily into ordinary graphs at transform/run time.
    """
    if not callable(fn) or getattr(fn, "__etl_defn__", None) is None:
        raise TypeError(
            f"register_portable: portable implementation for external kernel "
            f"{name!r} must be an etl.defn function (a graph decomposition "
            f"traced lazily), got {fn!r}"
        )


class ExternalKernel:
    """Handle returned by :func:`register_external_kernel` — the registration
    surface for one external kernel.

    Decorator-friendly methods (each returns ``fn``): :meth:`impl` registers
    a per-backend kernel, :meth:`portable` registers the ``@etl.defn`` graph
    decomposition, and :meth:`batching_rule` / :meth:`vjp_rule` /
    :meth:`jvp_rule` register transform rules under ``external:<name>``.

    Attributes:
        name: The kernel name this handle registers under.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def impl(self, backend: str, fn: Optional[Callable] = None):
        """Register a per-backend kernel implementation for this handle.

        Both forms work::

            handle.impl("numpy", my_kernel)          # direct registration
            @handle.impl("iree")                     # decorator form
            def iree_kernel(*arrays): ...

        ``backend`` is any string (backend-neutral — no registry check);
        registration itself goes through :func:`register_external_kernel`.
        Returns ``fn``.
        """
        if fn is None:

            def decorator(f: Callable) -> Callable:
                register_external_kernel(self.name, f, backend=backend)
                return f

            return decorator
        register_external_kernel(self.name, fn, backend=backend)
        return fn

    def portable(self, fn: Callable) -> Callable:
        """Register ``fn`` (an ``@etl.defn`` function) as the portable graph
        decomposition (``register_portable(self.name, fn)``). Returns ``fn``."""
        register_portable(self.name, fn)
        return fn

    def batching_rule(self, fn: Callable) -> Callable:
        """Register a vectorize/vmap batching rule under ``external:<name>``.

        Rule contract (owned by etl.transforms): ``fn(op, operands, axes) ->
        (new_values, new_axes)`` — ``operands`` is the tuple of ``ir.Value``
        operands, ``axes`` the aligned ``MappedAxes`` metadata; ``new_values``
        is aligned with ``op.results``, ``new_axes`` with ``new_values``.
        Returns ``fn``.
        """
        from .external_rules import register_batching_rule

        register_batching_rule(self.name, fn)
        return fn

    def vjp_rule(self, fn: Callable) -> Callable:
        """Register a reverse-mode derivative rule under ``external:<name>``.

        Stored in ``transforms.vjp_rules`` per the etl/CONTEXT.md contract.
        Returns ``fn``.
        """
        from .external_rules import register_vjp_rule

        register_vjp_rule(self.name, fn)
        return fn

    def jvp_rule(self, fn: Callable) -> Callable:
        """Register a forward-mode derivative rule under ``external:<name>``.

        Stored in ``transforms.jvp_rules`` (when absent, transforms derives
        jvp from the vjp rule). Returns ``fn``.
        """
        from .external_rules import register_jvp_rule

        register_jvp_rule(self.name, fn)
        return fn

    def __repr__(self) -> str:
        return f"ExternalKernel(name={self.name!r})"
