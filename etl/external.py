"""External-kernel registry — the RUN-TIME side of the external-kernel mechanism.

Kernels are registered by NAME in this process-global registry and resolved at
run time by the numpy backend when it executes an ``external_call`` op. The
mechanism is kernel-agnostic: the callable contract is plain numpy arrays in,
plain numpy arrays (or a ``core.Tensor`` / tuple of them) out — a triton
kernel written against this interface (in a DIFFERENT repo) will plug in via
the same ``register_external_kernel`` call.

The registry is PER-BACKEND: each name maps to a dict of backend slots
(``{backend_or_None: (callable, device_resident)}``), where ``None`` is the
DEFAULT slot.
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
- DEVICE-RESIDENT mode (explicit opt-in, no magic): ``device_resident=True``
  registers a kernel that receives and returns DEVICE tensors — never host
  numpy arrays. It REQUIRES an explicit per-backend slot (``backend`` must be
  a string; ``device_resident=True`` with ``backend=None`` raises
  ``TypeError`` — a device kernel in the default slot would receive host
  numpy arrays from the numpy backend). Any non-None backend string is
  accepted as today (backend-neutral); in v1 only the iree adapter consumes
  the flag: it passes the boundary's device-resident ``core.Tensor``
  operands DIRECTLY to the kernel (never ``.numpy()``-staged) and validates
  device-payload results METADATA-ONLY (count/dtype/shape — never a host
  copy). Host-mode kernels (the default, and every pre-existing
  registration) keep the exact numpy-in/numpy-out contract.
- Registration is process-global and REPLACES any previous registration in
  the same backend slot (last registration wins — documented overwrite
  semantics; this makes hot-reloading kernels and adapter re-registration
  safe). The device-resident mode is stored per slot and is replaced along
  with the callable on re-registration.
- The registry is NOT serialized: graph artifacts carry only the kernel name;
  any process that runs a graph must re-register its kernels first
  (``BackendError`` naming the kernel otherwise).
- Errors are loud: registering a non-callable / bad name / bad backend /
  non-bool ``device_resident`` raises ``TypeError``; unregistering a name
  that was never registered (neither kernel slots nor a portable) raises
  ``KeyError``; resolving an unknown name returns ``None`` (the numpy
  backend turns that into ``BackendError``).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

__all__ = [
    "ExternalKernel",
    "get_external_kernel",
    "get_external_kernel_entry",
    "get_portable",
    "register_external_kernel",
    "register_portable",
    "unregister_external_kernel",
]

#: name -> {backend_or_None: (callable, device_resident)}. Process-global;
#: the numpy backend and the compiler-adapter dispatch resolve through
#: :func:`get_external_kernel` / :func:`get_external_kernel_entry` with their
#: backend name (``None`` = the default slot, used by every backend when no
#: exact backend slot is registered).
_REGISTRY: Dict[str, Dict[Optional[str], Tuple[Callable, bool]]] = {}

#: name -> etl.defn function. The optional graph decomposition, traced LAZILY
#: by the fallback transform rules (and potentially by backends) when needed.
_PORTABLES: Dict[str, Callable] = {}


def register_external_kernel(
    name: str,
    callable_,
    backend: Optional[str] = None,
    device_resident: bool = False,
) -> "ExternalKernel":
    """Register ``callable_`` as the external kernel named ``name``.

    Registers into the ``backend`` slot of the name's slot dict (``None`` =
    the DEFAULT slot, used by every backend that has no exact backend slot).
    Overwrites any previous registration in the same slot (last wins —
    documented; there is no error on re-registration) — including its
    device-resident mode.

    Args:
        name: Non-empty kernel name (the same string passed to
            ``etl.external_call`` in traced graphs).
        callable_: ``fn(*np_arrays) -> ndarray | Tensor | tuple/list of them``
            (host mode, the default) or, with ``device_resident=True``, a
            device kernel receiving/returning device tensors (v1: only the
            iree adapter dispatches device kernels).
        backend: Optional backend name (``"numpy"``, ``"iree"``, ...) — any
            string is accepted (backend-neutral; there is no registry check).
            ``None`` registers the default slot.
        device_resident: If True, the kernel is a DEVICE kernel: the iree
            adapter passes the boundary's device-resident ``core.Tensor``
            operands directly (never host numpy arrays) and validates its
            results metadata-only (count/dtype/shape). Requires an explicit
            per-backend registration (see Raises).

    Returns:
        An :class:`ExternalKernel` handle for ``name`` (decorator-style
        registration of backend impls, the portable decomposition, and the
        transform rules).

    Raises:
        TypeError: ``name`` is not a non-empty str, ``backend`` is neither
            None nor a str, ``callable_`` is not callable,
            ``device_resident`` is not a bool, or ``device_resident=True``
            with ``backend=None`` (a device kernel in the default slot would
            receive host numpy arrays from the numpy backend — register it
            under an explicit per-backend slot instead).
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
    if not isinstance(device_resident, bool):
        raise TypeError(
            f"register_external_kernel: device_resident must be a bool, got "
            f"{type(device_resident).__name__}"
        )
    if device_resident and backend is None:
        raise TypeError(
            "register_external_kernel: device_resident=True requires an "
            "explicit backend — the default (backend=None) slot is also "
            "dispatched by the numpy backend, which passes host numpy "
            "arrays, never device tensors; register the device kernel under "
            "a per-backend slot instead, e.g. backend='iree'"
        )
    if not callable(callable_):
        raise TypeError(
            f"register_external_kernel: kernel for {name!r} must be "
            f"callable, got {type(callable_).__name__}"
        )
    _REGISTRY.setdefault(name, {})[backend] = (callable_, device_resident)
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


def get_external_kernel_entry(
    name: str, backend: Optional[str] = None
) -> Optional[Tuple[Callable, bool]]:
    """The ``(callable, device_resident)`` entry for ``name`` at ``backend``.

    The MODE-AWARE lookup behind dispatch: same resolution as
    :func:`get_external_kernel` — the exact ``backend`` slot first, then the
    default (``None``) slot, then ``None`` — but returning the full slot
    entry (callable + its registered device-resident mode) instead of just
    the callable. Used by the iree adapter's boundary dispatch to decide
    between host-mode (numpy in/out) and device-mode (device tensors
    in/out) kernel invocation. ``None`` when the name is unknown.
    """
    slots = _REGISTRY.get(name)
    if slots is None:
        return None
    if backend in slots:
        return slots[backend]
    return slots.get(None)


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
    backend name; the iree dispatch uses the mode-aware
    :func:`get_external_kernel_entry` instead) and turn ``None`` into
    ``core.BackendError`` naming the kernel. Public for diagnostics.
    """
    entry = get_external_kernel_entry(name, backend)
    return entry[0] if entry is not None else None


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

    def impl(
        self,
        backend: str,
        fn: Optional[Callable] = None,
        device_resident: bool = False,
    ):
        """Register a per-backend kernel implementation for this handle.

        Both forms work::

            handle.impl("numpy", my_kernel)              # direct registration
            @handle.impl("iree")                         # decorator form
            def iree_kernel(*arrays): ...

        ``device_resident=True`` registers a DEVICE kernel (see
        :func:`register_external_kernel`): the iree adapter then passes the
        boundary's device-resident tensors directly and validates its
        results metadata-only. A device kernel requires an explicit backend
        slot, so pass a backend string.

        ``backend`` is any string (backend-neutral — no registry check);
        registration itself goes through :func:`register_external_kernel`.
        Returns ``fn``.
        """
        if fn is None:

            def decorator(f: Callable) -> Callable:
                register_external_kernel(
                    self.name,
                    f,
                    backend=backend,
                    device_resident=device_resident,
                )
                return f

            return decorator
        register_external_kernel(
            self.name, fn, backend=backend, device_resident=device_resident
        )
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
