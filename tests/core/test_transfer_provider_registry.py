"""Per-backend device-transfer provider registry (regression suite).

Pins the ORDER-INDEPENDENT per-backend transfer-provider design: optional
compiler adapters (iree/xla) register their providers under per-backend
slots keyed by ``(kind, backend)``, ``etl.backends.registry.get(name)``
records the PREFERRED transfer backend per device kind, and ``Tensor.to``
resolves the preferred backend's slot → the flat DEFAULT slot → explicit
``DeviceError``. No backend's activation can clobber another backend's
provider or the flat slot (no last-wins across backends), so the active
transfer provider depends only on the last explicit backend resolution —
never on module import/activation order.

Pure core-level simulation: sentinel providers under a made-up device kind
(never ``"cuda"`` — the flat ``"cuda"`` slot holds ``etl.backends``' lazy
iree thunk, which must not be invoked in CPU-only tests), no real adapters,
no GPU, no iree import. The preference hook is exercised through the same
internal function ``etl.backends.registry.get`` calls
(``_note_backend_transfer_preference``); one test pins that the real
``etl.backends.get("numpy")`` never disturbs preferences (numpy serves no
transfer kinds).
"""

import itertools

import numpy as np
import pytest

import etl
from etl.core import Device, Tensor
from etl.core.tensor import (
    _BACKEND_DEVICE_TRANSFER_PROVIDERS,
    _DEVICE_TRANSFER_PROVIDERS,
    _PREFERRED_TRANSFER_BACKENDS,
    _get_backend_device_transfer_provider,
    _get_preferred_transfer_backend,
    _note_backend_transfer_preference,
    register_backend_device_transfer_provider,
    register_device_transfer_provider,
)

# A made-up device kind (never "cuda" — see module docstring).
KIND = "xpu_backend_registry"
DEV = Device(KIND, 0)

ALPHA, BETA = "alpha_backend", "beta_backend"  # sentinel "adapters"


def _snapshot():
    return (
        dict(_BACKEND_DEVICE_TRANSFER_PROVIDERS),
        dict(_PREFERRED_TRANSFER_BACKENDS),
        dict(_DEVICE_TRANSFER_PROVIDERS),
    )


def _restore(snap):
    slots, preferred, flat = snap
    _BACKEND_DEVICE_TRANSFER_PROVIDERS.clear()
    _BACKEND_DEVICE_TRANSFER_PROVIDERS.update(slots)
    _PREFERRED_TRANSFER_BACKENDS.clear()
    _PREFERRED_TRANSFER_BACKENDS.update(preferred)
    _DEVICE_TRANSFER_PROVIDERS.clear()
    _DEVICE_TRANSFER_PROVIDERS.update(flat)


@pytest.fixture(autouse=True)
def _clean_registries():
    """Snapshot/restore the process-global registries around every test."""
    snap = _snapshot()
    yield
    _restore(snap)


def _make_provider(name, calls):
    def provider(host_tensor, device):
        calls.append((host_tensor, device))
        return f"placed-by-{name}"

    return provider


def _host_tensor():
    return Tensor(np.zeros((2, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# per-backend slot registration semantics
# ---------------------------------------------------------------------------


def test_per_backend_registration_never_touches_flat_slot():
    # A per-backend registration under (KIND, ALPHA) leaves the flat slot
    # for KIND untouched (adapters must not use the last-wins flat hook).
    register_device_transfer_provider(KIND, lambda t, d: "flat-default")
    register_backend_device_transfer_provider(KIND, ALPHA, lambda t, d: "alpha")
    assert _DEVICE_TRANSFER_PROVIDERS[KIND]("x", DEV) == "flat-default"
    assert _get_backend_device_transfer_provider(KIND, ALPHA)("x", DEV) == "alpha"
    assert _get_backend_device_transfer_provider(KIND, BETA) is None


def test_per_backend_slots_are_keyed_per_backend():
    # Both backends' providers coexist under their own keys — registering
    # BETA must NOT clobber ALPHA (the old last-wins flat-slot behavior).
    register_backend_device_transfer_provider(KIND, ALPHA, _make_provider(ALPHA, []))
    register_backend_device_transfer_provider(KIND, BETA, _make_provider(BETA, []))
    assert _get_backend_device_transfer_provider(KIND, ALPHA) is not None
    assert _get_backend_device_transfer_provider(KIND, BETA) is not None
    assert (
        _get_backend_device_transfer_provider(KIND, ALPHA)
        is not _get_backend_device_transfer_provider(KIND, BETA)
    )


def test_registration_alone_never_sets_preference():
    # Module-import equivalence: a plain per-backend registration (what an
    # adapter module's register() does) must NOT make the backend preferred
    # — Tensor.to keeps resolving through the flat DEFAULT slot.
    calls = []
    register_backend_device_transfer_provider(KIND, ALPHA, _make_provider(ALPHA, calls))
    register_device_transfer_provider(KIND, _make_provider("flat", calls))
    out = _host_tensor().to(DEV)
    assert out == "placed-by-flat"  # flat slot served, not ALPHA's slot
    assert len(calls) == 1


def test_validation_matches_flat_hook_style():
    with pytest.raises(TypeError, match="non-empty string"):
        register_backend_device_transfer_provider("", ALPHA, lambda t, d: None)
    with pytest.raises(TypeError, match="non-empty string"):
        register_backend_device_transfer_provider(KIND, "", lambda t, d: None)
    with pytest.raises(TypeError, match="callable"):
        register_backend_device_transfer_provider(KIND, ALPHA, "not-callable")
    with pytest.raises(TypeError, match="non-empty string"):
        _note_backend_transfer_preference("")


# ---------------------------------------------------------------------------
# Tensor.to resolution: preferred (kind, backend) slot -> flat slot -> error
# ---------------------------------------------------------------------------


def test_to_resolves_preferred_backend_slot():
    flat_calls, alpha_calls, beta_calls = [], [], []
    register_device_transfer_provider(KIND, _make_provider("flat", flat_calls))
    register_backend_device_transfer_provider(KIND, ALPHA, _make_provider(ALPHA, alpha_calls))
    register_backend_device_transfer_provider(KIND, BETA, _make_provider(BETA, beta_calls))
    _note_backend_transfer_preference(ALPHA)  # what etl.backends.get(ALPHA) does
    assert _get_preferred_transfer_backend(KIND) == ALPHA
    out = _host_tensor().to(DEV)
    assert out == f"placed-by-{ALPHA}"
    assert len(alpha_calls) == 1
    assert len(beta_calls) == 0
    assert len(flat_calls) == 0  # the preferred slot wins over the flat slot


def test_to_preference_switches_back_and_forth():
    # Re-resolving the other backend switches Tensor.to to ITS provider —
    # both stay registered the whole time (nothing was clobbered).
    alpha_calls, beta_calls = [], []
    register_backend_device_transfer_provider(KIND, ALPHA, _make_provider(ALPHA, alpha_calls))
    register_backend_device_transfer_provider(KIND, BETA, _make_provider(BETA, beta_calls))
    _note_backend_transfer_preference(ALPHA)
    assert _host_tensor().to(DEV) == f"placed-by-{ALPHA}"
    _note_backend_transfer_preference(BETA)
    assert _host_tensor().to(DEV) == f"placed-by-{BETA}"
    _note_backend_transfer_preference(ALPHA)
    assert _host_tensor().to(DEV) == f"placed-by-{ALPHA}"
    assert len(alpha_calls) == 2
    assert len(beta_calls) == 1
    # Both per-backend slots are still registered (no clobber ever happened).
    assert _get_backend_device_transfer_provider(KIND, ALPHA) is not None
    assert _get_backend_device_transfer_provider(KIND, BETA) is not None


def test_to_falls_back_to_flat_slot_without_preference():
    # No backend was resolved via get(): the flat DEFAULT slot serves
    # (backward compat with the pre-per-backend contract).
    calls = []
    register_device_transfer_provider(KIND, _make_provider("flat", calls))
    out = _host_tensor().to(DEV)
    assert out == "placed-by-flat"
    assert len(calls) == 1


def test_to_preference_without_slot_falls_back_to_flat():
    # A preferred backend whose per-backend slot is absent (never happens in
    # practice — preference is only recorded for kinds the backend serves —
    # but resolution must degrade to the flat slot, never crash).
    calls = []
    register_device_transfer_provider(KIND, _make_provider("flat", calls))
    _note_backend_transfer_preference(BETA)  # BETA has no (KIND, BETA) slot
    assert _get_preferred_transfer_backend(KIND) is None  # not recorded: no slot
    out = _host_tensor().to(DEV)
    assert out == "placed-by-flat"


def test_to_with_no_provider_anywhere_raises():
    with pytest.raises(etl.DeviceError, match="no device-transfer provider"):
        _host_tensor().to(DEV)


# ---------------------------------------------------------------------------
# activation-order permutations (the defect's regression pin)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "order",
    list(itertools.permutations([ALPHA, BETA]))
    + list(itertools.permutations([ALPHA, BETA, "gamma_backend"])),
)
def test_activation_order_permutations_transfer_via_last_activated(order):
    # Simulates mixed-backend processes under EVERY import/activation order
    # (adapters registering per-backend slots, etl.backends.get(name)
    # recording the preference): Tensor.to must always route through the
    # LAST resolved backend's own provider — never a clobbered/wrong one —
    # and every backend's provider must stay registered throughout.
    calls = {backend: [] for backend in order}
    # Each "activation" = per-backend registration + the get() preference hook.
    for backend in order:
        register_backend_device_transfer_provider(
            KIND, backend, _make_provider(backend, calls[backend])
        )
        _note_backend_transfer_preference(backend)
    # Deterministic routing to the LAST activated backend.
    assert _host_tensor().to(DEV) == f"placed-by-{order[-1]}"
    assert len(calls[order[-1]]) == 1
    for backend in order[:-1]:
        assert _get_backend_device_transfer_provider(KIND, backend) is not None
        assert len(calls[backend]) == 0  # never consulted, never clobbered


def test_registry_get_numpy_disturbs_no_preference():
    # The real etl.backends.get("numpy") fast path runs the preference hook
    # but numpy registers no per-backend slots — preferences are untouched
    # (a no-op), and no error is raised.
    etl.backends.get("numpy")
    assert _PREFERRED_TRANSFER_BACKENDS.get(KIND) is None
