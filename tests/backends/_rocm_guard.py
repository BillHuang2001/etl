"""ROCm/HIP device guards for the backend test suites (shared helper).

NOT a test module (underscore prefix — pytest never collects it, same
convention as ``_adapter_utils.py``). Import it package-qualified::

    from tests.backends import _rocm_guard as guard

Why: the device-resident / same-device-loop suites
(``test_iree_device_resident.py``, ``test_iree_same_device_loop.py``, the
emitters-parity smokes, …) guard their GPU legs on CUDA hardware via
``nvidia-smi`` — on an AMD-only host those legs skip unconditionally even
though IREE can drive the AMD GPU through its ``hip`` HAL driver. This
helper gives the ROCm counterpart of that guard, WITHOUT ``nvidia-smi``:

* :func:`rocm_available` is the AUTHORITATIVE capability probe: it asks
  IREE's ``hip`` driver how many devices it reports (a non-empty answer
  means the ROCm runtime + HIP driver are actually usable, not merely
  installed). Every exception is treated as "unavailable".
* :func:`rocm_tools_present` is a fast, secondary signal only — whether a
  ROCm userspace tool (``rocminfo`` / ``rocm-smi`` / ``amdgpu-arch`` /
  ``rocm_agent_enumerator``) is on ``PATH``. It never decides availability
  on its own (``amdgpu-arch`` in particular is frequently absent).
* etl ``Device("rocm", idx)`` maps to IREE's 1-based ``hip`` driver
  ``device_id = idx + 1`` (the same convention as the cuda guard), and
  ``"hip"`` is an accepted etl alias for the ``"rocm"`` kind that resolves
  to the very same execution path while PRESERVING the ``"hip"`` label.

The module is importable with or without ``iree`` installed: all IREE (and
``etl``) imports are lazy, inside the functions.
"""

from __future__ import annotations

import shutil

import pytest

#: ROCm userspace tools whose presence is a (secondary) signal that a ROCm
#: stack is installed. ``amdgpu-arch`` is deliberately included even though
#: it is often missing — any ONE of these is enough for the pre-check.
ROCM_TOOL_NAMES = ("rocminfo", "rocm-smi", "amdgpu-arch", "rocm_agent_enumerator")


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def rocm_tool_paths() -> dict:
    """``{tool_name: resolved_path_or_None}`` for the ROCm toolchain tools."""
    return {name: shutil.which(name) for name in ROCM_TOOL_NAMES}


def rocm_tools_present() -> bool:
    """True when at least one ROCm userspace tool is on ``PATH``.

    A fast pre-check / diagnostic only — NOT a capability statement. Use
    :func:`rocm_available` to decide whether a device can actually be used.
    """
    return any(shutil.which(name) is not None for name in ROCM_TOOL_NAMES)


def hip_available_devices() -> list:
    """Devices IREE's ``hip`` HAL driver reports (``[]`` when unavailable).

    The authoritative probe. Returns ``[]`` — never raising — when ``iree``
    is missing, the HIP driver cannot be created, or the query fails, so
    callers can use it as a plain availability predicate.
    """
    try:
        import iree.runtime as rt
    except Exception:  # noqa: BLE001 — iree absent/unimportable => unavailable
        return []
    try:
        return list(rt.get_driver("hip").query_available_devices())
    except Exception:  # noqa: BLE001 — no ROCm runtime / driver failure
        return []


def rocm_device_count() -> int:
    """Number of usable ROCm devices (``0`` when none/unavailable)."""
    return len(hip_available_devices())


def available_rocm_indices() -> list:
    """etl device indices (IREE ``device_id`` - 1) of the usable ROCm devices.

    IREE's ``hip`` driver reports 1-based ``device_id`` s; etl
    ``Device("rocm", idx)`` therefore corresponds to ``device_id = idx + 1``.
    Malformed entries are skipped rather than crashing the guard.
    """
    indices = []
    for entry in hip_available_devices():
        try:
            indices.append(int(entry["device_id"]) - 1)
        except (TypeError, KeyError, ValueError, IndexError):
            continue
    return indices


def rocm_available() -> bool:
    """True when at least one usable ROCm/HIP device exists.

    Authoritative: TRUE iff IREE's ``hip`` driver reports >=1 device (any
    failure counts as unavailable). The ROCm toolchain presence is a
    secondary signal only — see :func:`rocm_tools_present`.
    """
    return rocm_device_count() > 0


def unavailable_reason() -> str:
    """Human-readable reason/context for a missing ROCm capability."""
    tools = ", ".join(
        f"{name}={'yes' if shutil.which(name) else 'no'}" for name in ROCM_TOOL_NAMES
    )
    return (
        "no usable ROCm/HIP device: IREE's 'hip' HAL driver reported no "
        "devices (the authoritative capability probe); ROCm toolchain on "
        f"PATH: {tools}"
    )


# ---------------------------------------------------------------------------
# devices
# ---------------------------------------------------------------------------


def rocm_device(index: int = 0, kind: str = "rocm"):
    """The etl ``Device`` for ROCm index ``index``.

    ``kind`` defaults to the canonical ``"rocm"``; ``"hip"`` is the accepted
    alias (same execution path, label preserved).
    """
    import etl

    return etl.core.Device(kind, index)


def pick_rocm_device_index() -> int:
    """etl index of the first usable ROCm device; ``pytest.skip`` otherwise."""
    require_rocm()
    return available_rocm_indices()[0]


# ---------------------------------------------------------------------------
# pytest gating
# ---------------------------------------------------------------------------


def require_rocm(index: int = 0, kind: str = "rocm"):
    """``pytest.skip`` unless a ROCm device is usable; return the etl device.

    Gating a test on the return value keeps the guard and the device it
    guards in lock-step::

        @pytest.fixture(scope="module")
        def rocm():
            return guard.require_rocm()
    """
    if not rocm_available():
        pytest.skip(unavailable_reason())
    return rocm_device(index, kind)


#: Alias — the same gate under the name some suites prefer.
skip_unless_rocm = require_rocm
