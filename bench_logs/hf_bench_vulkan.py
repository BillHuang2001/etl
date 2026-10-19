#!/usr/bin/env python3
"""**Device/provenance helpers + environment self-test** for the Vulkan benchmark legs.

``etl`` has a **first-class ``vulkan`` device kind**: ``etl.core.Device("vulkan", N)``
is an ordinary device, ``etl.pipeline`` infers ``target_backends=["vulkan-spirv"]``
from it for the ``iree`` backend, the IREE adapter maps the etl device index onto
IREE's own Vulkan device enumeration, and ``Tensor.to(Device("vulkan", N))`` has a
registered transfer provider. A Vulkan leg therefore needs **no library change and no
runtime shim** — ``bench_logs/hf_bench.py`` builds and runs it through the unmodified
public API (``etl_hf.build`` / ``etl_hf.run``). Nothing in this module patches etl,
and nothing under ``./etl/`` or ``./etl_hf/`` imports it.

What is left here is what a *benchmark record* needs to explain itself:

* :data:`VULKAN_KIND` / :data:`VULKAN_TARGET` — the etl device kind and the IREE
  target backend ``etl.pipeline`` derives from it;
* :func:`device_info` — what IREE's Vulkan driver actually enumerates (name, IREE's
  ``device_id``/``path``, count) — the view the *runtime* has;
* :func:`physical_device_properties` — ``VkPhysicalDeviceProperties`` of the same
  physical device, read straight from the Vulkan loader (name / apiVersion /
  driverVersion / vendorID / deviceType) — the provenance a record states next to
  its numbers;
* a ``--check`` self-test that enumerates the device **and** proves the public path
  works end to end (``Device("vulkan", 0)`` + a tiny ``defn`` graph built with
  ``backend="iree"`` and run device-resident, compared against numpy).

THE DEVICE: **lavapipe is software Vulkan on the CPU**
-----------------------------------------------------
The only Vulkan device this host exposes is Mesa **lavapipe** (``deviceType = CPU``),
so every number measured on it is a *Vulkan-path correctness/perf datapoint*, never a
GPU measurement. The AMD GPU of this host cannot run Vulkan at all (Mesa RADV refuses
CDNA3 by policy, AMDVLK's PAL has no CDNA path) — evidence in
``bench_logs/vulkan_findings.md``.

Two environment facts the harness depends on (both documented in that file):

* **ICD.** IREE 3.11's Vulkan driver hard-denylists device names starting with
  ``llvmpipe``, so lavapipe is only enumerable through a *renamed* ICD, pinned by
  ``VK_ICD_FILENAMES``. The path is machine-local and never hard-coded: the runner
  exports ``ETL_VULKAN_ICD`` and ``hf_bench.child_env`` forwards it to the leg as
  ``VK_ICD_FILENAMES``.
* **Hang.** With ``VK_ICD_FILENAMES`` unset this host's loader dlopens the wedged
  radeon ICD and blocks — so the ICD is always pinned explicitly, and a probe that
  touches Vulkan should be run under a timeout.

``iree.runtime`` on CPython 3.14 can emit a nanobind teardown dump at interpreter
exit; scripts that must exit cleanly flush and call ``os._exit(0)`` (the ``--check``
path does).
"""
from __future__ import annotations

import os
import sys
from typing import Any

__all__ = [
    "VULKAN_KIND",
    "VULKAN_TARGET",
    "device_info",
    "physical_device_properties",
]

#: The etl ``core.Device`` kind these helpers describe, and the IREE compile target
#: that ``etl.pipeline`` derives from such a device (a device kind is not a
#: target-backend name: IREE spells Vulkan ``vulkan-spirv``).
VULKAN_KIND = "vulkan"
VULKAN_TARGET = "vulkan-spirv"


# ---------------------------------------------------------------------------
# provenance: what the IREE runtime sees
# ---------------------------------------------------------------------------
def device_info() -> dict[str, Any]:
    """Return what the IREE Vulkan driver actually enumerates.

    ``{"count": n, "devices": [info dicts], "name": …, "device_id": …,
    "path": …, "dump_device_info": …}`` — the evidence a report needs. Creates
    (and caches) the IREE Vulkan *driver* only; it does not create a device.

    Raises:
        ImportError: ``iree.runtime`` is unavailable.
    """
    import iree.runtime as rt

    driver = rt.get_driver(VULKAN_KIND)
    infos = driver.query_available_devices()
    record: dict[str, Any] = {"count": len(infos), "devices": list(infos)}
    if infos:
        first = infos[0]
        record["name"] = first.get("name")
        record["device_id"] = first.get("device_id")
        record["path"] = first.get("path")
    try:
        # ``dump_device_info`` PRINTS the device properties to the process stdout
        # and returns ``None`` — capture stdout at the file-descriptor level to
        # read it (Python's ``contextlib.redirect_stdout`` does not see it).
        returned = driver.dump_device_info(0)
        record["dump_device_info"] = (
            f"<printed to the process stdout; returned {returned!r}>"
        )
    except Exception as exc:  # noqa: BLE001 - evidence only, never load-bearing
        record["dump_device_info"] = f"<dump_device_info failed: {type(exc).__name__}: {exc}>"
    return record


def physical_device_properties(index: int = 0) -> dict[str, Any]:
    """``VkPhysicalDeviceProperties`` of the enumerated device, read via the loader.

    ``iree.runtime``'s Vulkan ``query_available_devices()`` reports only the name, the
    path (UUID) and IREE's hashed ``device_id``, so the *apiVersion* / *driverVersion* a
    record must state come straight from Vulkan itself: this opens the loader
    (``libvulkan.so.1``, i.e. whatever ``VK_ICD_FILENAMES`` selects), creates a minimal
    instance, reads the properties of physical device ``index`` and destroys the
    instance again — ~2 ms, no device, no queue, nothing the IREE path can observe.

    Args:
        index: which physical device to read (0 = the one the leg runs on).

    Returns:
        dict: ``name``, ``api_version`` / ``driver_version`` (both as ``M.m.p`` strings
        decoded with ``VK_MAKE_VERSION``, plus their raw uint32), ``vendor_id``,
        ``device_id``, ``device_type`` (``CPU`` / ``GPU`` / …) and ``vk_icd_filenames``.

    Raises:
        RuntimeError: the loader is unavailable, an instance cannot be created, or
            ``index`` is out of range — never a silently empty answer.
    """
    import ctypes

    VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
    #: ``VkPhysicalDeviceType`` (``VK_PHYSICAL_DEVICE_TYPE_*``, in enum order).
    DEVICE_TYPES = ("other", "integrated GPU", "discrete GPU", "virtual GPU", "CPU")

    class VkApplicationInfo(ctypes.Structure):
        _fields_ = [
            ("sType", ctypes.c_uint32),
            ("pNext", ctypes.c_void_p),
            ("pApplicationName", ctypes.c_char_p),
            ("applicationVersion", ctypes.c_uint32),
            ("pEngineName", ctypes.c_char_p),
            ("engineVersion", ctypes.c_uint32),
            ("apiVersion", ctypes.c_uint32),
        ]

    class VkInstanceCreateInfo(ctypes.Structure):
        _fields_ = [
            ("sType", ctypes.c_uint32),
            ("pNext", ctypes.c_void_p),
            ("flags", ctypes.c_uint32),
            ("pApplicationInfo", ctypes.POINTER(VkApplicationInfo)),
            ("enabledLayerCount", ctypes.c_uint32),
            ("ppEnabledLayerNames", ctypes.c_void_p),
            ("enabledExtensionCount", ctypes.c_uint32),
            ("ppEnabledExtensionNames", ctypes.c_void_p),
        ]

    def decode(version: int) -> str:
        return f"{(version >> 22) & 0x7FF}.{(version >> 12) & 0x3FF}.{version & 0xFFF}"

    try:
        loader = ctypes.CDLL("libvulkan.so.1")
    except OSError as exc:
        raise RuntimeError(f"libvulkan.so.1 is not loadable: {exc}") from None
    loader.vkCreateInstance.restype = ctypes.c_int32
    loader.vkEnumeratePhysicalDevices.restype = ctypes.c_int32
    loader.vkDestroyInstance.restype = None
    application = VkApplicationInfo(
        sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pNext=None,
        pApplicationName=b"etl-hf-bench",
        applicationVersion=1,
        pEngineName=b"etl-hf-bench",
        engineVersion=1,
        # ``VK_API_VERSION_1_0`` — an instance (not a device) only needs the baseline
        # version, and 0 is explicitly rejected by the loader's validation layer.
        apiVersion=(1 << 22) | (0 << 12) | 0,
    )
    create_info = VkInstanceCreateInfo(
        sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pNext=None,
        flags=0,
        pApplicationInfo=ctypes.pointer(application),
        enabledLayerCount=0,
        ppEnabledLayerNames=None,
        enabledExtensionCount=0,
        ppEnabledExtensionNames=None,
    )
    instance = ctypes.c_void_p()
    result = loader.vkCreateInstance(ctypes.byref(create_info), None, ctypes.byref(instance))
    if result != 0:
        # VK_ERROR_INCOMPATIBLE_DRIVER (-9) is what a stale VK_ICD_FILENAMES produces.
        raise RuntimeError(f"vkCreateInstance failed with VkResult {result}")
    try:
        count = ctypes.c_uint32(0)
        result = loader.vkEnumeratePhysicalDevices(
            instance, ctypes.byref(count), None
        )
        if result != 0:
            raise RuntimeError(f"vkEnumeratePhysicalDevices failed with VkResult {result}")
        if index >= count.value:
            raise RuntimeError(
                f"only {count.value} physical Vulkan device(s) enumerated, index {index} "
                "does not exist"
            )
        devices = (ctypes.c_void_p * count.value)()
        result = loader.vkEnumeratePhysicalDevices(
            instance, ctypes.byref(count), devices
        )
        if result != 0:
            raise RuntimeError(f"vkEnumeratePhysicalDevices(2) failed with VkResult {result}")
        # ``VkPhysicalDeviceProperties`` is apiVersion, driverVersion, vendorID,
        # deviceID, deviceType (u32 each), deviceName[256], ... — a raw 4096-byte
        # buffer is comfortably larger than the real struct, and only its leading
        # fields are read here.
        props = ctypes.create_string_buffer(4096)
        loader.vkGetPhysicalDeviceProperties(devices[index], props)
        api_version, driver_version, vendor_id, device_id, device_type = (
            ctypes.c_uint32.from_buffer_copy(props, offset).value
            for offset in (0, 4, 8, 12, 16)
        )
        name = props.raw[20:20 + 256].split(b"\x00", 1)[0].decode("utf-8", "replace")
    finally:
        loader.vkDestroyInstance(instance, None)
    return {
        "name": name,
        "api_version": decode(api_version),
        "api_version_raw": api_version,
        "driver_version": decode(driver_version),
        "driver_version_raw": driver_version,
        "driver_version_scheme": "VK_MAKE_VERSION (Mesa/lavapipe spelling)",
        "vendor_id": f"0x{vendor_id:04x}",
        "device_id": device_id,
        "device_type": DEVICE_TYPES[device_type] if device_type < len(DEVICE_TYPES)
        else f"unknown ({device_type})",
        "vk_icd_filenames": os.environ.get("VK_ICD_FILENAMES"),
    }


# ---------------------------------------------------------------------------
# self-check: the device, and the public etl path onto it
# ---------------------------------------------------------------------------
def _round_trip(index: int = 0) -> dict[str, Any]:
    """Build + run a tiny ``@etl.defn`` graph on the vulkan device (public API only).

    Deliberately minimal and model-free: a ``(4, 4) @ (4, 4)`` matmul traced with
    :func:`etl.build` on ``Device("vulkan", index)``, its input placed with the
    explicit ``Tensor.to`` transfer, its output compared against the same
    computation on numpy. Exercises exactly the surfaces a benchmark leg uses —
    device kind → inferred ``vulkan-spirv`` target → compile → load → device-resident
    run — with **no** option, monkeypatch or harness helper involved.

    Returns:
        dict: ``device``, ``executable_device``, ``output_device``, ``device_resident``
        and ``max_abs`` (the error against the same matmul on numpy).

    Raises:
        Exception: whatever the library raises — never swallowed (a self-test that
            hides the failure would be worthless).
    """
    import numpy as np

    import etl

    device = etl.Device(VULKAN_KIND, index)

    @etl.defn
    def matmul(x, w):
        return etl.dot(x, w)

    spec = etl.TensorSpec((4, 4), "float32")
    exe = etl.build(matmul, spec, spec, backend="iree", device=device)

    rng = np.random.default_rng(0)
    host_x = rng.standard_normal((4, 4), dtype=np.float32)
    host_w = rng.standard_normal((4, 4), dtype=np.float32)
    out = etl.run(
        exe,
        etl.Tensor(host_x).to(device),
        etl.Tensor(host_w).to(device),
    )
    on_device = str(out.device)
    # The one explicit device->host copy: a vulkan tensor's ``.numpy()`` raises.
    result = out.to(etl.Device("cpu", 0)).numpy()
    expected = host_x.astype(np.float64) @ host_w.astype(np.float64)
    return {
        "device": str(device),
        "executable_device": str(exe.device),
        "output_device": on_device,
        "device_resident": on_device == str(device),
        "max_abs": float(np.max(np.abs(result.astype(np.float64) - expected))),
    }


def _main(argv: list[str]) -> int:
    """``python bench_logs/hf_bench_vulkan.py --check`` — the environment self-test.

    Prints (a) the Vulkan device the loader exposes, (b) what IREE's Vulkan driver
    enumerates for it, (c) the physical device properties a record carries, and
    (d) the result of :func:`_round_trip` — a real ``etl.build``/``etl.run`` on
    ``Device("vulkan", 0)`` with its max-abs error against numpy. Needs ``etl`` +
    ``iree.runtime``; no model and no checkpoint. Exit code 0 means every step
    worked, 1 means one of them raised (the traceback is printed).
    """
    import json
    import traceback

    print(f"etl vulkan device kind : {VULKAN_KIND!r}")
    print(f"derived IREE target    : {VULKAN_TARGET!r}")
    print(f"VK_ICD_FILENAMES       : {os.environ.get('VK_ICD_FILENAMES')!r}")
    print()

    status = 0
    try:
        properties = physical_device_properties(0)
    except Exception:  # noqa: BLE001 - the whole point is to report it
        traceback.print_exc()
        status = 1
    else:
        print("VkPhysicalDeviceProperties (via the loader):")
        for key in ("name", "api_version", "driver_version", "vendor_id", "device_type"):
            print(f"  {key}: {properties.get(key)}")
        print()

    try:
        info = device_info()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        status = 1
    else:
        print("IREE vulkan driver enumeration:")
        for key in ("count", "name", "device_id", "path"):
            print(f"  {key}: {info.get(key)}")
        print(f"  devices: {json.dumps(info['devices'], default=str)}")
        print(f"  dump_device_info:\n{info['dump_device_info']}")
        print()

    try:
        result = _round_trip(0)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        status = 1
    else:
        print("public path (etl.build + etl.run on Device('vulkan', 0)):")
        for key, value in result.items():
            print(f"  {key}: {value}")
        print()
        print(
            "OK — the vulkan device is a first-class etl device kind; no shim, no "
            "library change."
        )
    return status


if __name__ == "__main__":
    if "--check" not in sys.argv[1:] and len(sys.argv) > 1:
        print("usage: hf_bench_vulkan.py [--check]", file=sys.stderr)
        raise SystemExit(2)
    code = _main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # dodge iree.runtime's py3.14 nanobind teardown dump
