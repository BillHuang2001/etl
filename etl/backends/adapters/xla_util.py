"""jaxlib plumbing helpers for the XLA adapter (see ``xla.py``).

Everything in this module is imported from ``xla.py`` ONLY (it is not part
of any public surface). All jaxlib imports live inside function bodies —
the heavy-import rule applies here exactly as in ``xla.py``.

Acquisition summary (verified against jax 0.10.2 / jaxlib 0.10.2, CPU,
numpy 2.4.6; record every step when upgrading):

1. **PJRT client discovery.** NO standalone ``pjrt_c_api_cpu_plugin.so``
   exists in jaxlib 0.10.2 site-packages (searched exhaustively). The CPU
   PJRT client is EMBEDDED in jaxlib's native ``_xla.so`` and is acquired
   via ``xc.make_cpu_client()``. Client creation prints three INFO lines
   from ``pjrt_client.cc`` to stderr (C++ logging; absl-py is not even
   installed — cosmetic only, no Python knob).
2. **MLIR parsing.** ``_jax_mlir_ext.register_dialects(registry)`` +
   ``context.load_all_available_dialects()`` load the CORE dialects
   (func/cf/arith/…); the separately-shipped ``stablehlo``/``chlo`` .so
   dialects need their OWN ``register_dialect(context)`` calls.
   ``Context(load_on_create_dialects=[...])`` alone does NOT work (the
   name registry lacks stablehlo).
3. **Compilation.** ``client.compile_and_load(mlir_module,
   executable_devices=DeviceList, compile_options=CompileOptions())`` —
   the SAME entry point jax's own ``_src/compiler.py`` uses. The
   bytecode/``mlir_module_to_xla_computation`` route is NOT needed (and
   plain ``client.compile(module)`` hits a missing-compiler-factory
   error on CPU).
4. **Buffer creation.** ``client.buffer_from_pyval`` DOES NOT EXIST in
   jaxlib 0.10.2. The working pure-jaxlib path is ``xc.batched_device_put``
   with a duck-typed aval (``_Aval``) and a ``SingleDeviceSharding``
   instance patched with ``_to_xla_hlo_sharding`` (the pybind entry point
   CALLS that method; jax's frontend sharding subclass provides it).
   ``enable_x64=True`` is MANDATORY — with the default x64-off state,
   float64/int64 inputs are silently truncated to float32/int32.
5. **Execution/serialization.** ``exe.execute([bufs]) -> [ArrayImpl]``;
   ``np.asarray(buf)`` gives a numpy view. ``exe.serialize() -> bytes``
   and ``client.deserialize_executable(bytes, device_list)`` both exist
   and round-trip correctly.
"""

from __future__ import annotations

from typing import Any

from etl import core

__all__ = [
    "_Aval",
    "_StaticShapeError",
    "_resolve_static_shape",
    "_import_xla_runtime",
    "_verify_xla_api_surface",
    "_acquire_cpu_client",
    "_make_mlir_context",
    "_parse_stablehlo_module",
    "_make_buffer_putter",
    "VALIDATED_JAXLIB_VERSION",
]

#: jaxlib version these helpers were validated against.
VALIDATED_JAXLIB_VERSION = "0.10.2"

_PIP_HINT = "pip install etl[xla]"


class _Aval:
    """Duck-typed abstract value for ``xc.batched_device_put``.

    The pybind entry point only reads ``shape``/``dtype``/``weak_type``/
    ``named_shape`` off the aval (jax's ``core.ShapedArray`` would also
    work but lives in the jax frontend, which the adapter never imports).
    """

    __slots__ = ("shape", "dtype", "weak_type", "named_shape")

    def __init__(self, shape: tuple[int, ...], dtype: Any) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.weak_type = False
        self.named_shape: dict[str, Any] = {}


class _StaticShapeError(Exception):
    """Internal: a signature shape is not statically resolvable."""


def _resolve_static_shape(shape: tuple[Any, ...], where: str) -> tuple[int, ...]:
    """Evaluate a declared shape to concrete ints (static-shape gate).

    Accepts plain ints, ``Dim`` with a known size, and ``DimExpr`` that
    evaluates with NO free runtime dims. A ``None`` entry (runtime-dynamic
    dim), a ``Dim`` without a known size, an expression with free dims, or
    any other entry raises ``_StaticShapeError`` naming the offending dim.
    """
    resolved: list[int] = []
    for i, entry in enumerate(shape):
        if entry is None:
            raise _StaticShapeError(
                f"{where}, dim {i} is runtime-dynamic (None) — XLA requires "
                "a static size here"
            )
        if isinstance(entry, bool):
            raise _StaticShapeError(
                f"{where}, dim {i} is a Python bool ({entry!r}) — not a "
                "valid shape entry"
            )
        if isinstance(entry, int):
            resolved.append(entry)
            continue
        if isinstance(entry, core.Dim):
            if entry.size is None:
                raise _StaticShapeError(
                    f"{where}, dim {i} is the symbolic dimension "
                    f"Dim({entry.name!r}) without a known size"
                )
            resolved.append(entry.size)
            continue
        if isinstance(entry, core.DimExpr):
            try:
                resolved.append(entry.evaluate({}))
            except core.ShapeError as exc:
                raise _StaticShapeError(
                    f"{where}, dim {i} is the symbolic expression {entry!r} "
                    "with free runtime dimensions"
                ) from exc
            continue
        raise _StaticShapeError(
            f"{where}, dim {i} is an unsupported shape entry {entry!r} of "
            f"type {type(entry).__name__}"
        )
    return tuple(resolved)


def _import_xla_runtime():
    """Lazily import the jaxlib pieces; raise ``core.BackendError`` if missing.

    Returns ``(xla_client, jaxlib)``. This is the ONLY place jaxlib is
    imported (function body — the heavy-import rule).
    """
    try:
        import jaxlib
        from jaxlib import xla_client as xc
    except ImportError as exc:
        raise core.BackendError(
            "the xla backend requires jaxlib (the CPU PJRT client is "
            f"embedded in it) — install it with `{_PIP_HINT}`"
        ) from exc
    return xc, jaxlib


def _verify_xla_api_surface(xc: Any) -> None:
    """Check the jaxlib API surface the adapter depends on.

    Raises ``core.BackendError`` with a version hint on any drift — the
    adapter never silently degrades to a partial implementation.
    """
    missing = []
    for module_name in (
        "make_cpu_client",
        "batched_device_put",
        "SingleDeviceSharding",
        "HloSharding",
        "DeviceList",
        "CompileOptions",
    ):
        if not hasattr(xc, module_name):
            missing.append(f"xla_client.{module_name}")
    for method_name in ("compile_and_load", "deserialize_executable"):
        if not hasattr(xc.Client, method_name):
            missing.append(f"xla_client.Client.{method_name}")
    if missing:
        raise core.BackendError(
            "jaxlib API drift: the xla adapter needs "
            + ", ".join(missing)
            + f" — the adapter was validated against jaxlib "
            f"{VALIDATED_JAXLIB_VERSION}; upgrade jaxlib "
            f"(`{_PIP_HINT}`) or pin it to a compatible version"
        )
    try:
        import jaxlib.mlir.ir as jm_ir  # noqa: F401
        from jaxlib.mlir._mlir_libs import _jax_mlir_ext
        from jaxlib.mlir.dialects import chlo, stablehlo  # noqa: F401
    except ImportError as exc:
        raise core.BackendError(
            "jaxlib's MLIR bindings are unavailable ("
            f"{exc}) — the xla adapter needs them to parse StableHLO; "
            f"upgrade jaxlib (`{_PIP_HINT}`)"
        ) from exc
    if not hasattr(_jax_mlir_ext, "register_dialects"):
        raise core.BackendError(
            "jaxlib's _jax_mlir_ext.register_dialects hook is missing — "
            f"API drift vs the validated jaxlib {VALIDATED_JAXLIB_VERSION}"
        )


def _acquire_cpu_client(xc: Any):
    """Create the embedded CPU PJRT client (see module docstring, step 1).

    Returns the ``xc.Client``. A failure raises ``core.BackendError`` —
    never a silent fallback.
    """
    try:
        return xc.make_cpu_client()
    except Exception as exc:
        raise core.BackendError(
            f"failed to acquire the XLA CPU PJRT client: {exc} — no CPU "
            "PJRT plugin is available in this jaxlib"
        ) from exc


def _make_mlir_context():
    """Build a jaxlib MLIR context with stablehlo/chlo/func/cf registered.

    The exact working recipe (see module docstring, step 2): core dialects
    via ``_jax_mlir_ext.register_dialects`` + ``load_all_available_dialects``,
    the separately-shipped stablehlo/chlo .so dialects via their own
    ``register_dialect``. Returns the ``jaxlib.mlir.ir.Context``.
    """
    from jaxlib.mlir import ir as jm_ir
    from jaxlib.mlir._mlir_libs import _jax_mlir_ext
    from jaxlib.mlir.dialects import chlo, stablehlo

    registry = jm_ir.DialectRegistry()
    _jax_mlir_ext.register_dialects(registry)
    context = jm_ir.Context()
    context.append_dialect_registry(registry)
    context.load_all_available_dialects()
    stablehlo.register_dialect(context)
    chlo.register_dialect(context)
    return context


def _parse_stablehlo_module(mlir_text: str):
    """Parse StableHLO MLIR text into a jaxlib ``ir.Module``.

    Parse errors (dialect missing / malformed text) raise
    ``core.BackendError`` naming the cause — never a silent retry.
    """
    from jaxlib.mlir import ir as jm_ir

    try:
        return jm_ir.Module.parse(mlir_text, context=_make_mlir_context())
    except Exception as exc:
        raise core.BackendError(
            f"failed to parse the StableHLO MLIR text with jaxlib's MLIR "
            f"bindings: {exc}"
        ) from exc


def _make_buffer_putter(client: Any):
    """Build a ``put(ndarray) -> ArrayImpl`` closure for this client.

    See module docstring, step 4: ``batched_device_put`` with a duck-typed
    aval and a ``SingleDeviceSharding`` patched with
    ``_to_xla_hlo_sharding``. ``enable_x64=True`` preserves float64/int64
    (the default x64-off state silently truncates them).
    """
    from jaxlib import xla_client as xc

    devices = client.devices()
    if not devices:
        raise core.BackendError(
            "the XLA CPU client reports no addressable devices — cannot "
            "stage input buffers"
        )
    device = devices[0]
    sharding = xc.SingleDeviceSharding(device, memory_kind=None)
    # jax's frontend SingleDeviceSharding subclass adds this method; the
    # pybind batched_device_put CALLS it, so attach it to the raw pybind
    # instance (attribute assignment on the pybind object is supported).
    sharding._to_xla_hlo_sharding = lambda _nd: xc.HloSharding.replicate()

    def put(array: Any):
        aval = _Aval(array.shape, array.dtype)
        return xc.batched_device_put(
            aval, sharding, [array], [device], True, enable_x64=True
        )

    return put
