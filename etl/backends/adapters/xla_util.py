"""PJRT C API driver for the XLA adapter (see ``xla.py``).

Everything in this module is imported from ``xla.py`` ONLY (it is not part
of any public surface). This module replaces the former native-binding
plumbing with a pure-stdlib ctypes driver over a user-provided PJRT C API
plugin (``_pjrt_c_api.py`` holds the vendored ABI translation; the header
provenance and the version-gate contract are documented there).

Plugin flow (the header ``xla/pjrt/c/pjrt_c_api.h`` is the contract):

1. **Discovery.** The plugin .so is NOT shipped with this package (there is
   no standalone XLA PJRT plugin distribution on PyPI). The user provides
   it; resolution order: (a) ``options["plugin_path"]`` (the backend
   compile-options dict, e.g. ``etl.compile(lowered, backend="xla",
   plugin_path="/path/to/pjrt_c_api_cpu_plugin.so")``), (b) the
   ``ETL_PJRT_PLUGIN`` environment variable, (c) a small set of well-known
   paths (``/usr/local/lib``, ``/usr/lib``, ``$HOME/.local/lib``, the
   current directory). None found -> ``core.BackendError`` with build
   instructions (``bazel build //xla/pjrt/c:pjrt_c_api_cpu_plugin`` from
   OpenXLA, or any XLA build exporting ``GetPjRtApi``).
2. **Load.** ``ctypes.CDLL(path)`` -> ``GetPjRtApi()`` -> version gate
   (``verify_api`` in ``_pjrt_c_api.py``: ABI major + ``struct_size``
   coverage) -> one-time ``PJRT_Plugin_Initialize``.
3. **Compile.** ``PJRT_Client_Create`` (empty options) ->
   ``PJRT_Client_Compile`` with ``PJRT_Program{code=mlir_text,
   format="mlir"}`` — the plugin accepts StableHLO MLIR text directly (the
   header documents ``"mlir"`` as "MLIR module bytecode (or string)"). No
   MLIR parsing happens in this process.
4. **Inputs.** ``PJRT_Client_BufferFromHostBuffer`` with dense
   row-major numpy arrays and ``kImmutableUntilTransferCompletes``
   semantics: the driver awaits the returned ``done_with_host_buffer``
   event before returning (and keeps the array alive on the buffer object
   until the buffer is destroyed — the header's lifetime contract).
5. **Execute.** ``PJRT_LoadedExecutable_Execute`` with one device, the
   staged buffers, and a zeroed ``PJRT_ExecuteOptions``.
6. **Outputs.** Per output buffer: ``PJRT_Buffer_ElementType`` +
   ``PJRT_Buffer_Dimensions`` -> allocate a writable numpy array ->
   ``PJRT_Buffer_ToHostBuffer`` -> await + destroy the completion event.
7. **Persistence.** ``PJRT_Executable_Serialize`` (bytes copied out; the
   serialized wrapper is released via its deleter) and
   ``PJRT_Executable_DeserializeAndLoad`` for load.

Error handling (binding): every call returns ``PJRT_Error*`` (NULL =
success); on error the driver extracts ``PJRT_Error_Message`` text,
destroys the error, and raises ``core.BackendError`` naming the failing
step. No silent fallbacks. Lifecycle hygiene: every handle the plugin
creates (errors, events, buffers, executables, clients) is destroyed via
the matching ``PJRT_*_Destroy`` entry point in ``try/finally`` or the
handle wrappers' ``close()``.

Import discipline (binding): stdlib + ``etl.core`` + the sibling
``_pjrt_c_api`` module only at top level; numpy is imported inside
function bodies.
"""

from __future__ import annotations

import ctypes
import os
import threading
from typing import Any

from etl import core

from . import _pjrt_c_api as pjrt

__all__ = [
    "_StaticShapeError",
    "_resolve_static_shape",
    "PjrtPlugin",
    "_find_plugin_path",
    "_load_plugin",
    "_DEFAULT_PLUGIN_PATHS",
]

#: Well-known locations probed when no explicit plugin path is configured.
_DEFAULT_PLUGIN_PATHS = (
    "/usr/local/lib/pjrt_c_api_cpu_plugin.so",
    "/usr/lib/pjrt_c_api_cpu_plugin.so",
    os.path.expanduser("~/.local/lib/pjrt_c_api_cpu_plugin.so"),
    "./pjrt_c_api_cpu_plugin.so",
)

#: Environment variable naming the plugin .so path.
_ENV_PLUGIN = "ETL_PJRT_PLUGIN"

_PLUGIN_MISSING_MESSAGE = (
    "the xla adapter requires a PJRT C API plugin library (.so exporting "
    "GetPjRtApi) and none was found. Provide one via the backend compile "
    "options (etl.compile(lowered, backend='xla', "
    "plugin_path='/path/to/pjrt_c_api_cpu_plugin.so')) or the "
    f"{_ENV_PLUGIN} environment variable; well-known paths searched: "
    + ", ".join(_DEFAULT_PLUGIN_PATHS)
    + ". Build one from OpenXLA with `bazel build "
    "//xla/pjrt/c:pjrt_c_api_cpu_plugin` (or use any XLA build exporting "
    "GetPjRtApi). There is no pip-installable plugin; this adapter never "
    "imports a native binding package."
)

#: dtype name (numpy) -> PJRT_Buffer_Type. Covers ALL 14 etl dtypes,
#: including complex64/complex128 (numpy's complex layout matches C
#: ``std::complex``: interleaved real/imag pairs).
_DTYPE_NAME_TO_PJRT = {
    "float16": pjrt.PJRT_Buffer_Type.F16,
    "float32": pjrt.PJRT_Buffer_Type.F32,
    "float64": pjrt.PJRT_Buffer_Type.F64,
    "int8": pjrt.PJRT_Buffer_Type.S8,
    "int16": pjrt.PJRT_Buffer_Type.S16,
    "int32": pjrt.PJRT_Buffer_Type.S32,
    "int64": pjrt.PJRT_Buffer_Type.S64,
    "uint8": pjrt.PJRT_Buffer_Type.U8,
    "uint16": pjrt.PJRT_Buffer_Type.U16,
    "uint32": pjrt.PJRT_Buffer_Type.U32,
    "uint64": pjrt.PJRT_Buffer_Type.U64,
    "bool": pjrt.PJRT_Buffer_Type.PRED,
    "complex64": pjrt.PJRT_Buffer_Type.C64,
    "complex128": pjrt.PJRT_Buffer_Type.C128,
}

_PJRT_TO_DTYPE_NAME = {v: k for k, v in _DTYPE_NAME_TO_PJRT.items()}


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


# ---------------------------------------------------------------------------
# Plugin discovery / loading
# ---------------------------------------------------------------------------

#: Paths already initialized (PJRT_Plugin_Initialize is one-time per plugin).
_initialized_plugins: set[str] = set()
_initialized_lock = threading.Lock()


def _find_plugin_path(options: dict | None = None) -> str:
    """Resolve the PJRT plugin .so path (discovery order in the docstring)."""
    if isinstance(options, dict) and options.get("plugin_path"):
        candidate = options["plugin_path"]
        source = "options['plugin_path']"
    else:
        candidate = os.environ.get(_ENV_PLUGIN) or ""
        source = f"the {_ENV_PLUGIN} environment variable"
    if candidate:
        if not isinstance(candidate, str) or not os.path.isfile(candidate):
            raise core.BackendError(
                f"{source} names {candidate!r}, which is not a file — the "
                "xla adapter needs a PJRT C API plugin library there"
            )
        return os.path.abspath(candidate)
    for candidate in _DEFAULT_PLUGIN_PATHS:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise core.BackendError(_PLUGIN_MISSING_MESSAGE)


def _load_plugin(options: dict | None = None) -> "PjrtPlugin":
    """Load (and cache per path) the PJRT plugin; raise ``core.BackendError``.

    Wraps ``_pjrt_c_api.load_plugin_library`` (``PJRTError`` ->
    ``core.BackendError`` with an actionable message) and performs the
    one-time ``PJRT_Plugin_Initialize``.
    """
    path = _find_plugin_path(options)
    try:
        library, api = pjrt.load_plugin_library(path)
    except pjrt.PJRTError as exc:
        raise core.BackendError(
            f"the PJRT plugin at {path!r} failed the ABI gate: {exc} — "
            f"point {_ENV_PLUGIN} or options['plugin_path'] at a plugin "
            "built from the compatible PJRT C API header"
        ) from exc
    plugin = PjrtPlugin(path, library, api)
    with _initialized_lock:
        if path not in _initialized_plugins:
            plugin.initialize()
            _initialized_plugins.add(path)
    return plugin


# ---------------------------------------------------------------------------
# Handle wrappers (lifecycle hygiene: every plugin-created object is
# destroyed via the matching PJRT_*_Destroy entry point).
# ---------------------------------------------------------------------------


class _Handle:
    """Base wrapper for an opaque plugin-owned handle."""

    __slots__ = ("plugin", "ptr", "_closed")

    def __init__(self, plugin: "PjrtPlugin", ptr: int) -> None:
        self.plugin = plugin
        self.ptr = ptr
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed


class _Client(_Handle):
    """A live ``PJRT_Client`` (destroyed via ``PJRT_Client_Destroy``)."""

    __slots__ = ("_devices",)

    def __init__(self, plugin: "PjrtPlugin", ptr: int) -> None:
        super().__init__(plugin, ptr)
        self._devices: tuple[int, ...] | None = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.plugin._destroy(
            "PJRT_Client_Destroy", pjrt.PJRT_Client_Destroy_Args, self.ptr
        )

    def addressable_devices(self) -> tuple[int, ...]:
        """The client's addressable ``PJRT_Device*`` pointers (cached)."""
        if self._devices is None:
            args = pjrt.PJRT_Client_AddressableDevices_Args(
                struct_size=pjrt.sizeof(pjrt.PJRT_Client_AddressableDevices_Args),
                client=self.ptr,
            )
            self.plugin._check(
                self.plugin.api.PJRT_Client_AddressableDevices(ctypes.byref(args)),
                "PJRT_Client_AddressableDevices",
            )
            if args.num_addressable_devices == 0:
                raise core.BackendError(
                    "the PJRT plugin client reports no addressable devices — "
                    "cannot stage input buffers or execute"
                )
            self._devices = tuple(
                args.addressable_devices[i]
                for i in range(args.num_addressable_devices)
            )
        return self._devices

    def platform_info(self) -> tuple[str, str]:
        """``(platform_name, platform_version)`` strings (copied out)."""
        name = self._string_out(
            "PJRT_Client_PlatformName",
            pjrt.PJRT_Client_PlatformName_Args,
            "platform_name",
            "platform_name_size",
        )
        version = self._string_out(
            "PJRT_Client_PlatformVersion",
            pjrt.PJRT_Client_PlatformVersion_Args,
            "platform_version",
            "platform_version_size",
        )
        return name, version

    def _string_out(
        self, api_name: str, args_cls: type, ptr_field: str, size_field: str
    ) -> str:
        """Copy out a client-owned string (valid only during the call)."""
        args = args_cls(
            struct_size=pjrt.sizeof(args_cls), client=self.ptr
        )
        self.plugin._check(
            getattr(self.plugin.api, api_name)(ctypes.byref(args)), api_name
        )
        ptr = getattr(args, ptr_field)
        if not ptr:
            return ""
        return ctypes.string_at(ptr, getattr(args, size_field)).decode(
            "utf-8", "replace"
        )

    def compile(
        self, mlir_text: str, compile_options: bytes | None = None
    ) -> "_LoadedExecutable":
        """Compile StableHLO MLIR text (``PJRT_Client_Compile``).

        ``compile_options``: an optional SERIALIZED ``xla.CompileOptionsProto``
        (raw bytes) handed to the plugin through
        ``PJRT_Client_Compile_Args.compile_options``/``compile_options_size``
        — the plugin validates the payload (an invalid proto surfaces as a
        plugin error). Real plugins (jax_cuda12_pjrt 0.4.38's
        xla_cuda_plugin.so et al.) CHECK-crash on NULL — callers MUST pass
        the probe-validated single-replica proto bytes (the iree adapter's
        default ``xla_compile_options``) unless they know the plugin accepts
        NULL."""
        code_buf = ctypes.create_string_buffer(mlir_text.encode("utf-8"))
        fmt_buf = ctypes.create_string_buffer(b"mlir")
        program = pjrt.PJRT_Program(
            struct_size=pjrt.sizeof(pjrt.PJRT_Program),
            code=ctypes.cast(code_buf, ctypes.c_void_p),
            code_size=len(mlir_text),
            format=ctypes.cast(fmt_buf, ctypes.c_void_p),
            format_size=4,  # len("mlir")
        )
        compile_opts_buf = (
            ctypes.create_string_buffer(compile_options)
            if compile_options is not None
            else None
        )
        args = pjrt.PJRT_Client_Compile_Args(
            struct_size=pjrt.sizeof(pjrt.PJRT_Client_Compile_Args),
            client=self.ptr,
            program=ctypes.pointer(program),
            compile_options=(
                ctypes.cast(compile_opts_buf, ctypes.c_void_p)
                if compile_opts_buf is not None
                else None
            ),
            compile_options_size=(
                len(compile_options) if compile_options is not None else 0
            ),
            executable=None,
        )
        try:
            self.plugin._check(
                self.plugin.api.PJRT_Client_Compile(ctypes.byref(args)),
                "PJRT_Client_Compile",
            )
        except core.BackendError as exc:
            if "ptxas" in str(exc):
                raise core.BackendError(
                    f"{exc} — ensure ptxas (from the CUDA toolkit) is on "
                    "PATH (the XLA CUDA plugin invokes it at compile time)"
                ) from None
            raise
        return _LoadedExecutable(self.plugin, args.executable)

    def deserialize(self, serialized: bytes) -> "_LoadedExecutable":
        """Load a serialized executable (``PJRT_Executable_DeserializeAndLoad``)."""
        buf = ctypes.create_string_buffer(serialized)
        args = pjrt.PJRT_Executable_DeserializeAndLoad_Args(
            struct_size=pjrt.sizeof(pjrt.PJRT_Executable_DeserializeAndLoad_Args),
            client=self.ptr,
            serialized_executable=ctypes.cast(buf, ctypes.c_void_p),
            serialized_executable_size=len(serialized),
            loaded_executable=None,
            overridden_serialized_compile_options=None,
            overridden_serialized_compile_options_size=0,
            load_options=None,
        )
        self.plugin._check(
            self.plugin.api.PJRT_Executable_DeserializeAndLoad(ctypes.byref(args)),
            "PJRT_Executable_DeserializeAndLoad",
        )
        return _LoadedExecutable(self.plugin, args.loaded_executable)

    def buffer_from_host(self, array: Any) -> "_Buffer":
        """Stage a numpy array as a device buffer (dense, row-major)."""
        import numpy as np

        arr = np.ascontiguousarray(array)
        dtype_name = np.dtype(arr.dtype).name
        try:
            buffer_type = _DTYPE_NAME_TO_PJRT[dtype_name]
        except KeyError:
            raise core.DTypeError(
                f"the xla adapter cannot stage an input of dtype "
                f"{dtype_name!r} — supported: "
                + ", ".join(sorted(_DTYPE_NAME_TO_PJRT))
            ) from None
        dims = (ctypes.c_int64 * max(1, len(arr.shape)))(*arr.shape)
        data_ptr = (
            ctypes.cast(arr.ctypes.data, ctypes.c_void_p)
            if arr.size
            else ctypes.cast((ctypes.c_byte * 1)(), ctypes.c_void_p)
        )
        device = self.addressable_devices()[0]
        args = pjrt.PJRT_Client_BufferFromHostBuffer_Args(
            struct_size=pjrt.sizeof(pjrt.PJRT_Client_BufferFromHostBuffer_Args),
            client=self.ptr,
            data=data_ptr,
            type=int(buffer_type),
            dims=dims,
            num_dims=len(arr.shape),
            byte_strides=None,
            num_byte_strides=0,
            host_buffer_semantics=int(
                pjrt.PJRT_HostBufferSemantics.kImmutableUntilTransferCompletes
            ),
            device=device,
            memory=None,
            device_layout=None,
            done_with_host_buffer=None,
            buffer=None,
        )
        self.plugin._check(
            self.plugin.api.PJRT_Client_BufferFromHostBuffer(ctypes.byref(args)),
            "PJRT_Client_BufferFromHostBuffer",
        )
        event = args.done_with_host_buffer
        try:
            if event:
                self.plugin.await_event(event)
        finally:
            if event:
                self.plugin.destroy_event(event)
        # Keep the host array alive with the buffer: some plugins alias host
        # memory beyond the documented transfer-complete point.
        return _Buffer(self.plugin, args.buffer, keepalive=(arr, dims))


class _Executable(_Handle):
    """An unloaded ``PJRT_Executable`` (destroyed via ``PJRT_Executable_Destroy``)."""

    __slots__ = ()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.plugin._destroy(
            "PJRT_Executable_Destroy", pjrt.PJRT_Executable_Destroy_Args, self.ptr
        )


class _LoadedExecutable(_Handle):
    """A loaded executable + its unloaded twin for serialize/num_outputs.

    ``PJRT_LoadedExecutable_GetExecutable`` hands out an unloaded
    ``PJRT_Executable`` the caller must destroy separately; the driver
    keeps both and destroys the inner one first.
    """

    __slots__ = ("_inner",)

    def __init__(self, plugin: "PjrtPlugin", ptr: int) -> None:
        super().__init__(plugin, ptr)
        args = pjrt.PJRT_LoadedExecutable_GetExecutable_Args(
            struct_size=pjrt.sizeof(pjrt.PJRT_LoadedExecutable_GetExecutable_Args),
            loaded_executable=ptr,
            executable=None,
        )
        plugin._check(
            plugin.api.PJRT_LoadedExecutable_GetExecutable(ctypes.byref(args)),
            "PJRT_LoadedExecutable_GetExecutable",
        )
        self._inner = _Executable(plugin, args.executable)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._inner is not None:
            self._inner.close()
            self._inner = None
        self.plugin._destroy(
            "PJRT_LoadedExecutable_Destroy",
            pjrt.PJRT_LoadedExecutable_Destroy_Args,
            self.ptr,
        )

    def num_outputs(self) -> int:
        """Output count per device (``PJRT_Executable_NumOutputs``)."""
        args = pjrt.PJRT_Executable_NumOutputs_Args(
            struct_size=pjrt.sizeof(pjrt.PJRT_Executable_NumOutputs_Args),
            executable=self._inner.ptr,
            num_outputs=0,
        )
        self.plugin._check(
            self.plugin.api.PJRT_Executable_NumOutputs(ctypes.byref(args)),
            "PJRT_Executable_NumOutputs",
        )
        return args.num_outputs

    def serialize(self) -> bytes:
        """Serialize to bytes (``PJRT_Executable_Serialize``)."""
        args = pjrt.PJRT_Executable_Serialize_Args(
            struct_size=pjrt.sizeof(pjrt.PJRT_Executable_Serialize_Args),
            executable=self._inner.ptr,
            serialized_bytes=None,
            serialized_bytes_size=0,
            serialized_executable=None,
            # serialized_executable_deleter is an out-only CFUNCTYPE field;
            # it stays zero-initialized (ctypes rejects None initializers).
        )
        self.plugin._check(
            self.plugin.api.PJRT_Executable_Serialize(ctypes.byref(args)),
            "PJRT_Executable_Serialize",
        )
        try:
            return ctypes.string_at(args.serialized_bytes, args.serialized_bytes_size)
        finally:
            deleter = args.serialized_executable_deleter
            if deleter:
                deleter(args.serialized_executable)

    def execute(self, buffers: list["_Buffer"]) -> list["_Buffer"]:
        """Execute on the first addressable device; return output buffers.

        The caller owns the returned ``_Buffer`` objects (destroy via
        ``close()``); input buffers stay owned by the caller and may be
        destroyed once execute returns.
        """
        num_args = len(buffers)
        device_ptrs = (ctypes.c_void_p * max(1, num_args))(
            *[b.ptr for b in buffers]
        )
        argument_lists = (ctypes.c_void_p * 1)(
            ctypes.cast(device_ptrs, ctypes.c_void_p)
        )
        num_outputs = self.num_outputs()
        out_inner = (ctypes.c_void_p * max(1, num_outputs))()
        output_lists = (ctypes.c_void_p * 1)(
            ctypes.cast(out_inner, ctypes.c_void_p)
        )
        options = pjrt.PJRT_ExecuteOptions(
            struct_size=pjrt.sizeof(pjrt.PJRT_ExecuteOptions)
        )
        args = pjrt.PJRT_LoadedExecutable_Execute_Args(
            struct_size=pjrt.sizeof(pjrt.PJRT_LoadedExecutable_Execute_Args),
            executable=self.ptr,
            options=ctypes.pointer(options),
            argument_lists=ctypes.cast(
                argument_lists, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
            ),
            num_devices=1,
            num_args=num_args,
            output_lists=ctypes.cast(
                output_lists, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
            ),
            device_complete_events=None,
            execute_device=None,
        )
        self.plugin._check(
            self.plugin.api.PJRT_LoadedExecutable_Execute(ctypes.byref(args)),
            "PJRT_LoadedExecutable_Execute",
        )
        return [_Buffer(self.plugin, out_inner[i]) for i in range(num_outputs)]


class _Buffer(_Handle):
    """A device buffer (destroyed via ``PJRT_Buffer_Destroy``)."""

    __slots__ = ("_keepalive",)

    def __init__(self, plugin: "PjrtPlugin", ptr: int, keepalive: tuple = ()) -> None:
        super().__init__(plugin, ptr)
        self._keepalive = keepalive  # host memory that must outlive the buffer

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._keepalive = ()
        self.plugin._destroy("PJRT_Buffer_Destroy", pjrt.PJRT_Buffer_Destroy_Args, self.ptr)

    def to_host(self) -> Any:
        """Copy the buffer into a fresh row-major numpy array."""
        import numpy as np

        type_args = pjrt.PJRT_Buffer_ElementType_Args(
            struct_size=pjrt.sizeof(pjrt.PJRT_Buffer_ElementType_Args),
            buffer=self.ptr,
            type=0,
        )
        self.plugin._check(
            self.plugin.api.PJRT_Buffer_ElementType(ctypes.byref(type_args)),
            "PJRT_Buffer_ElementType",
        )
        try:
            dtype_name = _PJRT_TO_DTYPE_NAME[pjrt.PJRT_Buffer_Type(type_args.type)]
        except KeyError:
            raise core.BackendError(
                f"the PJRT plugin produced an output buffer of element type "
                f"{type_args.type}, which has no numpy equivalent — the xla "
                "adapter supports "
                + ", ".join(sorted(_DTYPE_NAME_TO_PJRT))
            ) from None
        dim_args = pjrt.PJRT_Buffer_Dimensions_Args(
            struct_size=pjrt.sizeof(pjrt.PJRT_Buffer_Dimensions_Args),
            buffer=self.ptr,
            dims=None,
            num_dims=0,
        )
        self.plugin._check(
            self.plugin.api.PJRT_Buffer_Dimensions(ctypes.byref(dim_args)),
            "PJRT_Buffer_Dimensions",
        )
        shape = tuple(dim_args.dims[i] for i in range(dim_args.num_dims))
        array = np.empty(shape, dtype=dtype_name)
        dst = (
            ctypes.cast(array.ctypes.data, ctypes.c_void_p)
            if array.size
            else ctypes.cast((ctypes.c_byte * 1)(), ctypes.c_void_p)
        )
        host_args = pjrt.PJRT_Buffer_ToHostBuffer_Args(
            struct_size=pjrt.sizeof(pjrt.PJRT_Buffer_ToHostBuffer_Args),
            src=self.ptr,
            host_layout=None,
            dst=dst,
            dst_size=array.nbytes,
            event=None,
        )
        self.plugin._check(
            self.plugin.api.PJRT_Buffer_ToHostBuffer(ctypes.byref(host_args)),
            "PJRT_Buffer_ToHostBuffer",
        )
        event = host_args.event
        try:
            if event:
                self.plugin.await_event(event)
        finally:
            if event:
                self.plugin.destroy_event(event)
        return array


# ---------------------------------------------------------------------------
# The plugin driver
# ---------------------------------------------------------------------------


class PjrtPlugin:
    """A loaded PJRT C API plugin (ctypes driver; see module docstring).

    Holds the ``ctypes.CDLL`` handle (kept alive for the driver's
    lifetime) and the validated ``PJRT_Api`` function table. Instances are
    cached per path by ``_load_plugin``.
    """

    __slots__ = ("path", "_library", "api")

    def __init__(self, path: str, library: ctypes.CDLL, api: pjrt.PJRT_Api) -> None:
        self.path = path
        self._library = library
        self.api = api

    # ------------------------------------------------------------- plumbing

    def initialize(self) -> None:
        """One-time ``PJRT_Plugin_Initialize`` (see header: must precede all)."""
        if not self.api.PJRT_Plugin_Initialize:
            raise core.BackendError(
                f"the PJRT plugin at {self.path!r} does not implement the "
                "mandatory PJRT_Plugin_Initialize entry point"
            )
        args = pjrt.PJRT_Plugin_Initialize_Args(
            struct_size=pjrt.sizeof(pjrt.PJRT_Plugin_Initialize_Args)
        )
        self._check(
            self.api.PJRT_Plugin_Initialize(ctypes.byref(args)),
            "PJRT_Plugin_Initialize",
        )

    def _check(self, err: int, step: str) -> None:
        """Raise ``core.BackendError`` (naming the step) when ``err`` set."""
        if not err:
            return
        message = self._error_message(err)
        self._destroy_error(err)
        raise core.BackendError(
            f"the xla PJRT plugin failed at {step}: {message}"
        )

    def _error_message(self, err: int) -> str:
        args = pjrt.PJRT_Error_Message_Args(
            struct_size=pjrt.sizeof(pjrt.PJRT_Error_Message_Args),
            error=err,
            message=None,
            message_size=0,
        )
        try:
            self.api.PJRT_Error_Message(ctypes.byref(args))
        except Exception:  # pragma: no cover - defensive; message must not fail
            return "<unreadable plugin error>"
        if args.message:
            return ctypes.string_at(args.message, args.message_size).decode(
                "utf-8", "replace"
            )
        return "<empty plugin error>"

    def _destroy_error(self, err: int) -> None:
        if not err:
            return
        args = pjrt.PJRT_Error_Destroy_Args(
            struct_size=pjrt.sizeof(pjrt.PJRT_Error_Destroy_Args), error=err
        )
        self.api.PJRT_Error_Destroy(ctypes.byref(args))

    def _destroy(self, api_name: str, args_cls: type, ptr: int) -> None:
        """Destroy a handle through the given PJRT_*_Destroy entry point."""
        if not ptr:
            return
        fn = getattr(self.api, api_name)
        args = args_cls(struct_size=pjrt.sizeof(args_cls), extension_start=None)
        setattr(args, args_cls._fields_[-1][0], ptr)
        self._check(fn(ctypes.byref(args)), api_name)

    # ------------------------------------------------------------ lifecycle

    def create_client(self) -> _Client:
        """Create a client with empty options (``PJRT_Client_Create``)."""
        args = pjrt.PJRT_Client_Create_Args(
            struct_size=pjrt.sizeof(pjrt.PJRT_Client_Create_Args),
            create_options=None,
            num_options=0,
            kv_get_callback=None,
            kv_get_user_arg=None,
            kv_put_callback=None,
            kv_put_user_arg=None,
            kv_try_get_callback=None,
            kv_try_get_user_arg=None,
        )
        self._check(self.api.PJRT_Client_Create(ctypes.byref(args)), "PJRT_Client_Create")
        return _Client(self, args.client)

    def await_event(self, event: int) -> None:
        """Block until an event is ready (``PJRT_Event_Await``)."""
        args = pjrt.PJRT_Event_Await_Args(
            struct_size=pjrt.sizeof(pjrt.PJRT_Event_Await_Args), event=event
        )
        self._check(self.api.PJRT_Event_Await(ctypes.byref(args)), "PJRT_Event_Await")

    def destroy_event(self, event: int) -> None:
        if not event:
            return
        args = pjrt.PJRT_Event_Destroy_Args(
            struct_size=pjrt.sizeof(pjrt.PJRT_Event_Destroy_Args), event=event
        )
        self._check(self.api.PJRT_Event_Destroy(ctypes.byref(args)), "PJRT_Event_Destroy")
