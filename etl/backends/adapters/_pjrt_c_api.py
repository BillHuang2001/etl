"""Vendored ctypes translation of the PJRT C API header (OpenXLA).

This module is a hand-written ctypes translation of the canonical PJRT C
API header (the ABI contract between PJRT plugins and their drivers). It
is the ONLY place in the XLA adapter that knows the plugin ABI; the driver
lives in ``xla_util.py``.

Header provenance (recorded per the adapter's serialization contract):

- URL: https://raw.githubusercontent.com/openxla/xla/main/xla/pjrt/c/pjrt_c_api.h
- Commit: 70fe66213b73c5953d92eb25d2606bd6004d47c3 (openxla/xla main,
  retrieved 2026-07-15 via the GitHub API; the commit that last touched
  the header).
- File size at retrieval: 127045 bytes.
- License: Apache-2.0, Copyright 2022 The OpenXLA Authors. This module
  re-expresses the ABI shapes (struct layouts, enum values, function
  signatures) from that header; it does NOT vendor the header text.

Faithfulness rules (binding):

1. ``PJRT_Api`` field ORDER is exactly the header's order — field offsets
   matter (the plugin binary was compiled against the real header). Every
   ``PJRT_Api`` field is a function pointer except the leading
   ``struct_size`` / ``extension_start`` / ``pjrt_api_version`` fields.
2. Functions the driver calls get full ctypes prototypes (argtypes +
   restype via ``CFUNCTYPE``); every other field is a ``c_void_p``
   placeholder (all pointers are the same size, so layout is safe).
3. Argument structs are declared FAITHFULLY through their last field, and
   the driver sets ``struct_size = ctypes.sizeof(...)`` — per the header's
   versioning contract, implementations use ``struct_size`` to detect how
   many struct fields the caller is aware of. Declaring the full struct is
   therefore always forward-compatible.
4. Version gate: ``PJRT_API_MAJOR`` must match (ABI family); the plugin's
   ``PJRT_Api.struct_size`` must cover the last function field this driver
   reads (``PJRT_Buffer_ToHostBuffer``) — see ``verify_api``.

This module imports ONLY the standard library (``ctypes``) — it must stay
free of any third-party or etl import.
"""

from __future__ import annotations

import ctypes
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    byref,
    c_bool,
    c_int,
    c_int64,
    c_size_t,
    c_void_p,
    cast,
)
from enum import IntEnum

__all__ = [
    "PJRTError",
    "PJRT_Buffer_Type",
    "PJRT_HostBufferSemantics",
    "PJRT_Api_Version",
    "PJRT_Program",
    "PJRT_ExecuteOptions",
    "PJRT_Api",
    "PJRT_API_MAJOR",
    "PJRT_API_MINOR",
    "HEADER_URL",
    "HEADER_COMMIT",
    "HEADER_RETRIEVED",
    "HEADER_SIZE_BYTES",
    "sizeof",
    "verify_api",
    "load_plugin_library",
]

#: The exact header revision this translation was generated from.
HEADER_URL = "https://raw.githubusercontent.com/openxla/xla/main/xla/pjrt/c/pjrt_c_api.h"
HEADER_COMMIT = "70fe66213b73c5953d92eb25d2606bd6004d47c3"
HEADER_RETRIEVED = "2026-07-15"
HEADER_SIZE_BYTES = 127045

#: Version constants from the header (the plugin records the version IT was
#: compiled with in ``PJRT_Api.pjrt_api_version``).
PJRT_API_MAJOR = 0
PJRT_API_MINOR = 114


class PJRTError(Exception):
    """Internal error of the PJRT C API bindings (stdlib-only module).

    Raised for ABI/version-gate failures and plugin-load failures; the
    driver in ``xla_util.py`` wraps it into ``core.BackendError``.
    """


# --------------------------------------------------------------------------
# Opaque handle types. All handles are pointers to plugin-owned objects;
# the driver treats them as ``c_void_p`` and destroys them via the
# corresponding PJRT_*_Destroy entry point.
# --------------------------------------------------------------------------

_PJRT_Error = c_void_p
_PJRT_Event = c_void_p
_PJRT_Client = c_void_p
_PJRT_Device = c_void_p
_PJRT_Memory = c_void_p
_PJRT_Executable = c_void_p
_PJRT_LoadedExecutable = c_void_p
_PJRT_Buffer = c_void_p
_PJRT_SerializedExecutable = c_void_p


class PJRT_Buffer_Type(IntEnum):
    """Element types of PJRT buffers (``PJRT_Buffer_Type`` in the header)."""

    INVALID = 0
    PRED = 1
    S8 = 2
    S16 = 3
    S32 = 4
    S64 = 5
    U8 = 6
    U16 = 7
    U32 = 8
    U64 = 9
    F16 = 10
    F32 = 11
    F64 = 12
    BF16 = 13
    C64 = 14
    C128 = 15
    F8E5M2 = 16
    F8E4M3FN = 17
    F8E4M3B11FNUZ = 18
    F8E5M2FNUZ = 19
    F8E4M3FNUZ = 20
    S4 = 21
    U4 = 22
    TOKEN = 23
    S2 = 24
    U2 = 25
    F8E4M3 = 26
    F8E3M4 = 27
    F8E8M0FNU = 28
    F4E2M1FN = 29
    S1 = 30
    U1 = 31
    F6E2M3FN = 32
    F6E3M2FN = 33


class PJRT_HostBufferSemantics(IntEnum):
    """Lifetime contract for host buffers (header, ``PJRT_HostBufferSemantics``)."""

    kImmutableOnlyDuringCall = 0
    kImmutableUntilTransferCompletes = 1
    kImmutableZeroCopy = 2
    kMutableZeroCopy = 3


class PJRT_Api_Version(Structure):
    """Version record embedded BY VALUE in ``PJRT_Api``."""

    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("major_version", c_int),
        ("minor_version", c_int),
    ]


class PJRT_Program(Structure):
    """The program handed to ``PJRT_Client_Compile``.

    Header contract: ``format`` is one of ``"hlo"``, ``"hlo_with_config"``,
    ``"mlir"`` (MLIR module bytecode or text). ``code``/``format`` are owned
    by the caller and only need to stay alive for the duration of the call;
    sizes exclude the NUL terminator.
    """

    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("code", c_void_p),
        ("code_size", c_size_t),
        ("format", c_void_p),
        ("format_size", c_size_t),
    ]


class PJRT_ExecuteOptions(Structure):
    """Execution options for ``PJRT_LoadedExecutable_Execute``.

    Declared faithfully through ``num_hlo_output_callbacks`` (the header's
    last field); the driver zeroes it and sets ``struct_size``.
    """

    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("send_callbacks", POINTER(c_void_p)),
        ("recv_callbacks", POINTER(c_void_p)),
        ("num_send_ops", c_size_t),
        ("num_recv_ops", c_size_t),
        ("launch_id", c_int),
        ("non_donatable_input_indices", POINTER(c_int64)),
        ("num_non_donatable_input_indices", c_size_t),
        ("context", c_void_p),
        ("call_location", c_void_p),
        ("num_tasks", c_size_t),
        ("task_ids", POINTER(c_int)),
        ("incarnation_ids", POINTER(c_int64)),
        ("multi_slice_config", c_void_p),
        ("use_major_to_minor_data_layout_for_callbacks", c_bool),
        ("hlo_output_callbacks", c_void_p),
        ("num_hlo_output_callbacks", c_size_t),
    ]


# --------------------------------------------------------------------------
# Argument structs for the calls the driver makes. Declared faithfully in
# field order through the header's last field; ``struct_size`` is set by the
# driver to ``ctypes.sizeof(...)`` at every call site.
# --------------------------------------------------------------------------


class PJRT_Error_Destroy_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("error", c_void_p),
    ]


class PJRT_Error_Message_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("error", c_void_p),
        ("message", c_void_p),  # out: const char*, lifetime of error
        ("message_size", c_size_t),  # out
    ]


class PJRT_Plugin_Initialize_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
    ]


class PJRT_Event_Destroy_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("event", c_void_p),
    ]


class PJRT_Event_Await_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("event", c_void_p),
    ]


class PJRT_Client_Create_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("create_options", c_void_p),  # const PJRT_NamedValue* (NULL = none)
        ("num_options", c_size_t),
        ("kv_get_callback", c_void_p),  # NULL = single-process
        ("kv_get_user_arg", c_void_p),
        ("kv_put_callback", c_void_p),
        ("kv_put_user_arg", c_void_p),
        ("client", c_void_p),  # out: PJRT_Client*
        ("kv_try_get_callback", c_void_p),
        ("kv_try_get_user_arg", c_void_p),
    ]


class PJRT_Client_Destroy_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("client", c_void_p),
    ]


class PJRT_Client_PlatformName_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("client", c_void_p),
        ("platform_name", c_void_p),  # out, lifetime of client
        ("platform_name_size", c_size_t),  # out
    ]


class PJRT_Client_PlatformVersion_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("client", c_void_p),
        ("platform_version", c_void_p),  # out, lifetime of client
        ("platform_version_size", c_size_t),  # out
    ]


class PJRT_Client_Devices_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("client", c_void_p),
        ("devices", POINTER(c_void_p)),  # out: PJRT_Device* const*
        ("num_devices", c_size_t),  # out
    ]


class PJRT_Client_AddressableDevices_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("client", c_void_p),
        ("addressable_devices", POINTER(c_void_p)),  # out
        ("num_addressable_devices", c_size_t),  # out
    ]


class PJRT_Client_LookupDevice_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("client", c_void_p),
        ("id", c_int),
        ("device", c_void_p),  # out, lifetime of client
    ]


class PJRT_Client_Compile_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("client", c_void_p),
        ("program", POINTER(PJRT_Program)),  # const, caller-owned
        ("compile_options", c_void_p),  # serialized CompileOptionsProto
        ("compile_options_size", c_size_t),
        ("executable", c_void_p),  # out: PJRT_LoadedExecutable*
    ]


class PJRT_Client_BufferFromHostBuffer_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("client", c_void_p),
        ("data", c_void_p),  # const void*, host buffer
        ("type", c_int),  # PJRT_Buffer_Type
        ("dims", POINTER(c_int64)),  # const int64_t*
        ("num_dims", c_size_t),
        ("byte_strides", POINTER(c_int64)),  # NULL/empty = dense row-major
        ("num_byte_strides", c_size_t),
        ("host_buffer_semantics", c_int),  # PJRT_HostBufferSemantics
        ("device", c_void_p),
        ("memory", c_void_p),  # NULL = copy to device
        ("device_layout", c_void_p),  # NULL = dense row-major
        ("done_with_host_buffer", c_void_p),  # out: PJRT_Event* (destroy)
        ("buffer", c_void_p),  # out: PJRT_Buffer* (destroy)
    ]


class PJRT_Executable_Destroy_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("executable", c_void_p),
    ]


class PJRT_LoadedExecutable_Destroy_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("executable", c_void_p),
    ]


class PJRT_LoadedExecutable_GetExecutable_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("loaded_executable", c_void_p),
        ("executable", c_void_p),  # out: PJRT_Executable* (destroy)
    ]


class PJRT_Executable_NumOutputs_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("executable", c_void_p),
        ("num_outputs", c_size_t),  # out
    ]


_PJRT_SerializedExecutable_Deleter = CFUNCTYPE(None, c_void_p)


class PJRT_Executable_Serialize_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("executable", c_void_p),  # const PJRT_Executable*
        ("serialized_bytes", c_void_p),  # out
        ("serialized_bytes_size", c_size_t),  # out
        ("serialized_executable", c_void_p),  # out: backs serialized_bytes
        ("serialized_executable_deleter", _PJRT_SerializedExecutable_Deleter),  # out
    ]


class PJRT_Executable_DeserializeAndLoad_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("client", c_void_p),
        ("serialized_executable", c_void_p),  # const char*
        ("serialized_executable_size", c_size_t),
        ("loaded_executable", c_void_p),  # out: PJRT_LoadedExecutable*
        ("overridden_serialized_compile_options", c_void_p),  # NULL = keep
        ("overridden_serialized_compile_options_size", c_size_t),
        ("load_options", c_void_p),  # PJRT_LoadOptions* (NULL)
    ]


class PJRT_LoadedExecutable_Execute_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("executable", c_void_p),
        ("options", POINTER(PJRT_ExecuteOptions)),  # caller-owned, per call
        ("argument_lists", POINTER(POINTER(c_void_p))),  # [num_devices][num_args]
        ("num_devices", c_size_t),
        ("num_args", c_size_t),
        ("output_lists", POINTER(POINTER(c_void_p))),  # in/out, caller-allocated
        ("device_complete_events", POINTER(c_void_p)),  # in/out, NULL ok
        ("execute_device", c_void_p),  # NULL = compile-time device
    ]


class PJRT_Buffer_Destroy_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("buffer", c_void_p),
    ]


class PJRT_Buffer_ElementType_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("buffer", c_void_p),
        ("type", c_int),  # out: PJRT_Buffer_Type
    ]


class PJRT_Buffer_Dimensions_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("buffer", c_void_p),
        ("dims", POINTER(c_int64)),  # out, lifetime of buffer
        ("num_dims", c_size_t),  # out
    ]


class PJRT_Buffer_ToHostBuffer_Args(Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("src", c_void_p),
        ("host_layout", c_void_p),  # NULL = buffer's layout
        ("dst", c_void_p),  # in/out; NULL queries required dst_size
        ("dst_size", c_size_t),  # in/out
        ("event", c_void_p),  # out: PJRT_Event* (destroy)
    ]


# --------------------------------------------------------------------------
# Full prototypes for every PJRT_Api entry point the driver calls. All
# PJRT_Api methods return PJRT_Error* (NULL = success); PJRT_Error_Destroy /
# PJRT_Error_Message return void.
# --------------------------------------------------------------------------

PJRT_Error_Destroy_t = CFUNCTYPE(None, POINTER(PJRT_Error_Destroy_Args))
PJRT_Error_Message_t = CFUNCTYPE(None, POINTER(PJRT_Error_Message_Args))
PJRT_Plugin_Initialize_t = CFUNCTYPE(c_void_p, POINTER(PJRT_Plugin_Initialize_Args))
PJRT_Event_Destroy_t = CFUNCTYPE(c_void_p, POINTER(PJRT_Event_Destroy_Args))
PJRT_Event_Await_t = CFUNCTYPE(c_void_p, POINTER(PJRT_Event_Await_Args))
PJRT_Client_Create_t = CFUNCTYPE(c_void_p, POINTER(PJRT_Client_Create_Args))
PJRT_Client_Destroy_t = CFUNCTYPE(c_void_p, POINTER(PJRT_Client_Destroy_Args))
PJRT_Client_PlatformName_t = CFUNCTYPE(c_void_p, POINTER(PJRT_Client_PlatformName_Args))
PJRT_Client_PlatformVersion_t = CFUNCTYPE(
    c_void_p, POINTER(PJRT_Client_PlatformVersion_Args)
)
PJRT_Client_Devices_t = CFUNCTYPE(c_void_p, POINTER(PJRT_Client_Devices_Args))
PJRT_Client_AddressableDevices_t = CFUNCTYPE(
    c_void_p, POINTER(PJRT_Client_AddressableDevices_Args)
)
PJRT_Client_LookupDevice_t = CFUNCTYPE(c_void_p, POINTER(PJRT_Client_LookupDevice_Args))
PJRT_Client_Compile_t = CFUNCTYPE(c_void_p, POINTER(PJRT_Client_Compile_Args))
PJRT_Client_BufferFromHostBuffer_t = CFUNCTYPE(
    c_void_p, POINTER(PJRT_Client_BufferFromHostBuffer_Args)
)
PJRT_Executable_Destroy_t = CFUNCTYPE(c_void_p, POINTER(PJRT_Executable_Destroy_Args))
PJRT_LoadedExecutable_Destroy_t = CFUNCTYPE(
    c_void_p, POINTER(PJRT_LoadedExecutable_Destroy_Args)
)
PJRT_LoadedExecutable_GetExecutable_t = CFUNCTYPE(
    c_void_p, POINTER(PJRT_LoadedExecutable_GetExecutable_Args)
)
PJRT_Executable_NumOutputs_t = CFUNCTYPE(
    c_void_p, POINTER(PJRT_Executable_NumOutputs_Args)
)
PJRT_Executable_Serialize_t = CFUNCTYPE(
    c_void_p, POINTER(PJRT_Executable_Serialize_Args)
)
PJRT_Executable_DeserializeAndLoad_t = CFUNCTYPE(
    c_void_p, POINTER(PJRT_Executable_DeserializeAndLoad_Args)
)
PJRT_LoadedExecutable_Execute_t = CFUNCTYPE(
    c_void_p, POINTER(PJRT_LoadedExecutable_Execute_Args)
)
PJRT_Buffer_Destroy_t = CFUNCTYPE(c_void_p, POINTER(PJRT_Buffer_Destroy_Args))
PJRT_Buffer_ElementType_t = CFUNCTYPE(c_void_p, POINTER(PJRT_Buffer_ElementType_Args))
PJRT_Buffer_Dimensions_t = CFUNCTYPE(c_void_p, POINTER(PJRT_Buffer_Dimensions_Args))
PJRT_Buffer_ToHostBuffer_t = CFUNCTYPE(c_void_p, POINTER(PJRT_Buffer_ToHostBuffer_Args))

#: The PJRT_Api function-pointer field names, EXACTLY in header order
#: (extracted from the header at HEADER_COMMIT). Unused fields are declared
#: as c_void_p placeholders — all pointers are the same size, so the layout
#: matches the plugin's compiled struct regardless of the placeholder type.
_API_FIELD_NAMES = (
    "PJRT_Error_Destroy",
    "PJRT_Error_Message",
    "PJRT_Error_GetCode",
    "PJRT_Plugin_Initialize",
    "PJRT_Plugin_Attributes",
    "PJRT_Event_Destroy",
    "PJRT_Event_IsReady",
    "PJRT_Event_Error",
    "PJRT_Event_Await",
    "PJRT_Event_OnReady",
    "PJRT_Client_Create",
    "PJRT_Client_Destroy",
    "PJRT_Client_PlatformName",
    "PJRT_Client_ProcessIndex",
    "PJRT_Client_PlatformVersion",
    "PJRT_Client_Devices",
    "PJRT_Client_AddressableDevices",
    "PJRT_Client_LookupDevice",
    "PJRT_Client_LookupAddressableDevice",
    "PJRT_Client_AddressableMemories",
    "PJRT_Client_Compile",
    "PJRT_Client_DefaultDeviceAssignment",
    "PJRT_Client_BufferFromHostBuffer",
    "PJRT_DeviceDescription_Id",
    "PJRT_DeviceDescription_ProcessIndex",
    "PJRT_DeviceDescription_Attributes",
    "PJRT_DeviceDescription_Kind",
    "PJRT_DeviceDescription_DebugString",
    "PJRT_DeviceDescription_ToString",
    "PJRT_Device_GetDescription",
    "PJRT_Device_IsAddressable",
    "PJRT_Device_LocalHardwareId",
    "PJRT_Device_AddressableMemories",
    "PJRT_Device_DefaultMemory",
    "PJRT_Device_MemoryStats",
    "PJRT_Memory_Id",
    "PJRT_Memory_Kind",
    "PJRT_Memory_DebugString",
    "PJRT_Memory_ToString",
    "PJRT_Memory_AddressableByDevices",
    "PJRT_Executable_Destroy",
    "PJRT_Executable_Name",
    "PJRT_Executable_NumReplicas",
    "PJRT_Executable_NumPartitions",
    "PJRT_Executable_NumOutputs",
    "PJRT_Executable_SizeOfGeneratedCodeInBytes",
    "PJRT_Executable_GetCostAnalysis",
    "PJRT_Executable_OutputMemoryKinds",
    "PJRT_Executable_OptimizedProgram",
    "PJRT_Executable_Serialize",
    "PJRT_LoadedExecutable_Destroy",
    "PJRT_LoadedExecutable_GetExecutable",
    "PJRT_LoadedExecutable_AddressableDevices",
    "PJRT_LoadedExecutable_Delete",
    "PJRT_LoadedExecutable_IsDeleted",
    "PJRT_LoadedExecutable_Execute",
    "PJRT_Executable_DeserializeAndLoad",
    "PJRT_LoadedExecutable_Fingerprint",
    "PJRT_Buffer_Destroy",
    "PJRT_Buffer_ElementType",
    "PJRT_Buffer_Dimensions",
    "PJRT_Buffer_UnpaddedDimensions",
    "PJRT_Buffer_DynamicDimensionIndices",
    "PJRT_Buffer_GetMemoryLayout",
    "PJRT_Buffer_OnDeviceSizeInBytes",
    "PJRT_Buffer_Device",
    "PJRT_Buffer_Memory",
    "PJRT_Buffer_Delete",
    "PJRT_Buffer_IsDeleted",
    "PJRT_Buffer_CopyToDevice",
    "PJRT_Buffer_ToHostBuffer",
    "PJRT_Buffer_IsOnCpu",
    "PJRT_Buffer_ReadyEvent",
    "PJRT_Buffer_UnsafePointer",
    "PJRT_Buffer_IncreaseExternalReferenceCount",
    "PJRT_Buffer_DecreaseExternalReferenceCount",
    "PJRT_Buffer_OpaqueDeviceMemoryDataPointer",
    "PJRT_CopyToDeviceStream_Destroy",
    "PJRT_CopyToDeviceStream_AddChunk",
    "PJRT_CopyToDeviceStream_TotalBytes",
    "PJRT_CopyToDeviceStream_GranuleSize",
    "PJRT_CopyToDeviceStream_CurrentBytes",
    "PJRT_TopologyDescription_Create",
    "PJRT_TopologyDescription_Destroy",
    "PJRT_TopologyDescription_PlatformName",
    "PJRT_TopologyDescription_PlatformVersion",
    "PJRT_TopologyDescription_GetDeviceDescriptions",
    "PJRT_TopologyDescription_Serialize",
    "PJRT_TopologyDescription_Attributes",
    "PJRT_Compile",
    "PJRT_Executable_OutputElementTypes",
    "PJRT_Executable_OutputDimensions",
    "PJRT_Buffer_CopyToMemory",
    "PJRT_Client_CreateViewOfDeviceBuffer",
    "PJRT_Executable_Fingerprint",
    "PJRT_Client_TopologyDescription",
    "PJRT_Executable_GetCompiledMemoryStats",
    "PJRT_Memory_Kind_Id",
    "PJRT_ExecuteContext_Create",
    "PJRT_ExecuteContext_Destroy",
    "PJRT_Buffer_CopyRawToHost",
    "PJRT_AsyncHostToDeviceTransferManager_Destroy",
    "PJRT_AsyncHostToDeviceTransferManager_TransferData",
    "PJRT_Client_CreateBuffersForAsyncHostToDevice",
    "PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer",
    "PJRT_AsyncHostToDeviceTransferManager_Device",
    "PJRT_AsyncHostToDeviceTransferManager_BufferCount",
    "PJRT_AsyncHostToDeviceTransferManager_BufferSize",
    "PJRT_AsyncHostToDeviceTransferManager_SetBufferError",
    "PJRT_AsyncHostToDeviceTransferManager_AddMetadata",
    "PJRT_Client_DmaMap",
    "PJRT_Client_DmaUnmap",
    "PJRT_Client_CreateUninitializedBuffer",
    "PJRT_Client_UpdateGlobalProcessInfo",
    "PJRT_TopologyDescription_Deserialize",
    "PJRT_Client_CreateAliasBuffer",
    "PJRT_Client_FulfillAliasBuffer",
    "PJRT_LoadedExecutable_GetDeviceAssignment",
    "PJRT_Client_CreateErrorBuffer",
    "PJRT_AsyncHostToDeviceTransferManager_TransferLiteral",
    "PJRT_Buffer_CopyRawToHostFuture",
    "PJRT_Device_PoisonExecution",
    "PJRT_Device_CreateAsyncTrackingEvent",
    "PJRT_AsyncTrackingEvent_Destroy",
    "PJRT_Executable_GetCompileOptions",
    "PJRT_Buffer_DonateWithControlDependency",
    "PJRT_Event_Create",
    "PJRT_Event_Set",
    "PJRT_Device_GetAttributes",
    "PJRT_Client_Load",
    "PJRT_LoadedExecutable_AddressableDeviceLogicalIds",
    "PJRT_Buffer_Bitcast",
    "PJRT_Error_ForEachPayload",
    "PJRT_TopologyDescription_Fingerprint",
    "PJRT_Executable_ParameterMemoryKinds",
    "PJRT_Device_ClearMemoryStats",
    "PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace",
    "PJRT_TopologyDescription_GetMemorySpaceKindIds",
)

#: name -> prototype for the entry points the driver actually calls.
_TYPED_API_FIELDS = {
    "PJRT_Error_Destroy": PJRT_Error_Destroy_t,
    "PJRT_Error_Message": PJRT_Error_Message_t,
    "PJRT_Plugin_Initialize": PJRT_Plugin_Initialize_t,
    "PJRT_Event_Destroy": PJRT_Event_Destroy_t,
    "PJRT_Event_Await": PJRT_Event_Await_t,
    "PJRT_Client_Create": PJRT_Client_Create_t,
    "PJRT_Client_Destroy": PJRT_Client_Destroy_t,
    "PJRT_Client_PlatformName": PJRT_Client_PlatformName_t,
    "PJRT_Client_PlatformVersion": PJRT_Client_PlatformVersion_t,
    "PJRT_Client_Devices": PJRT_Client_Devices_t,
    "PJRT_Client_AddressableDevices": PJRT_Client_AddressableDevices_t,
    "PJRT_Client_LookupDevice": PJRT_Client_LookupDevice_t,
    "PJRT_Client_Compile": PJRT_Client_Compile_t,
    "PJRT_Client_BufferFromHostBuffer": PJRT_Client_BufferFromHostBuffer_t,
    "PJRT_Executable_Destroy": PJRT_Executable_Destroy_t,
    "PJRT_LoadedExecutable_Destroy": PJRT_LoadedExecutable_Destroy_t,
    "PJRT_LoadedExecutable_GetExecutable": PJRT_LoadedExecutable_GetExecutable_t,
    "PJRT_Executable_NumOutputs": PJRT_Executable_NumOutputs_t,
    "PJRT_Executable_Serialize": PJRT_Executable_Serialize_t,
    "PJRT_Executable_DeserializeAndLoad": PJRT_Executable_DeserializeAndLoad_t,
    "PJRT_LoadedExecutable_Execute": PJRT_LoadedExecutable_Execute_t,
    "PJRT_Buffer_Destroy": PJRT_Buffer_Destroy_t,
    "PJRT_Buffer_ElementType": PJRT_Buffer_ElementType_t,
    "PJRT_Buffer_Dimensions": PJRT_Buffer_Dimensions_t,
    "PJRT_Buffer_ToHostBuffer": PJRT_Buffer_ToHostBuffer_t,
}


class PJRT_Api(Structure):
    """The PJRT_Api function table (header, ``struct PJRT_Api``).

    Field order and count are EXACTLY the header's (138 function-pointer
    fields after the three leading metadata fields) — offsets must match the
    plugin's compiled struct. The plugin sets ``struct_size`` to the size of
    the struct IT was compiled with; the driver checks it (see
    ``verify_api``).
    """

    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", c_void_p),
        ("pjrt_api_version", PJRT_Api_Version),
    ] + [(name, _TYPED_API_FIELDS.get(name, c_void_p)) for name in _API_FIELD_NAMES]


#: Offset (plus one pointer) of the LAST PJRT_Api entry point this driver
#: reads — the structural floor for plugin compatibility.
_LAST_NEEDED_API_FIELD = "PJRT_Buffer_ToHostBuffer"
_LAST_NEEDED_API_OFFSET = getattr(PJRT_Api, _LAST_NEEDED_API_FIELD).offset

#: ctypes.sizeof re-export (convenience for the driver's struct_size args).
sizeof = ctypes.sizeof


def verify_api(api: "PJRT_Api") -> None:
    """Gate a plugin's ``PJRT_Api`` against the ABI this module requires.

    Per the header's versioning contract: the ABI family (``major_version``)
    must match exactly, and the plugin's ``struct_size`` must cover the last
    function field this driver reads — fields are only ever APPENDED to
    ``PJRT_Api``, so ``struct_size`` records exactly how many the plugin
    knows. Raises ``PJRTError`` with an explicit message when incompatible —
    never a silent fallback.
    """
    if api.struct_size == 0:
        raise PJRTError(
            "the plugin returned a PJRT_Api with struct_size=0 — it did not "
            "initialize the ABI record"
        )
    version = api.pjrt_api_version
    if version.major_version != PJRT_API_MAJOR:
        raise PJRTError(
            f"ABI mismatch: the plugin was compiled against PJRT API major "
            f"{version.major_version}, but this adapter requires major "
            f"{PJRT_API_MAJOR} (the header at commit {HEADER_COMMIT[:12]})"
        )
    required = _LAST_NEEDED_API_OFFSET + sizeof(c_void_p)
    if api.struct_size < required:
        raise PJRTError(
            f"the plugin's PJRT_Api is too small for this adapter: its "
            f"struct_size is {api.struct_size} bytes but the adapter needs "
            f"at least {required} bytes (through "
            f"{_LAST_NEEDED_API_FIELD}); the plugin is older than the "
            f"header at commit {HEADER_COMMIT[:12]} (PJRT API minor "
            f"{PJRT_API_MINOR})"
        )


def load_plugin_library(path: str):
    """``dlopen`` a PJRT plugin and return ``(library, PJRT_Api)``.

    Loads ``path`` with ``ctypes.CDLL``, resolves ``GetPjRtApi``, and
    validates the returned ``PJRT_Api`` (non-NULL + ``verify_api``).
    Raises ``PJRTError`` on any failure (missing symbol, NULL api, version
    gate) — the driver wraps it into ``core.BackendError``.
    """
    try:
        library = ctypes.CDLL(path)
    except OSError as exc:
        raise PJRTError(
            f"failed to load the PJRT plugin library {path!r}: {exc}"
        ) from exc
    try:
        get_api = getattr(library, "GetPjRtApi", None)
        if get_api is None:
            # Real-world plugins (e.g. jax_cuda12_pjrt 0.4.38's
            # xla_cuda_plugin.so) export the C symbol as ``GetPjrtApi``
            # (lowercase "t" — the plugin's C++ shim spelling); accept
            # both spellings.
            get_api = getattr(library, "GetPjrtApi", None)
        if get_api is None:
            raise PJRTError(
                f"the library at {path!r} does not export GetPjRtApi()/"
                "GetPjrtApi() — it is not a PJRT C API plugin"
            )
    except AttributeError as exc:
        raise PJRTError(
            f"the library at {path!r} does not export GetPjRtApi()/"
            "GetPjrtApi() — it is not a PJRT C API plugin"
        ) from exc
    get_api.restype = POINTER(PJRT_Api)
    get_api.argtypes = []
    api_ptr = get_api()
    if not api_ptr:
        raise PJRTError(
            f"GetPjRtApi() returned NULL from the plugin at {path!r}"
        )
    api = api_ptr.contents
    verify_api(api)
    return library, api
