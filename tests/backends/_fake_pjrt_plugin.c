/* _fake_pjrt_plugin.c — TEST-ONLY fake PJRT C API plugin shim.
 *
 * A minimal PJRT C API plugin (.so exporting GetPjRtApi()) used by
 * tests/backends/test_pjrt_ctypes_plugin.py to exercise the ctypes PJRT
 * driver (etl/backends/adapters/_pjrt_c_api.py + xla_util.py) WITHOUT a
 * real XLA plugin. It accepts StableHLO MLIR text at compile time, records
 * the program text, and returns ZERO-FILLED output buffers of the shapes
 * declared in the entry function's result types — the adapter's plumbing
 * (discovery, version gate, compile, serialize, deserialize, execute,
 * copy-back, error reporting) is exercised for real; numerical results are
 * deliberately not meaningful.
 *
 * ABI fidelity (binding): the struct field ORDER below is EXACTLY the order
 * in etl/backends/adapters/_pjrt_c_api.py — both are derived from the
 * canonical OpenXLA header xla/pjrt/c/pjrt_c_api.h at commit
 * 70fe66213b73c5953d92eb25d2606bd6004d47c3. The _Static_asserts pin
 * sizeof(PJRT_Api) == 1144 (3 metadata fields + 138 function pointers on
 * LP64) and the offsets the driver gates on, so layout drift fails at
 * compile time.
 *
 * Build (64-bit Linux):  gcc -shared -fPIC -std=c99 -O1 -o plugin.so this.c
 * Test-only knobs (via -D at build time, and env vars at run time):
 *   -DETL_FAKE_PJRT_STRUCT_SIZE=N   override PJRT_Api.struct_size (ABI gate)
 *   -DETL_FAKE_PJRT_MAJOR_VERSION=N override pjrt_api_version.major_version
 *   ETL_FAKE_PJRT_FAIL_STEP=<entry point name>  (env): make that call fail
 *   ETL_FAKE_PJRT_FAIL_MESSAGE=<text>           (env): the error message
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ types */

/* PJRT_ExecuteOptions is passed by pointer but never dereferenced here. */
typedef struct PJRT_ExecuteOptions PJRT_ExecuteOptions;

/* PJRT_Buffer_Type values used by this fake (from the header). */
enum {
  FAKE_PRED = 1, FAKE_S8 = 2, FAKE_S16 = 3, FAKE_S32 = 4, FAKE_S64 = 5,
  FAKE_U8 = 6, FAKE_U16 = 7, FAKE_U32 = 8, FAKE_U64 = 9,
  FAKE_F16 = 10, FAKE_F32 = 11, FAKE_F64 = 12, FAKE_C64 = 14, FAKE_C128 = 15
};

/* PJRT_Program (header: code/format owned by the caller, valid for the
 * duration of the call; sizes exclude the NUL terminator). */
typedef struct PJRT_Program {
  size_t struct_size;
  void* extension_start;
  void* code;
  size_t code_size;
  void* format;
  size_t format_size;
} PJRT_Program;

/* The opaque PJRT handles are defined CONCRETELY here — the fake owns the
 * storage layout of everything it creates. */
typedef struct PJRT_Error {
  char* message;
  size_t message_size;
} PJRT_Error;

typedef struct PJRT_Event {
  int ready;
} PJRT_Event;

typedef struct PJRT_Client {
  int dummy;
} PJRT_Client;

typedef struct PJRT_Device {
  int dummy;
} PJRT_Device;

typedef struct PJRT_Memory {
  int dummy;
} PJRT_Memory;

typedef struct PJRT_Buffer {
  int type;      /* PJRT_Buffer_Type */
  int64_t* dims; /* num_dims entries */
  size_t num_dims;
  size_t nbytes;
  unsigned char* data;
} PJRT_Buffer;

typedef struct fake_output {
  int type;
  int64_t* dims;
  size_t num_dims;
} fake_output;

typedef struct PJRT_Executable { /* the unloaded PJRT_Executable */
  char* program;                 /* NUL-terminated copy of the program text */
  size_t program_size;
  size_t num_outputs;
  fake_output* outputs;
} PJRT_Executable;

typedef struct PJRT_LoadedExecutable { /* the PJRT_LoadedExecutable */
  PJRT_Executable* exe;
} PJRT_LoadedExecutable;

/* Argument structs — field ORDER and types match _pjrt_c_api.py exactly
 * (const qualifiers dropped; layout is identical). */
typedef struct PJRT_Error_Destroy_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_Error* error;
} PJRT_Error_Destroy_Args;

typedef struct PJRT_Error_Message_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_Error* error;
  char* message;       /* out */
  size_t message_size; /* out */
} PJRT_Error_Message_Args;

typedef struct PJRT_Plugin_Initialize_Args {
  size_t struct_size;
  void* extension_start;
} PJRT_Plugin_Initialize_Args;

typedef struct PJRT_Event_Destroy_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_Event* event;
} PJRT_Event_Destroy_Args;

typedef struct PJRT_Event_Await_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_Event* event;
} PJRT_Event_Await_Args;

typedef struct PJRT_Client_Create_Args {
  size_t struct_size;
  void* extension_start;
  void* create_options;
  size_t num_options;
  void* kv_get_callback;
  void* kv_get_user_arg;
  void* kv_put_callback;
  void* kv_put_user_arg;
  PJRT_Client* client; /* out */
  void* kv_try_get_callback;
  void* kv_try_get_user_arg;
} PJRT_Client_Create_Args;

typedef struct PJRT_Client_Destroy_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_Client* client;
} PJRT_Client_Destroy_Args;

typedef struct PJRT_Client_PlatformName_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_Client* client;
  const char* platform_name; /* out */
  size_t platform_name_size; /* out */
} PJRT_Client_PlatformName_Args;

typedef struct PJRT_Client_PlatformVersion_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_Client* client;
  const char* platform_version; /* out */
  size_t platform_version_size; /* out */
} PJRT_Client_PlatformVersion_Args;

typedef struct PJRT_Client_AddressableDevices_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_Client* client;
  PJRT_Device** addressable_devices; /* out */
  size_t num_addressable_devices;    /* out */
} PJRT_Client_AddressableDevices_Args;

typedef struct PJRT_Client_Compile_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_Client* client;
  PJRT_Program* program; /* const, caller-owned */
  void* compile_options;
  size_t compile_options_size;
  PJRT_LoadedExecutable* executable; /* out */
} PJRT_Client_Compile_Args;

typedef struct PJRT_Client_BufferFromHostBuffer_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_Client* client;
  const void* data; /* host buffer */
  int type;         /* PJRT_Buffer_Type */
  int64_t* dims;    /* const int64_t* */
  size_t num_dims;
  int64_t* byte_strides; /* NULL/empty = dense row-major */
  size_t num_byte_strides;
  int host_buffer_semantics; /* PJRT_HostBufferSemantics */
  PJRT_Device* device;
  PJRT_Memory* memory;
  void* device_layout;
  PJRT_Event* done_with_host_buffer; /* out */
  PJRT_Buffer* buffer;              /* out */
} PJRT_Client_BufferFromHostBuffer_Args;

typedef struct PJRT_Executable_Destroy_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_Executable* executable;
} PJRT_Executable_Destroy_Args;

typedef struct PJRT_LoadedExecutable_Destroy_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_LoadedExecutable* executable;
} PJRT_LoadedExecutable_Destroy_Args;

typedef struct PJRT_LoadedExecutable_GetExecutable_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_LoadedExecutable* loaded_executable;
  PJRT_Executable* executable; /* out */
} PJRT_LoadedExecutable_GetExecutable_Args;

typedef struct PJRT_Executable_NumOutputs_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_Executable* executable;
  size_t num_outputs; /* out */
} PJRT_Executable_NumOutputs_Args;

typedef void (*fake_serialized_deleter_fn)(void*);

typedef struct PJRT_Executable_Serialize_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_Executable* executable; /* const */
  void* serialized_bytes;      /* out */
  size_t serialized_bytes_size; /* out */
  void* serialized_executable;  /* out: backs serialized_bytes */
  fake_serialized_deleter_fn serialized_executable_deleter; /* out */
} PJRT_Executable_Serialize_Args;

typedef struct PJRT_Executable_DeserializeAndLoad_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_Client* client;
  void* serialized_executable; /* const char* */
  size_t serialized_executable_size;
  PJRT_LoadedExecutable* loaded_executable; /* out */
  void* overridden_serialized_compile_options;
  size_t overridden_serialized_compile_options_size;
  void* load_options;
} PJRT_Executable_DeserializeAndLoad_Args;

typedef struct PJRT_LoadedExecutable_Execute_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_LoadedExecutable* executable;
  PJRT_ExecuteOptions* options;        /* caller-owned, per call */
  PJRT_Buffer** argument_lists;        /* [num_devices][num_args] */
  size_t num_devices;
  size_t num_args;
  PJRT_Buffer** output_lists;          /* in/out, caller-allocated */
  PJRT_Event** device_complete_events; /* in/out, NULL ok */
  PJRT_Device* execute_device;         /* NULL = compile-time device */
} PJRT_LoadedExecutable_Execute_Args;

typedef struct PJRT_Buffer_Destroy_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_Buffer* buffer;
} PJRT_Buffer_Destroy_Args;

typedef struct PJRT_Buffer_ElementType_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_Buffer* buffer;
  int type; /* out: PJRT_Buffer_Type */
} PJRT_Buffer_ElementType_Args;

typedef struct PJRT_Buffer_Dimensions_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_Buffer* buffer;
  int64_t* dims;   /* out, lifetime of buffer */
  size_t num_dims; /* out */
} PJRT_Buffer_Dimensions_Args;

typedef struct PJRT_Buffer_ToHostBuffer_Args {
  size_t struct_size;
  void* extension_start;
  PJRT_Buffer* src;
  void* host_layout; /* NULL = buffer's layout */
  void* dst;         /* in/out; NULL queries required dst_size */
  size_t dst_size;   /* in/out */
  PJRT_Event* event; /* out */
} PJRT_Buffer_ToHostBuffer_Args;

/* Entry-point function types (all return PJRT_Error* = NULL on success,
 * except Error_Destroy/Error_Message which return void). */
typedef void (*PJRT_Error_Destroy_fn)(PJRT_Error_Destroy_Args*);
typedef void (*PJRT_Error_Message_fn)(PJRT_Error_Message_Args*);
typedef PJRT_Error* (*PJRT_Plugin_Initialize_fn)(PJRT_Plugin_Initialize_Args*);
typedef PJRT_Error* (*PJRT_Event_Destroy_fn)(PJRT_Event_Destroy_Args*);
typedef PJRT_Error* (*PJRT_Event_Await_fn)(PJRT_Event_Await_Args*);
typedef PJRT_Error* (*PJRT_Client_Create_fn)(PJRT_Client_Create_Args*);
typedef PJRT_Error* (*PJRT_Client_Destroy_fn)(PJRT_Client_Destroy_Args*);
typedef PJRT_Error* (*PJRT_Client_PlatformName_fn)(PJRT_Client_PlatformName_Args*);
typedef PJRT_Error* (*PJRT_Client_PlatformVersion_fn)(PJRT_Client_PlatformVersion_Args*);
typedef PJRT_Error* (*PJRT_Client_AddressableDevices_fn)(PJRT_Client_AddressableDevices_Args*);
typedef PJRT_Error* (*PJRT_Client_Compile_fn)(PJRT_Client_Compile_Args*);
typedef PJRT_Error* (*PJRT_Client_BufferFromHostBuffer_fn)(PJRT_Client_BufferFromHostBuffer_Args*);
typedef PJRT_Error* (*PJRT_Executable_Destroy_fn)(PJRT_Executable_Destroy_Args*);
typedef PJRT_Error* (*PJRT_LoadedExecutable_Destroy_fn)(PJRT_LoadedExecutable_Destroy_Args*);
typedef PJRT_Error* (*PJRT_LoadedExecutable_GetExecutable_fn)(PJRT_LoadedExecutable_GetExecutable_Args*);
typedef PJRT_Error* (*PJRT_Executable_NumOutputs_fn)(PJRT_Executable_NumOutputs_Args*);
typedef PJRT_Error* (*PJRT_Executable_Serialize_fn)(PJRT_Executable_Serialize_Args*);
typedef PJRT_Error* (*PJRT_Executable_DeserializeAndLoad_fn)(PJRT_Executable_DeserializeAndLoad_Args*);
typedef PJRT_Error* (*PJRT_LoadedExecutable_Execute_fn)(PJRT_LoadedExecutable_Execute_Args*);
typedef PJRT_Error* (*PJRT_Buffer_Destroy_fn)(PJRT_Buffer_Destroy_Args*);
typedef PJRT_Error* (*PJRT_Buffer_ElementType_fn)(PJRT_Buffer_ElementType_Args*);
typedef PJRT_Error* (*PJRT_Buffer_Dimensions_fn)(PJRT_Buffer_Dimensions_Args*);
typedef PJRT_Error* (*PJRT_Buffer_ToHostBuffer_fn)(PJRT_Buffer_ToHostBuffer_Args*);

/* ------------------------------------------------------------ error path */

static PJRT_Error* make_error(const char* msg) {
  PJRT_Error* e = calloc(1, sizeof(*e));
  if (!e) return NULL;
  size_t n = strlen(msg);
  e->message = malloc(n + 1);
  if (!e->message) {
    free(e);
    return NULL;
  }
  memcpy(e->message, msg, n + 1);
  e->message_size = n;
  return e;
}

/* Test-only error injection: when ETL_FAKE_PJRT_FAIL_STEP names the entry
 * point being called, return an error carrying ETL_FAKE_PJRT_FAIL_MESSAGE
 * (or a default). */
static PJRT_Error* maybe_fail(const char* step) {
  const char* want = getenv("ETL_FAKE_PJRT_FAIL_STEP");
  if (want && strcmp(want, step) == 0) {
    const char* msg = getenv("ETL_FAKE_PJRT_FAIL_MESSAGE");
    return make_error(msg && *msg ? msg : "injected fake PJRT plugin failure");
  }
  return NULL;
}

static void fake_PJRT_Error_Destroy(PJRT_Error_Destroy_Args* args) {
  if (!args || !args->error) return;
  free(args->error->message);
  free(args->error);
}

static void fake_PJRT_Error_Message(PJRT_Error_Message_Args* args) {
  if (!args || !args->error) return;
  args->message = args->error->message;
  args->message_size = args->error->message_size;
}

/* ---------------------------------------------------------- event helpers */

static PJRT_Event* make_event(void) {
  PJRT_Event* ev = calloc(1, sizeof(*ev));
  if (ev) ev->ready = 1;
  return ev;
}

static PJRT_Error* fake_PJRT_Event_Destroy(PJRT_Event_Destroy_Args* args) {
  PJRT_Error* err = maybe_fail("PJRT_Event_Destroy");
  if (err) return err;
  free(args->event);
  return NULL;
}

static PJRT_Error* fake_PJRT_Event_Await(PJRT_Event_Await_Args* args) {
  PJRT_Error* err = maybe_fail("PJRT_Event_Await");
  if (err) return err;
  if (args->event) args->event->ready = 1;
  return NULL;
}

/* --------------------------------------------------------- buffer helpers */

static size_t element_size(int type) {
  switch (type) {
    case FAKE_PRED: case FAKE_S8: case FAKE_U8: return 1;
    case FAKE_S16: case FAKE_U16: case FAKE_F16: return 2;
    case FAKE_S32: case FAKE_U32: case FAKE_F32: return 4;
    case FAKE_S64: case FAKE_U64: case FAKE_F64: case FAKE_C64: return 8;
    case FAKE_C128: return 16;
    default: return 4;
  }
}

static PJRT_Error* make_buffer(int type, const int64_t* dims, size_t num_dims,
                               const void* data, int zero_fill,
                               PJRT_Buffer** out) {
  PJRT_Buffer* b = calloc(1, sizeof(*b));
  if (!b) return make_error("fake: out of memory");
  b->type = type;
  b->num_dims = num_dims;
  if (num_dims > 0) {
    b->dims = malloc(num_dims * sizeof(int64_t));
    if (!b->dims) {
      free(b);
      return make_error("fake: out of memory");
    }
    memcpy(b->dims, dims, num_dims * sizeof(int64_t));
  }
  size_t elems = 1;
  for (size_t i = 0; i < num_dims; i++) elems *= (size_t)b->dims[i];
  b->nbytes = elems * element_size(type);
  b->data = malloc(b->nbytes ? b->nbytes : 1);
  if (!b->data) {
    free(b->dims);
    free(b);
    return make_error("fake: out of memory");
  }
  if (zero_fill) {
    memset(b->data, 0, b->nbytes);
  } else if (data && b->nbytes) {
    memcpy(b->data, data, b->nbytes);
  }
  *out = b;
  return NULL;
}

static PJRT_Error* fake_PJRT_Buffer_Destroy(PJRT_Buffer_Destroy_Args* args) {
  PJRT_Error* err = maybe_fail("PJRT_Buffer_Destroy");
  if (err) return err;
  free(args->buffer->data);
  free(args->buffer->dims);
  free(args->buffer);
  return NULL;
}

static PJRT_Error* fake_PJRT_Buffer_ElementType(PJRT_Buffer_ElementType_Args* args) {
  PJRT_Error* err = maybe_fail("PJRT_Buffer_ElementType");
  if (err) return err;
  args->type = args->buffer->type;
  return NULL;
}

static PJRT_Error* fake_PJRT_Buffer_Dimensions(PJRT_Buffer_Dimensions_Args* args) {
  PJRT_Error* err = maybe_fail("PJRT_Buffer_Dimensions");
  if (err) return err;
  args->dims = args->buffer->dims;
  args->num_dims = args->buffer->num_dims;
  return NULL;
}

static PJRT_Error* fake_PJRT_Buffer_ToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* args) {
  PJRT_Error* err = maybe_fail("PJRT_Buffer_ToHostBuffer");
  if (err) return err;
  if (!args->dst) {
    args->dst_size = args->src->nbytes; /* query mode */
  } else {
    size_t n = args->src->nbytes < args->dst_size ? args->src->nbytes
                                                  : args->dst_size;
    memcpy(args->dst, args->src->data, n);
  }
  args->event = make_event();
  return NULL;
}

/* ------------------------------------------------------- MLIR type parser */

/* Map an MLIR element-type token to PJRT_Buffer_Type (or -1). */
static int dtype_token_to_pjrt(const char* s, size_t n) {
  static const struct {
    const char* name;
    size_t len;
    int value;
  } table[] = {
      {"f16", 3, FAKE_F16}, {"f32", 3, FAKE_F32}, {"f64", 3, FAKE_F64},
      {"i1", 2, FAKE_PRED}, {"i8", 2, FAKE_S8},   {"i16", 3, FAKE_S16},
      {"i32", 3, FAKE_S32}, {"i64", 3, FAKE_S64}, {"ui8", 3, FAKE_U8},
      {"ui16", 4, FAKE_U16}, {"ui32", 4, FAKE_U32}, {"ui64", 4, FAKE_U64},
      {"complex<f32>", 12, FAKE_C64}, {"complex<f64>", 12, FAKE_C128},
  };
  for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++) {
    if (table[i].len == n && memcmp(s, table[i].name, n) == 0)
      return table[i].value;
  }
  return -1;
}

/* Parse one `tensor<...>` type at *p; advance *p past it. Fills *type
 * (PJRT_Buffer_Type), *dims (malloc'd int64 array, caller frees),
 * *num_dims. Returns 0 on success, -1 on any parse failure. */
static int parse_tensor_type(const char** p, int* type, int64_t** dims,
                             size_t* num_dims) {
  const char* s = *p;
  if (strncmp(s, "tensor<", 7) != 0) return -1;
  s += 7;
  const char* end = s;
  int depth = 0;
  while (*end != '\0' && (*end != '>' || depth > 0)) {
    if (*end == '<') depth++;
    else if (*end == '>') depth--;
    end++;
  }
  if (*end != '>') return -1;

  /* Count 'x'-separated segments at depth 0; the last is the dtype token. */
  size_t nseg = 1;
  int seg_depth = 0;
  for (const char* q = s; q < end; q++) {
    if (*q == '<') seg_depth++;
    else if (*q == '>') seg_depth--;
    else if (*q == 'x' && seg_depth == 0) nseg++;
  }

  int64_t* dims_out = NULL;
  if (nseg > 1) {
    dims_out = malloc((nseg - 1) * sizeof(int64_t));
    if (!dims_out) return -1;
  }
  size_t ndims = 0;
  const char* q = s;
  seg_depth = 0;
  int rc = -1;
  for (;;) {
    const char* seg_start = q;
    while (q < end && !(*q == 'x' && seg_depth == 0)) {
      if (*q == '<') seg_depth++;
      else if (*q == '>') seg_depth--;
      q++;
    }
    size_t seg_len = (size_t)(q - seg_start);
    int is_last = (q >= end);
    if (is_last) {
      int t = dtype_token_to_pjrt(seg_start, seg_len);
      if (t < 0) break;
      *type = t;
      rc = 0;
      break;
    }
    char buf[64];
    if (seg_len >= sizeof(buf)) break;
    memcpy(buf, seg_start, seg_len);
    buf[seg_len] = '\0';
    if (buf[0] == '?' || buf[0] == '\0') break; /* dynamic dims unsupported */
    char* endptr = NULL;
    long long v = strtoll(buf, &endptr, 10);
    if (endptr == buf || *endptr != '\0' || v < 0) break;
    dims_out[ndims++] = (int64_t)v;
    q++; /* skip 'x' */
  }
  if (rc != 0) {
    free(dims_out);
    return -1;
  }
  *dims = dims_out;
  *num_dims = ndims;
  *p = end + 1;
  return 0;
}

static void free_fake_outputs(fake_output* outputs, size_t count) {
  if (!outputs) return;
  for (size_t i = 0; i < count; i++) free(outputs[i].dims);
  free(outputs);
}

/* Parse the entry function's result types from StableHLO MLIR text: the
 * etl writer emits `func.func @main(...) -> tensor<...>` (or a
 * parenthesized list) on one line — take the FIRST func.func. */
static int parse_output_types(const char* text, fake_output** out,
                              size_t* out_count) {
  *out = NULL;
  *out_count = 0;
  const char* fn = strstr(text, "func.func @");
  if (!fn) return -1;
  const char* ret = strstr(fn, ") -> ");
  if (!ret) return -1;
  const char* p = ret + 5;
  fake_output* outputs = NULL;
  size_t count = 0;
  int rc = -1;
  if (*p == '(') {
    p++;
    for (;;) {
      while (*p == ' ' || *p == ',' || *p == '\t') p++;
      if (*p == ')') {
        rc = 0;
        break;
      }
      if (*p == '\0') break;
      int type;
      int64_t* dims;
      size_t ndims;
      if (parse_tensor_type(&p, &type, &dims, &ndims) != 0) break;
      fake_output* grown = realloc(outputs, (count + 1) * sizeof(*grown));
      if (!grown) {
        free(dims);
        break;
      }
      outputs = grown;
      outputs[count].type = type;
      outputs[count].dims = dims;
      outputs[count].num_dims = ndims;
      count++;
    }
  } else {
    int type;
    int64_t* dims;
    size_t ndims;
    if (parse_tensor_type(&p, &type, &dims, &ndims) == 0) {
      outputs = malloc(sizeof(*outputs));
      if (outputs) {
        outputs[0].type = type;
        outputs[0].dims = dims;
        outputs[0].num_dims = ndims;
        count = 1;
        rc = 0;
      } else {
        free(dims);
      }
    }
  }
  if (rc != 0) {
    free_fake_outputs(outputs, count);
    return -1;
  }
  *out = outputs;
  *out_count = count;
  return 0;
}

/* ------------------------------------------------------- exe construction */

static PJRT_Executable* make_exe_from_program(const char* code, size_t code_size,
                                              PJRT_Error** err_out) {
  *err_out = NULL;
  PJRT_Executable* exe = calloc(1, sizeof(*exe));
  if (!exe) {
    *err_out = make_error("fake: out of memory");
    return NULL;
  }
  exe->program = malloc(code_size + 1);
  if (!exe->program) {
    free(exe);
    *err_out = make_error("fake: out of memory");
    return NULL;
  }
  memcpy(exe->program, code, code_size);
  exe->program[code_size] = '\0';
  exe->program_size = code_size;
  if (parse_output_types(exe->program, &exe->outputs, &exe->num_outputs) != 0) {
    free(exe->program);
    free(exe);
    *err_out = make_error(
        "fake: cannot parse the entry function's result types from the MLIR "
        "text");
    return NULL;
  }
  return exe;
}

static PJRT_Error* make_loaded_exe(PJRT_Executable* exe,
                                   PJRT_LoadedExecutable** out) {
  PJRT_LoadedExecutable* loaded = calloc(1, sizeof(*loaded));
  if (!loaded) return make_error("fake: out of memory");
  loaded->exe = exe;
  *out = loaded;
  return NULL;
}

/* ---------------------------------------------------------- entry points */

static PJRT_Error* fake_PJRT_Plugin_Initialize(PJRT_Plugin_Initialize_Args* args) {
  (void)args;
  return maybe_fail("PJRT_Plugin_Initialize");
}

static PJRT_Client g_fake_client_storage;
static PJRT_Device g_fake_device;
static PJRT_Device* g_fake_devices[1] = {&g_fake_device};

static PJRT_Error* fake_PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  PJRT_Error* err = maybe_fail("PJRT_Client_Create");
  if (err) return err;
  if (args->num_options != 0)
    return make_error("fake: only empty client create options are supported");
  args->client = &g_fake_client_storage;
  return NULL;
}

static PJRT_Error* fake_PJRT_Client_Destroy(PJRT_Client_Destroy_Args* args) {
  (void)args;
  return maybe_fail("PJRT_Client_Destroy");
}

static PJRT_Error* fake_PJRT_Client_PlatformName(PJRT_Client_PlatformName_Args* args) {
  PJRT_Error* err = maybe_fail("PJRT_Client_PlatformName");
  if (err) return err;
  args->platform_name = "fake";
  args->platform_name_size = 4;
  return NULL;
}

static PJRT_Error* fake_PJRT_Client_PlatformVersion(PJRT_Client_PlatformVersion_Args* args) {
  PJRT_Error* err = maybe_fail("PJRT_Client_PlatformVersion");
  if (err) return err;
  args->platform_version = "fake-cpu 0.0.1";
  args->platform_version_size = 14;
  return NULL;
}

static PJRT_Error* fake_PJRT_Client_AddressableDevices(
    PJRT_Client_AddressableDevices_Args* args) {
  PJRT_Error* err = maybe_fail("PJRT_Client_AddressableDevices");
  if (err) return err;
  args->addressable_devices = g_fake_devices;
  args->num_addressable_devices = 1;
  return NULL;
}

static PJRT_Error* fake_PJRT_Client_Compile(PJRT_Client_Compile_Args* args) {
  PJRT_Error* err = maybe_fail("PJRT_Client_Compile");
  if (err) return err;
  if (!args->program || !args->program->code || args->program->code_size == 0)
    return make_error("fake: PJRT_Program with empty code");
  PJRT_Executable* exe =
      make_exe_from_program(args->program->code, args->program->code_size, &err);
  if (!exe) return err;
  err = make_loaded_exe(exe, &args->executable);
  if (err) {
    free(exe->program);
    free_fake_outputs(exe->outputs, exe->num_outputs);
    free(exe);
  }
  return err;
}

static PJRT_Error* fake_PJRT_Client_BufferFromHostBuffer(
    PJRT_Client_BufferFromHostBuffer_Args* args) {
  PJRT_Error* err = maybe_fail("PJRT_Client_BufferFromHostBuffer");
  if (err) return err;
  PJRT_Buffer* buffer = NULL;
  err = make_buffer(args->type, args->dims, args->num_dims, args->data, 0,
                    &buffer);
  if (err) return err;
  args->buffer = buffer;
  args->done_with_host_buffer = make_event();
  if (!args->done_with_host_buffer) {
    free(buffer->data);
    free(buffer->dims);
    free(buffer);
    return make_error("fake: out of memory");
  }
  return NULL;
}

static PJRT_Error* fake_PJRT_Executable_Destroy(PJRT_Executable_Destroy_Args* args) {
  PJRT_Error* err = maybe_fail("PJRT_Executable_Destroy");
  if (err) return err;
  free(args->executable->program);
  free_fake_outputs(args->executable->outputs, args->executable->num_outputs);
  free(args->executable);
  return NULL;
}

static PJRT_Error* fake_PJRT_LoadedExecutable_Destroy(
    PJRT_LoadedExecutable_Destroy_Args* args) {
  PJRT_Error* err = maybe_fail("PJRT_LoadedExecutable_Destroy");
  if (err) return err;
  free(args->executable); /* the inner exe is destroyed separately */
  return NULL;
}

static PJRT_Error* fake_PJRT_LoadedExecutable_GetExecutable(
    PJRT_LoadedExecutable_GetExecutable_Args* args) {
  PJRT_Error* err = maybe_fail("PJRT_LoadedExecutable_GetExecutable");
  if (err) return err;
  args->executable = args->loaded_executable->exe;
  return NULL;
}

static PJRT_Error* fake_PJRT_Executable_NumOutputs(PJRT_Executable_NumOutputs_Args* args) {
  PJRT_Error* err = maybe_fail("PJRT_Executable_NumOutputs");
  if (err) return err;
  args->num_outputs = args->executable->num_outputs;
  return NULL;
}

static void fake_serialized_deleter(void* ptr) { free(ptr); }

static PJRT_Error* fake_PJRT_Executable_Serialize(PJRT_Executable_Serialize_Args* args) {
  PJRT_Error* err = maybe_fail("PJRT_Executable_Serialize");
  if (err) return err;
  void* bytes = malloc(args->executable->program_size ? args->executable->program_size : 1);
  if (!bytes) return make_error("fake: out of memory");
  memcpy(bytes, args->executable->program, args->executable->program_size);
  args->serialized_bytes = bytes;
  args->serialized_bytes_size = args->executable->program_size;
  args->serialized_executable = bytes;
  args->serialized_executable_deleter = fake_serialized_deleter;
  return NULL;
}

static PJRT_Error* fake_PJRT_Executable_DeserializeAndLoad(
    PJRT_Executable_DeserializeAndLoad_Args* args) {
  PJRT_Error* err = maybe_fail("PJRT_Executable_DeserializeAndLoad");
  if (err) return err;
  if (!args->serialized_executable || args->serialized_executable_size == 0)
    return make_error("fake: empty serialized executable");
  PJRT_Executable* exe = make_exe_from_program(args->serialized_executable,
                                               args->serialized_executable_size,
                                               &err);
  if (!exe) return err;
  err = make_loaded_exe(exe, &args->loaded_executable);
  if (err) {
    free(exe->program);
    free_fake_outputs(exe->outputs, exe->num_outputs);
    free(exe);
  }
  return err;
}

static PJRT_Error* fake_PJRT_LoadedExecutable_Execute(
    PJRT_LoadedExecutable_Execute_Args* args) {
  PJRT_Error* err = maybe_fail("PJRT_LoadedExecutable_Execute");
  if (err) return err;
  if (args->num_devices != 1)
    return make_error("fake: only one device is supported");
  if (!args->output_lists || !args->output_lists[0])
    return make_error("fake: output_lists missing");
  /* output_lists[0] is a PJRT_Buffer* whose VALUE is the caller-allocated
   * array of num_outputs PJRT_Buffer* (see the header's contract). */
  PJRT_Buffer** out = (PJRT_Buffer**)args->output_lists[0];
  for (size_t i = 0; i < args->executable->exe->num_outputs; i++) {
    fake_output* o = &args->executable->exe->outputs[i];
    PJRT_Buffer* buffer = NULL;
    err = make_buffer(o->type, o->dims, o->num_dims, NULL, 1, &buffer);
    if (err) return err;
    out[i] = buffer;
  }
  return NULL;
}

/* ------------------------------------------------------------- PJRT_Api */

struct PJRT_Api_Version {
  size_t struct_size;
  void* extension_start;
  int major_version;
  int minor_version;
};

/* The PJRT_Api function table — 138 function-pointer fields EXACTLY in the
 * header's order (see _pjrt_c_api.py); unused fields are void* placeholders
 * (all pointers are the same size, so the layout matches). Only the entry
 * points the driver actually calls are implemented above. */
struct PJRT_Api {
  size_t struct_size;
  void* extension_start;
  struct PJRT_Api_Version pjrt_api_version;

  PJRT_Error_Destroy_fn PJRT_Error_Destroy;
  PJRT_Error_Message_fn PJRT_Error_Message;
  void* PJRT_Error_GetCode;
  PJRT_Plugin_Initialize_fn PJRT_Plugin_Initialize;
  void* PJRT_Plugin_Attributes;
  PJRT_Event_Destroy_fn PJRT_Event_Destroy;
  void* PJRT_Event_IsReady;
  void* PJRT_Event_Error;
  PJRT_Event_Await_fn PJRT_Event_Await;
  void* PJRT_Event_OnReady;
  PJRT_Client_Create_fn PJRT_Client_Create;
  PJRT_Client_Destroy_fn PJRT_Client_Destroy;
  PJRT_Client_PlatformName_fn PJRT_Client_PlatformName;
  void* PJRT_Client_ProcessIndex;
  PJRT_Client_PlatformVersion_fn PJRT_Client_PlatformVersion;
  void* PJRT_Client_Devices;
  PJRT_Client_AddressableDevices_fn PJRT_Client_AddressableDevices;
  void* PJRT_Client_LookupDevice;
  void* PJRT_Client_LookupAddressableDevice;
  void* PJRT_Client_AddressableMemories;
  PJRT_Client_Compile_fn PJRT_Client_Compile;
  void* PJRT_Client_DefaultDeviceAssignment;
  PJRT_Client_BufferFromHostBuffer_fn PJRT_Client_BufferFromHostBuffer;
  void* PJRT_DeviceDescription_Id;
  void* PJRT_DeviceDescription_ProcessIndex;
  void* PJRT_DeviceDescription_Attributes;
  void* PJRT_DeviceDescription_Kind;
  void* PJRT_DeviceDescription_DebugString;
  void* PJRT_DeviceDescription_ToString;
  void* PJRT_Device_GetDescription;
  void* PJRT_Device_IsAddressable;
  void* PJRT_Device_LocalHardwareId;
  void* PJRT_Device_AddressableMemories;
  void* PJRT_Device_DefaultMemory;
  void* PJRT_Device_MemoryStats;
  void* PJRT_Memory_Id;
  void* PJRT_Memory_Kind;
  void* PJRT_Memory_DebugString;
  void* PJRT_Memory_ToString;
  void* PJRT_Memory_AddressableByDevices;
  PJRT_Executable_Destroy_fn PJRT_Executable_Destroy;
  void* PJRT_Executable_Name;
  void* PJRT_Executable_NumReplicas;
  void* PJRT_Executable_NumPartitions;
  PJRT_Executable_NumOutputs_fn PJRT_Executable_NumOutputs;
  void* PJRT_Executable_SizeOfGeneratedCodeInBytes;
  void* PJRT_Executable_GetCostAnalysis;
  void* PJRT_Executable_OutputMemoryKinds;
  void* PJRT_Executable_OptimizedProgram;
  PJRT_Executable_Serialize_fn PJRT_Executable_Serialize;
  PJRT_LoadedExecutable_Destroy_fn PJRT_LoadedExecutable_Destroy;
  PJRT_LoadedExecutable_GetExecutable_fn PJRT_LoadedExecutable_GetExecutable;
  void* PJRT_LoadedExecutable_AddressableDevices;
  void* PJRT_LoadedExecutable_Delete;
  void* PJRT_LoadedExecutable_IsDeleted;
  PJRT_LoadedExecutable_Execute_fn PJRT_LoadedExecutable_Execute;
  PJRT_Executable_DeserializeAndLoad_fn PJRT_Executable_DeserializeAndLoad;
  void* PJRT_LoadedExecutable_Fingerprint;
  PJRT_Buffer_Destroy_fn PJRT_Buffer_Destroy;
  PJRT_Buffer_ElementType_fn PJRT_Buffer_ElementType;
  PJRT_Buffer_Dimensions_fn PJRT_Buffer_Dimensions;
  void* PJRT_Buffer_UnpaddedDimensions;
  void* PJRT_Buffer_DynamicDimensionIndices;
  void* PJRT_Buffer_GetMemoryLayout;
  void* PJRT_Buffer_OnDeviceSizeInBytes;
  void* PJRT_Buffer_Device;
  void* PJRT_Buffer_Memory;
  void* PJRT_Buffer_Delete;
  void* PJRT_Buffer_IsDeleted;
  void* PJRT_Buffer_CopyToDevice;
  PJRT_Buffer_ToHostBuffer_fn PJRT_Buffer_ToHostBuffer;
  void* PJRT_Buffer_IsOnCpu;
  void* PJRT_Buffer_ReadyEvent;
  void* PJRT_Buffer_UnsafePointer;
  void* PJRT_Buffer_IncreaseExternalReferenceCount;
  void* PJRT_Buffer_DecreaseExternalReferenceCount;
  void* PJRT_Buffer_OpaqueDeviceMemoryDataPointer;
  void* PJRT_CopyToDeviceStream_Destroy;
  void* PJRT_CopyToDeviceStream_AddChunk;
  void* PJRT_CopyToDeviceStream_TotalBytes;
  void* PJRT_CopyToDeviceStream_GranuleSize;
  void* PJRT_CopyToDeviceStream_CurrentBytes;
  void* PJRT_TopologyDescription_Create;
  void* PJRT_TopologyDescription_Destroy;
  void* PJRT_TopologyDescription_PlatformName;
  void* PJRT_TopologyDescription_PlatformVersion;
  void* PJRT_TopologyDescription_GetDeviceDescriptions;
  void* PJRT_TopologyDescription_Serialize;
  void* PJRT_TopologyDescription_Attributes;
  void* PJRT_Compile;
  void* PJRT_Executable_OutputElementTypes;
  void* PJRT_Executable_OutputDimensions;
  void* PJRT_Buffer_CopyToMemory;
  void* PJRT_Client_CreateViewOfDeviceBuffer;
  void* PJRT_Executable_Fingerprint;
  void* PJRT_Client_TopologyDescription;
  void* PJRT_Executable_GetCompiledMemoryStats;
  void* PJRT_Memory_Kind_Id;
  void* PJRT_ExecuteContext_Create;
  void* PJRT_ExecuteContext_Destroy;
  void* PJRT_Buffer_CopyRawToHost;
  void* PJRT_AsyncHostToDeviceTransferManager_Destroy;
  void* PJRT_AsyncHostToDeviceTransferManager_TransferData;
  void* PJRT_Client_CreateBuffersForAsyncHostToDevice;
  void* PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer;
  void* PJRT_AsyncHostToDeviceTransferManager_Device;
  void* PJRT_AsyncHostToDeviceTransferManager_BufferCount;
  void* PJRT_AsyncHostToDeviceTransferManager_BufferSize;
  void* PJRT_AsyncHostToDeviceTransferManager_SetBufferError;
  void* PJRT_AsyncHostToDeviceTransferManager_AddMetadata;
  void* PJRT_Client_DmaMap;
  void* PJRT_Client_DmaUnmap;
  void* PJRT_Client_CreateUninitializedBuffer;
  void* PJRT_Client_UpdateGlobalProcessInfo;
  void* PJRT_TopologyDescription_Deserialize;
  void* PJRT_Client_CreateAliasBuffer;
  void* PJRT_Client_FulfillAliasBuffer;
  void* PJRT_LoadedExecutable_GetDeviceAssignment;
  void* PJRT_Client_CreateErrorBuffer;
  void* PJRT_AsyncHostToDeviceTransferManager_TransferLiteral;
  void* PJRT_Buffer_CopyRawToHostFuture;
  void* PJRT_Device_PoisonExecution;
  void* PJRT_Device_CreateAsyncTrackingEvent;
  void* PJRT_AsyncTrackingEvent_Destroy;
  void* PJRT_Executable_GetCompileOptions;
  void* PJRT_Buffer_DonateWithControlDependency;
  void* PJRT_Event_Create;
  void* PJRT_Event_Set;
  void* PJRT_Device_GetAttributes;
  void* PJRT_Client_Load;
  void* PJRT_LoadedExecutable_AddressableDeviceLogicalIds;
  void* PJRT_Buffer_Bitcast;
  void* PJRT_Error_ForEachPayload;
  void* PJRT_TopologyDescription_Fingerprint;
  void* PJRT_Executable_ParameterMemoryKinds;
  void* PJRT_Device_ClearMemoryStats;
  void* PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace;
  void* PJRT_TopologyDescription_GetMemorySpaceKindIds;
};

/* Layout pins (LP64, matching _pjrt_c_api.py's ctypes translation of the
 * header at commit 70fe66213b73c5953d92eb25d2606bd6004d47c3). */
_Static_assert(sizeof(struct PJRT_Api) == 1144,
               "PJRT_Api layout drift: expected 1144 bytes (3 metadata "
               "fields + 138 function pointers at 8 bytes)");
_Static_assert(offsetof(struct PJRT_Api, pjrt_api_version) == 16,
               "PJRT_Api layout drift: pjrt_api_version offset");
_Static_assert(offsetof(struct PJRT_Api, PJRT_Buffer_ToHostBuffer) == 600,
               "PJRT_Api layout drift: PJRT_Buffer_ToHostBuffer offset");

#ifndef ETL_FAKE_PJRT_STRUCT_SIZE
#define ETL_FAKE_PJRT_STRUCT_SIZE (sizeof(struct PJRT_Api))
#endif
#ifndef ETL_FAKE_PJRT_MAJOR_VERSION
#define ETL_FAKE_PJRT_MAJOR_VERSION 0
#endif

static struct PJRT_Api g_api = {
    .struct_size = ETL_FAKE_PJRT_STRUCT_SIZE,
    .extension_start = NULL,
    .pjrt_api_version =
        {
            .struct_size = sizeof(struct PJRT_Api_Version),
            .extension_start = NULL,
            .major_version = ETL_FAKE_PJRT_MAJOR_VERSION,
            .minor_version = 114, /* PJRT_API_MINOR at HEADER_COMMIT */
        },
    .PJRT_Error_Destroy = fake_PJRT_Error_Destroy,
    .PJRT_Error_Message = fake_PJRT_Error_Message,
    .PJRT_Plugin_Initialize = fake_PJRT_Plugin_Initialize,
    .PJRT_Event_Destroy = fake_PJRT_Event_Destroy,
    .PJRT_Event_Await = fake_PJRT_Event_Await,
    .PJRT_Client_Create = fake_PJRT_Client_Create,
    .PJRT_Client_Destroy = fake_PJRT_Client_Destroy,
    .PJRT_Client_PlatformName = fake_PJRT_Client_PlatformName,
    .PJRT_Client_PlatformVersion = fake_PJRT_Client_PlatformVersion,
    .PJRT_Client_AddressableDevices = fake_PJRT_Client_AddressableDevices,
    .PJRT_Client_Compile = fake_PJRT_Client_Compile,
    .PJRT_Client_BufferFromHostBuffer = fake_PJRT_Client_BufferFromHostBuffer,
    .PJRT_Executable_Destroy = fake_PJRT_Executable_Destroy,
    .PJRT_LoadedExecutable_Destroy = fake_PJRT_LoadedExecutable_Destroy,
    .PJRT_LoadedExecutable_GetExecutable = fake_PJRT_LoadedExecutable_GetExecutable,
    .PJRT_Executable_NumOutputs = fake_PJRT_Executable_NumOutputs,
    .PJRT_Executable_Serialize = fake_PJRT_Executable_Serialize,
    .PJRT_Executable_DeserializeAndLoad = fake_PJRT_Executable_DeserializeAndLoad,
    .PJRT_LoadedExecutable_Execute = fake_PJRT_LoadedExecutable_Execute,
    .PJRT_Buffer_Destroy = fake_PJRT_Buffer_Destroy,
    .PJRT_Buffer_ElementType = fake_PJRT_Buffer_ElementType,
    .PJRT_Buffer_Dimensions = fake_PJRT_Buffer_Dimensions,
    .PJRT_Buffer_ToHostBuffer = fake_PJRT_Buffer_ToHostBuffer,
};

struct PJRT_Api* GetPjRtApi(void) { return &g_api; }
