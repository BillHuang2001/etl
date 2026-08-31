# etl/ir — EvoXIR: the region-based SSA IR

## Intent

EvoXIR is etl's compiler-neutral, region-based SSA intermediate representation
(MLIR-inspired, but not MLIR): every `etl.trace` produces a `Module` of
`Function`s of `Block`s of `Op`s over typed `Value`s. EvoXIR is *the* frontend
IR — StableHLO is an important export target, never the definition. This
directory owns the IR data model, the op-definition registry (full canonical
v1 op set, 105 ops), the `Builder` (op-construction API), `verify`
(structural/type/attribute validation), serialization (the self-describing IR
payload — core of the `.etlgraph` format; the outer container lives in
`../persist`), and `pretty_print`.

**Import rule (binding):** `ir` imports ONLY `etl.core` (Dim/DimExpr, errors),
stdlib, and numpy (constant payloads). Nothing else from etl. `ops`, `trace`,
`backends`, … import from `ir` — never the reverse.

## API surface (must-expose names; re-exported from `__init__.py`)

- SSA structures: `Module`, `Function`, `Region`, `Block`, `Op`, `Value`, `Use`.
- Types/location: `ValueType` (dtype + shape of `int | DimExpr | None`),
  `Location(file, line, col, code_snippet=None)`.
- Effects: `EFFECT_PURE/EFFECT_WRITE/EFFECT_READ/EFFECT_COLLECTIVE/EFFECT_CALLBACK`
  + `EFFECT_KINDS` (see "Effect model").
- `Builder` — op construction; see "Builder" section.
- Op registry: `OpDef`, `AttrSpec`, `opdef(name)` (KeyError if unknown),
  `has_opdef`, `op_names()`, `all_opdefs()`, `register_opdef()`, attr-type
  tags (`ATTR_INT`, …).
- `verify(module) -> None` raising `VerificationError` (owned by `etl.core`,
  re-exported).
- `IR_FORMAT_VERSION` (in `version.py`), `serialize_module(module) -> dict`,
  `deserialize_module(payload) -> Module`.
- `pretty_print(module) -> str`.

## Design decisions

### 1. Value types: reuse core.DimExpr
`ValueType(dtype, shape)`; `shape` is a tuple of `int | DimExpr | None`:
int = static dim, `Dim`/`DimExpr` = symbolic (e.g. batch `B`), `None` =
runtime-dynamic (unchecked; backends declare `dynamic_shapes` capability).
Rank is always known. `ValueType` is IR-level; the trace-level `TensorSpec`
(core) additionally carries device/name and is not used inside the IR. Shape
arithmetic over symbolic dims uses core's `DimExpr`.

### 2. Nested regions (if/while)
Control-flow ops own their bodies as `Op.regions`: `if` holds `(true, false)`,
`while` holds `(cond, body)`. Operand values are passed by binding them to the
entry-block arguments of each region (MLIR style): block-arg count/type must
match operand count/type. v1: every region is single-block (function regions
and nested); multi-block regions are reserved for future versions. `return`
is the v1 terminator (category "terminator") — **added to the canonical set**
because verification requires a terminator and the frontend op list contains
none; the frontend never names it directly.

### 3. Effect model
Effect is declared per-op in its `OpDef` (one of the five kinds). There are no
public effect tokens in v1: ordering is positional — op order inside a `Block`
IS program order for effectful ops. Binding rules for transforms/backends:
pure ops may be reordered/CSE'd but never moved across an effectful op nor out
of their region; effectful ops are never reordered relative to each other,
duplicated, or eliminated; `collective` ops additionally impose
device-synchronization ordering; `callback` ops are opaque side effects.
`write` is reserved for future stateful ops — in v1 only `rank`/`world_size`/
`block_call` use `read`.

### 4. call + multi-function modules vs v1 Graphs
The IR supports N functions per module and the `call` op (callee by name) —
needed for custom blocks and future lowering. v1 `trace.Graph`s build exactly
one function (conventionally `main`) and never emit `call`. `call`'s effect is
declared `pure` at the IR level; a backend supporting `call` must account for
callee effects itself (the v1 numpy backend may simply reject `call`).

### 5. Serialization integrity
`serialize_module` returns a self-describing, JSON-able payload dict:
`{format, version, module, functions, ops, values-by-id, constants (base64
npy), sha256}` — see the schema in `serialize.py`. The sha256 covers the
canonical JSON of the whole payload and is recomputed on load: tampering or
structural violation raises `VerificationError`; unknown format/version raises
`PersistenceError` (both owned by core). `../persist` wraps this payload in
its own container (magic header etc.) — the payload already carries everything
it needs, so the wrapper stays dumb. `IR_FORMAT_VERSION` (IR payload) is
distinct from `ETL_FORMAT_VERSION` (persist container).

Wire-format decisions (binding, documented in `serialize.py`): block op lists
reference the flat ops table via `"ref"` = the op id as a decimal STRING;
dims encode as `{"int"}` | `{"dim": name[, "size"]}` | `{"expr": {op, args}}`
| `null` (decoded back to real `Dim`/`DimExpr`, preserving names/sizes);
ndarray attr values move to the `constants` table (attr becomes
`{"__etl_ndarray__": key}`); tuples are wrapped as `{"__etl_tuple__": [...]}`
to survive the JSON round-trip; `runtime_call`/`block_call` `result_specs`
are encoded as a list of type dicts and NORMALIZED back to `ValueType`
instances on decode (so `verify`'s in-memory comparison still passes).
Rebuilt modules keep their ORIGINAL ids and their `_op_ids`/`_value_ids`
counters are fast-forwarded past the payload maximum, so post-load
`Builder.create` calls never collide ids. `serialize_module` runs `verify`
first; `deserialize_module` runs it on the result.

### 6. Result-type resolution
`OpDef.shape_fn: (input_types, attributes) -> result_types` computes result
types when the builder has no explicit `result_types`. `shape_fn=None` means
op-specific resolution by the Builder (`constant`: from payload; `call`: from
the callee signature; `if`: from region terminators; `runtime_call`/
`block_call`: from declared specs) — or explicit `result_types=` at the call
site. `verify` enforces agreement between recorded result types and the op's
contract.

### 7. Identity & ownership
`Value.id`/`Op.id` are module-unique, assigned by the Builder from the owning
Module's counters (stable ids for serialization). Parent pointers
(`op.parent`, `region.parent`, `fn.parent`) are wired by the constructing API;
`verify` checks consistency. `Value.uses` is maintained by the Builder and by
`Value.replace_all_uses_with`.

## Op registry (canonical v1 set — 105 ops, all declared in `op_defs/`)

| File | Category | Ops | Effect |
|---|---|---|---|
| elementwise.py | elementwise | add subtract multiply divide power remainder maximum minimum; abs negate square sqrt exp log log1p sin cos tan tanh sigmoid relu gelu erf sign; logical_and logical_or logical_not; bitwise_and bitwise_or bitwise_xor; cast nan_to_num | pure |
| elementwise.py | comparison | equal not_equal less less_equal greater greater_equal (result bool) | pure |
| structure.py | structure | select broadcast reshape transpose slice gather scatter concatenate pad tile flip roll diag | pure |
| reduction.py | reduction | reduce_sum reduce_max reduce_min reduce_mean reduce_prod argmax argmin cumsum cumprod | pure |
| sorting.py | sorting | sort argsort | pure |
| linalg.py | linalg | dot conv tril triu solve | pure |
| control.py | control | constant stop_gradient if while call runtime_call block_call | pure; runtime_call=c**allback**; block_call=read |
| control.py | terminator | return | pure |
| collective.py | collective | all_reduce all_gather reduce_scatter all_to_all broadcast_collective collective_permute | collective |
| collective.py | collective | rank world_size (scalar int64) | read |
| sparse.py | sparse | sparse_from_dense sparse_to_dense sparse_coo_to_csr sparse_csr_to_coo sparse_coo_to_csc sparse_csc_to_coo sparse_negate sparse_add sparse_multiply sparse_multiply_dense sparse_reduce_sum sparse_transpose sparse_reshape sparse_concatenate sparse_dot_dense dense_dot_sparse | pure |
| random.py | random | random_key_mix random_uniform random_normal random_randint random_permutation random_multinomial | pure |

**Multi-algorithm random framework (binding):** `op_defs/random.py` is the
SINGLE source of truth for the canonical algorithm names — `ALGORITHMS =
("splitmix64", "threefry2x32", "philox4x32_10")`, `DEFAULT_ALGORITHM =
ALGORITHMS[0]` (splitmix64, the v1 default), `validate_algorithm(name)` and
`algorithm_key_type(name) -> (shape, dtype)` — imported by `ops` and the
backend writers (never duplicate the literals). Key types: splitmix64 →
rank-0 int64; threefry2x32 → shape (2,) int32; philox4x32_10 → shape (4,)
int32. All 6 random OpDefs carry the shared `algorithm` attribute
(`AttrSpec("algorithm", ATTR_STR, default=DEFAULT_ALGORITHM)`); the frontend
stamps it explicitly once multi-algorithm support lands — until then the
default applies and behavior is unchanged. `inference.py` validates key
shape/dtype against the op's `algorithm` in `_check_random_key` (key form of
a different algorithm → `ShapeError` naming both; a form matching no
algorithm → `ShapeError` naming the three accepted forms; unknown algorithm
name → `ValueError` via `validate_algorithm`); `infer_random_key_mix`'s
result type is the algorithm's key type via `algorithm_key_type`. Hooks read
the attribute with `attributes.get("algorithm", DEFAULT_ALGORITHM)` so
graphs serialized before the attribute existed still verify with the default.

Declaring an op here does NOT mean every backend implements it — backends
reject unsupported ops explicitly via capabilities, never silently. IR name
note: the collective is `broadcast_collective` (the shape op `broadcast`
already owns that name).

**15-op batch (design notes):** `topk` is a frontend composition over
sort/argsort + gather (no IR op; static `k` ≤ extent → `ShapeError`, symbolic
extent → runtime error). `stack` = reshape + concatenate composition. `flip`/
`roll` are dedicated IR ops with np-exact kernels. `clamp` = maximum/minimum
composition (scalar bounds pre-cast to x's dtype when same_kind-castable; a
float bound on an int tensor → float64 promotion — numpy 2.x parity, deviation
vs numpy 1.x's TypeError; documented edge: int bound on uint tensor falls back
to weak promotion int64). `eye`/`linspace` = Constant-op compositions
(`linspace` float64 default — deliberate deviation from the etl float32
creation convention, explicit dtype param; symbolic bounds → `TraceError`, v2
deferral). `matmul` = frontend sugar over dot with rank-1 promote/squeeze
(dot's rank ≥ 2 contract and `__matmul__` → dot unchanged). No vjp/batching
rules for the 8 new IR ops + eye/linspace → `TransformError` (the random-op
pattern); clamp/matmul/isnan/stack/topk inherit their composition's rules.
Numpy backend: full reference. Compiler backends (stablehlo/iree/xla/tvm)
defer the 8 new IR ops with explicit `BackendError` (no exporter entries
added); compositions work via their components.

## Builder (op-construction API)

`Builder.create(op_name, operands=(), attributes=None, result_types=None,
location=None, regions=()) -> Op` — creates result `Value`s (ids from module
counters; types via `shape_fn` or explicit `result_types`), wires operand
`Use`s and parent pointers, validates arity/attrs/regions eagerly.
`emit(...) -> Value` is the single-result convenience (raises unless exactly
one result). Helpers: `set_insertion_point(block_or_region)`,
`push_region/pop_region`, `build_region(input_types) -> Region` (entry block +
args — the body of an upcoming if/while), `insert_block(region,
position=None)`, `set_terminator(block, op_name, ...)`,
`build_module(name, metadata=None)`, `build_function(name, input_types,
metadata=None)`. The frontend `ops` module obtains the active builder from
`trace.current_builder()`; ALL IR mutation funnels through Builder.

**Implementation notes (binding for `verify` agreement):** result types
resolve in order — explicit `result_types` → `OpDef.shape_fn` → op-specific
rules (`constant`: dtype/shape from the payload; `call`: callee output
signature; `if`: operand types of both branches' `return` terminators, which
must agree; `runtime_call`/`block_call`: `result_specs` entries — `ValueType`
| core `TensorSpec` | `{"dtype", "shape"}` dict — converted to `ValueType`);
anything else with `shape_fn=None` demands explicit types. A `ShapeError`
from a `shape_fn` propagates with the op's source location appended to its
message (when a real location is known — never for `Location.unknown()`).
NOTE: the `result_specs` conversion applies
to result-TYPE RESOLUTION only — the stored attribute is NOT rewritten, and
`verify` requires the stored `result_specs` to be a sequence of `ValueType`
instances. Frontends must pass `ValueType`s (not `TensorSpec`/dict entries)
to pass verification. Attributes are validated eagerly against the
schema (unknown keys / missing required / wrong tag → `VerificationError`),
stored as a copy with defaults applied: `dtype` values normalize to the
dtype-name string, sequence tags normalize to tuples, `int` tags accept `None`
only where the spec's `default is None`. Region count must equal
`OpDef.regions` exactly. `set_terminator` appends (last by construction),
never uses the insertion point, and rejects non-terminator names and blocks
that already have a terminator.

## verify(module)

First-violation-wins, `VerificationError` with source location when available.
Full invariant list is the docstring of `verify` in `verify.py`; summary:
module version + unique function names; function: single-block region (v1),
block args match `input_types`, terminator is `return` and last; block/region:
parent wiring, nested-region counts and entry-arg binding per `OpDef.regions`;
op: registered name, arity/result-count/region-count within OpDef, attributes
match schema (no unknown keys, required present, types tagged correctly),
result types agree with `shape_fn` (when not None); value/SSA: unique ids,
operands defined before use (dominance), `Use` bookkeeping consistent.

## Serialization schema (payload dict)

See the docstring in `serialize.py` (the full schema). Summary: `{format:
"etl-ir", version, module, functions, ops, constants, sha256}`; values are
defined inline by ops (`results`) and blocks (`arguments`) with module-unique
ids, operands referenced by id; dims encode as `{"int": n}` | `{"dim": name}`
| `{"expr": {"op", "args"}}` | `null`; dtypes as numpy `dtype.name` strings;
constant payloads as base64 `np.save` bytes in the `constants` table; sha256
over canonical JSON (`sort_keys`, compact separators) of everything else.

## pretty_print

Readable SSA text, one op per line: `%0 = etl.add(%arg0, %arg1) :
tensor<BxNxf32> loc("model.py":12:8)`. Block args print as `%argN`; results
numbered sequentially per function; nested regions indent inline. Format spec
in `printer.py`.

## Constraints

- Import rule above; nothing from `ops`/`trace`/`backends`/etc.
- Implementation status: data structures, registry, shape-inference hooks
  (`inference.py`, 49 hooks), `pretty_print`, `verify`, the `Builder`, and
  serialization (`serialize_module`/`deserialize_module`) are implemented.
- Shape-inference conventions (binding for `verify` agreement): broadcasting
  resolves symbolic conflicts as `DimExpr("max", a, b)` (left dim first);
  `None` dims are runtime-dynamic and yield `None`; element-count checks
  raise `ShapeError` only on definite mismatch; sum folds are
  left-associative `DimExpr("add", ...)` chains. Dtype promotion uses
  `np.result_type`; reduction dtypes follow numpy per `reduce_op` (the
  `reduce_*` OpDefs carry a `reduce_op` attribute: 'sum'|'max'|'min'|'mean'|
  'prod').
- Files < ~1000 lines; op_defs split by category (done).
- Unknown op name in `opdef()` → `KeyError`; duplicate registration →
  `ValueError`; version/format mismatch on deserialize → `PersistenceError`;
  structural violations → `VerificationError`.
- No `numpy()`, storage, device, or DLPack on any IR object — the IR is pure
  structure (values are references, not data).

## Routing table

| Path | Area |
|---|---|
| `./op_defs/` | OpDef/AttrSpec, registry, category tables (elementwise, comparison, structure, reduction, sorting, linalg, control, collective, sparse, random) |
| `./value.py`, `./op.py`, `./block.py`, `./region.py`, `./function.py`, `./module.py` | SSA data model |
| `./types.py`, `./location.py`, `./effects.py`, `./version.py` | Small shared definitions |
| `./inference.py` | Shape-inference hooks referenced by OpDefs (49 hooks, implemented; ~1644 lines — legitimately long, one hook module for all categories; split only if it grows much further) |
| `./op_defs/sparse.py` | Sparse op defs: 16 ops (from_dense/to_dense, coo/csr/csc conversions, negate, add, multiply, multiply_dense, reduce_sum, transpose, reshape, concatenate, dot variants), all pure |
| `./builder.py` | Op-construction API (implemented) |
| `./verify.py` | Structural/type/attribute verification (implemented) |
| `./serialize.py` | IR payload serialization: self-describing payload, sha256 integrity, round-trip rebuild with original ids (implemented) |
| `./printer.py` | SSA text printing (implemented) |

Sibling: `../tests/` → test suite (read-only from here; escalate test-related
writes to root).

## Test strategy (Phase 2+)

pytest under `../tests/ir/`: registry completeness (every public op in the etl
contract is declared; arities/attrs/effects correct); shape-inference hooks
(static + symbolic dims per hook, broadcast/reshape/conv formulas, ShapeError
cases); `verify()` invariants
(valid modules + one test per violation class); serialization round-trips
(shapes with symbolic dims, constants as base64 npy, tamper detection, version
rejection); `pretty_print` golden output; Builder wiring (uses, ids, regions,
terminators). CPU only.

## Known Issues

- **Generic inference + ops-side compensation (sanctioned division of
  labor):** the registry keeps generic hooks — `divide` →
  `infer_elementwise_binary` (plain `np.result_type`, int/int stays int),
  unary math ops (`sqrt`, `exp`, `log`, `log1p`, `sin`, `cos`, `tan`,
  `tanh`, `sigmoid`, `relu`, `gelu`, `erf`) → `infer_elementwise_unary`
  (dtype-preserving), `cumsum` → `infer_identity` — while `../ops` achieves
  its binding dtype rules (true division int/bool → float64; int/bool →
  float64 for unary math; bool cumsum → int64) by composing explicit
  pre-cast ops — documented in `../ops/CONTEXT.md` as sanctioned transparent
  sugar, not a bug. `abs` has its own dedicated `infer_abs` hook (complex →
  real magnitude dtype), so no compensation is needed there. If dedicated
  hooks (`infer_true_divide`, `infer_unary_math`, `infer_cumsum`) are ever
  added here, the ops compositions may be removed — but only when registered
  in `inference.py` + `op_defs/` (the numpy backend kernel must agree on
  either form).
- Builder attribute schema is stricter than inference in two places:
  `pad.padding_config` (ATTR_NESTED_INTS) requires `(lo, hi)` PAIR entries
  (bare int entries fail `VerificationError`, though `infer_pad` accepts
  them); `slice.limit_indices` is schema-required and can be neither `None`
  nor contain `None` entries (the None-limit branch of `infer_slice` is
  unreachable via the Builder).

## Status

Phase 2 complete for this directory: SSA data model, op registry (105 ops),
shape-inference hooks (`inference.py`, 49 hooks), `pretty_print`, `verify`
(the full invariant set — module/function/region/op/value levels, SSA
dominance, use bookkeeping, shape_fn result-type agreement), the `Builder`,
and serialization (`serialize_module`/`deserialize_module` — payload schema,
integrity hash, round-trip rebuild with original ids and fast-forwarded
counters) are implemented.
