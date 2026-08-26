# etl/ir — EvoXIR: the region-based SSA IR

## Intent

EvoXIR is etl's compiler-neutral, region-based SSA intermediate representation
(MLIR-inspired, but not MLIR): every `etl.trace` produces a `Module` of
`Function`s of `Block`s of `Op`s over typed `Value`s. EvoXIR is *the* frontend
IR — StableHLO is an important export target, never the definition. This
directory owns the IR data model, the op-definition registry (full canonical
v1 op set, ~75 ops), the `Builder` (op-construction API), `verify`
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

## Op registry (canonical v1 set — 75 ops, all declared in `op_defs/`)

| File | Category | Ops | Effect |
|---|---|---|---|
| elementwise.py | elementwise | add subtract multiply divide power remainder maximum minimum; abs negate square sqrt exp log log1p sin cos tan tanh sigmoid relu gelu erf sign; logical_and logical_or logical_not; bitwise_and bitwise_or bitwise_xor; cast | pure |
| elementwise.py | comparison | equal not_equal less less_equal greater greater_equal (result bool) | pure |
| structure.py | structure | select broadcast reshape transpose slice gather scatter concatenate pad | pure |
| reduction.py | reduction | reduce_sum reduce_max reduce_min reduce_mean reduce_prod argmax argmin cumsum | pure |
| linalg.py | linalg | dot conv tril triu solve | pure |
| control.py | control | constant stop_gradient if while call runtime_call block_call | pure; runtime_call=c**allback**; block_call=read |
| control.py | terminator | return | pure |
| collective.py | collective | all_reduce all_gather reduce_scatter all_to_all broadcast_collective collective_permute | collective |
| collective.py | collective | rank world_size (scalar int64) | read |

Declaring an op here does NOT mean every backend implements it — backends
reject unsupported ops explicitly via capabilities, never silently. IR name
note: the collective is `broadcast_collective` (the shape op `broadcast`
already owns that name).

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
- Architecture phase: data structures, registry, shape-inference hooks
  (`inference.py`, 23 hooks), and `pretty_print` are implemented; the
  remaining behavioral bodies (`Builder`, `verify`,
  `serialize_module/deserialize_module`) raise `NotImplementedError` —
  Phase 2 (implementation) fills them.
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
| `./op_defs/` | OpDef/AttrSpec, registry, category tables (elementwise, structure, reduction, linalg, control, collective) |
| `./value.py`, `./op.py`, `./block.py`, `./region.py`, `./function.py`, `./module.py` | SSA data model |
| `./types.py`, `./location.py`, `./effects.py`, `./version.py` | Small shared definitions |
| `./inference.py` | Shape-inference hooks referenced by OpDefs (23 hooks, implemented) |
| `./builder.py` | Op-construction API (stub) |
| `./verify.py` | Structural/type/attribute verification (stub) |
| `./serialize.py` | IR payload serialization (stub) |
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

## Status

Phase 2 in progress: SSA data model, op registry (75 ops), shape-inference
hooks (`inference.py`, 23 hooks), and `pretty_print` are implemented. The
remaining behavioral bodies (Builder, verify, serialize) are
`NotImplementedError` stubs being filled by Phase 2 (delegated to a Manager).
