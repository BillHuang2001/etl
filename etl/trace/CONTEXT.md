# etl/trace — tracing machinery (defn, trace, Graph, control flow)

## Intent

The tracer: turns a plain Python function into an EvoXIR graph by executing
it ONCE under an active `ir.Builder`. Owns `@etl.defn`, `etl.trace`, the
`Graph` type (IR module + structured I/O trees), the active-builder context
(the hook `etl.ops` uses), and runtime tensor control flow (`cond` /
`while_loop` / `scan`, traced into `if`/`while` IR regions).

Principles honored here (see root `../../CONTEXT.md`): **no implicit eager**
(`Defn` calls always raise), **explicit staging** (`trace` never
verifies/compiles), **Python value → Python semantics** (static values
specialize the graph; only tensor values need explicit control flow), and
**no magic** (no caching, no hidden re-traces).

Architecture rule for this module: **all trace-time state lives in one
`_TraceSession` object; there is no global state besides the contextvars
builder stack.** See "Data structures & ownership" below — that section is
the map you need before touching any file here.

## API Surface

Exported from `__init__.py` (the public contract — exact names, unchanged
by the 2025-08 cleanup refactor):

- `defn(fn=None, **options) -> Defn` — decorator (bare or with options).
  `Defn` stores `fn` (+ `__etl_defn__` marker on both), `options` dict
  (reserved for future defn-compiler config; ignored in v1). Calling a
  `Defn` ALWAYS raises `core.TraceError` directing to
  `etl.trace(defn, *specs)` / `etl.evaluate(defn, *args)`. Idempotent.
- `trace(fn_or_defn, *specs) -> Graph` — THE tracer (algorithm below).
  Accepts `Defn`, `__etl_defn__`-marked objects, or plain callables.
- `Graph` — attrs: `module` (`ir.Module`), `input_specs` (`core.TreeSpec`
  whose leaves are the original `TensorSpec`/static objects),
  `tensor_specs` (tuple of `TensorSpec` in block-arg order),
  `output_tree` (`core.TreeSpec`), `static_values` /
  `output_static_values` (tuples of `StaticValue` records: index, path,
  value, kind), `source_locations` (dict: input value ids →
  `ir.Location`). Methods: `print()`, `verify()`, `save(path)` /
  `load(path)` (classmethod; lazy-import `etl.persist`),
  `flatten_inputs(args) -> list[core.Tensor]`, `validate_inputs(args)`
  (alias), `unflatten_outputs(flat) -> structured`, `signature_info()`.
  Constructor accepts a prebuilt module + trees (transforms build Graphs
  directly).
- `current_builder() -> ir.Builder` — innermost active builder; raises
  `core.TraceError` when no trace is active (this is `etl.ops`'s hook).
- `with_builder(builder)` — nestable context manager installing the active
  builder; alias `builder_stack`.
- `cond(pred, true_fn, false_fn, *operands, **static_kwargs) -> outputs`
- `while_loop(cond_fn, body_fn, init) -> outputs`
- `scan(f, init, xs, length=None) -> (carry, stacked_outputs)`
- `StaticValue` — frozen dataclass record for static leaves.

## Module layout & routing table

Flat module — no child directories. Responsibility split (single owner per
concern):

| File | Owns |
|---|---|
| `defn.py` | the `Defn` marker decorator (nothing else) |
| `builder.py` | the active-builder context (contextvars tuple stack) + `_return_terminator` (the canonical block-ending op) |
| `_tree.py` | ALL shared pytree/static helpers: `_is_static_value`, `_registered_pytree_base`, `_flatten`/`_flatten_into` (parameterized leaf policy), `_iter_leaf_paths`, `_to_symbolic`, `_PYTREE_NODE_REGISTRY` alias. Imports core/ir ONLY — any trace module may import it. |
| `trace.py` | the 7-step tracer: `_TraceSession`, input classification, output classification, `_trace_call_site`. Re-exports `_format_path` / `_iter_leaf_paths` for `etl.transforms.grad` (private cross-module import — keep the names and paths stable). |
| `graph.py` | the `Graph` type: I/O validation, `StaticValue` records, `_static_record` factory, persistence delegation. Also exports `_normalize_leaf_types` (imported by `etl.transforms.vmap` — keep stable). |
| `control_flow.py` | `cond`/`while_loop`/`scan`: region building via the `ir.Builder` (`_run_in_region`), operand classification (`_classify_operands`), registered-node rules. NEVER imports `etl.ops` (ops imports trace — the DAG stays acyclic). |
| `__init__.py` | re-exports only + the shadowing note |

Sibling: `../tests/` → test suite (read-only from here; escalate test
writes to the package root).

Private cross-module imports that MUST keep working (do not rename/move
without updating consumers): `etl.pipeline` imports
`_SymbolicLeaf`/`_TensorSpecLeaf` from `etl.trace.trace`;
`etl.transforms.grad` imports `_format_path`/`_iter_leaf_paths` from
`etl.trace.trace`; `etl.transforms.vmap` imports `_normalize_leaf_types`
from `etl.trace.graph`.

## Data structures & ownership (the map)

**`_TraceSession`** (`trace.py`) — the single named owner of one `trace()`
call. Fields: `builder`/`module`/`function` (the IR under construction; ONE
builder serves the whole trace — control flow re-positions it via
`push_region`/`pop_region`, never creates a second), `input_tree`/
`tensor_specs`/`static_values` (the classified input contract),
`symbolics` (block-arg SymbolicTensors, one per tensor spec),
`source_locations` (input value id → trace call-site), `args` (the
reconstructed call arguments), `call_site` (`ir.Location`). Lifecycle maps
1:1 onto the trace algorithm: `open(specs)` = steps 2–4, `run(fn)` = step
5, `finish(outputs)` = steps 6–7. One session per call — traces are never
cached or reused.

**Builder context** (`builder.py`) — a `contextvars.ContextVar` holding an
immutable tuple stack (outermost first). `with_builder` copies the tuple on
push and restores the saved parent on exit (even on error) — never mutated
in place, so nested contexts and async tasks cannot corrupt each other.
`current_builder()` returns the innermost builder or raises `TraceError`.

**Leaf markers & static records** — trace trees record tensor positions as
plain non-dataclass markers (`_TensorSpecLeaf` carrying the `TensorSpec`,
`_SymbolicLeaf` carrying the `SymbolicTensor`) so leaf-type equality
distinguishes tensor leaves from static leaves while the TreeSpec stays
`core.unflatten`-compatible. Static leaves are recorded as the object
itself with `TreeSpec.type = type(obj)` (pipeline's `_is_tensor_leaf_spec`
relies on this: any non-marker, non-None leaf type = static position).
`StaticValue` (`graph.py`) is the run-time validation record: frozen
dataclass with `index` (flat leaf index), `path`, `value`, `kind`
(`type(value).__qualname__` — so a recorded `1` never matches run-time
`True`). The single construction site is `graph._static_record(index, path,
value)`.

**The shared walker** (`_tree.py::_flatten`/`_flatten_into`) — one pytree
walk with two knobs so every caller's leaf convention stays explicit
instead of being copy-pasted:
- `leaf_spec(obj) -> (recorded_leaf, TreeSpec) | None` — decides which
  objects are leaves and what gets appended (trace: marker leaves; control
  flow: plain leaves for `_LEAF_TYPES`).
- `plain_leaf_type(obj) -> type` — the `TreeSpec.type` for fallback leaves
  (objects the policy declined that are not containers). trace passes
  `type` (static leaves record their own Python type — pipeline depends on
  it); control flow uses the default constant `None`.
Container descent mirrors `core.tree._flatten_into` (registered nodes via
MRO walk first, namedtuple before tuple, user dataclasses — etl-module
dataclasses stay leaves, tuple/list/dict with sorted keys). Deliberate
difference: no `defaultdict`/`Counter` default-factory handling in
`node_data` — that is a `core.flatten`-only feature.

## Trace algorithm (binding contract)

1. Unwrap `Defn`/`__etl_defn__` → plain fn (`_unwrap_defn`).
2. Treat `specs` (positional args) as ONE pytree; flatten via the shared
   walker with the trace leaf policy. Leaves must be `TensorSpec` (tensor
   input; shape may hold `Dim`/`DimExpr`, `None` = dynamic) or static
   Python values (`None`/bool/int/float/complex/str/`Enum`/numpy `dtype`/
   `slice`/`core.Dim`/`core.DimExpr`/`core.Device` — see `_is_static_value`).
   Anything else (concrete `Tensor`, ndarray, `SymbolicTensor`, …) →
   `TraceError` with the pytree path. Capturing a concrete tensor is NEVER
   silently allowed.
3. Build `ir.Module` + entry `ir.Function` "main" with one block arg per
   tensor leaf (arg type = (shape, dtype)). Wrap each block arg as
   `core.SymbolicTensor`; record trace-call-site `ir.Location`s.
4. Reconstruct args (unflatten: SymbolicTensors at tensor positions,
   original static values at static positions).
5. Call `fn(*args)` ONCE under `with_builder(builder)`. Normal Python
   execution — static control flow specializes the graph; ops build IR via
   `current_builder()`. Closure-captured concrete tensors fail inside ops
   (ops' `TraceError`) — the tracer does not pre-scan closures.
6. Flatten outputs (pytree). `SymbolicTensor` leaves → results; static
   leaves → `output_static_values` (re-inserted by `unflatten_outputs`);
   anything else → `TraceError`. Emit the function `return` terminator with
   the symbolic leaves (zero-result graphs are legal).
7. Return `Graph(...)` — NOT verified automatically (staging explicit;
   `etl.build` verifies). Every `trace` call yields a NEW Graph.

## Static-value snapshot semantics

Closure static values are naturally snapshotted because the function body
runs once at trace time — the values are read when ops are built, and only
explicit tensor values ever reach the IR. Run-time graph specialization is
validated by `Graph.flatten_inputs` (static leaves must match recorded
type+value; dtype/shape/device checked against specs with
`DTypeError`/`ShapeError`/`DeviceError`; tree mismatch → `TraceError`).
numpy arrays are accepted at run time and wrapped via `core.from_numpy`
(documented convenience).

## Graph layout

Entry `ir.Function` "main": block args = tensor inputs in `tensor_specs`
order; terminator `return` yields results in output-tree SymbolicTensor-leaf
order. `input_specs`/`output_tree` carry the structured I/O contract
(tuple/list/dict/namedtuple/dataclass — pytree machinery from `core`).
Static leaves live ONLY in `static_values`/`output_static_values` records,
never in the IR. `unflatten_outputs` interleaves tensor results and static
records in ONE pass over the combined leaf positions (validates indices in
range, no duplicates).

## Control-flow region conventions (binding — implemented against the real `etl/ir` registry)

Region ops are built via `ir.opdef(name)` + the `ir.Builder` — this module
NEVER imports `etl.ops` (ops imports trace — keep the DAG acyclic). The
conventions below are what the IMPLEMENTED `ir` registry + `verify` enforce
(they differ from earlier architecture assumptions — `ir` is authoritative):

- **Region execution** is unified in `_run_in_region(builder, region, fn,
  *args, after=None)` (`control_flow.py`): it pushes the region's entry
  block onto the builder's insertion-point stack (creating the block first
  if the region is empty), installs the builder as the active builder for
  `current_builder()` (the SAME builder — control flow never creates a
  second), runs `fn`, then calls `after(result)` still inside the region
  (callbacks emit the `return` terminator via `_return_terminator`), and
  pops the region in a `finally` so the stack stays consistent on error.
- `if` op ("if", effects none, 2 regions, arity (1, None)): **operand 0 IS
  the boolean predicate**; operands 1..n are the captured tensor values.
  `ir.verify` binds EVERY operand (predicate included) to each region's
  entry-block args — one arg per operand, identical count and types. Branch
  callables therefore receive the REGION BLOCK ARGS (wrapped as
  SymbolicTensor), not the enclosing captured values. Each block's `return`
  terminator yields the branch outputs (identical trees, unified
  dtypes/shapes across branches — mismatch raises `DTypeError`/`ShapeError`/
  `TraceError`); explicit `result_types` are passed to `create`. Branch
  callables run once each at trace time.
- `while` op ("while", effects none, 2 regions, arity (1, None)):
  operands = n initial carried values (≥1 — all-static init raises
  `TraceError`); regions = condition, body — one block each with n block args
  bound to the operands (same verify convention). `shape_fn=infer_identity`
  → op results = operand types. Condition region returns ONE 0-d bool; body
  region returns n next-carried values. Loop-carried types must stay constant
  across iterations (checked at trace time). cond_fn/body_fn run once each.
- `return` op: terminator-only (operands = yielded values), emitted via
  `_return_terminator(builder, values)` (`builder.py`) which delegates to
  `Builder.set_terminator` — it APPENDS as the block's last op (never via
  `create` at the insertion point).
- Registered pytree nodes (types registered via `core.register_pytree_node`,
  e.g. sparse tensors) are valid carried values / branch outputs /
  scan carries: the shared walker consults the registry FIRST (MRO walk over
  `core.tree._PYTREE_NODE_REGISTRY`), the node's registered `flatten_fn`
  yields its children (SymbolicTensor leaves + static leaves) which recurse
  and classify as usual, and reconstruction goes through `core.unflatten`
  with the registered (polymorphic) `unflatten_fn` — tensor children become
  region block args / op results, static children pass through unchanged.
- `scan` desugaring uses these registered op defs (raw, via `ir.opdef`):
  `constant` (int32 0/1/length scalars), `add` (counter increment), `less`
  (counter < length), `gather` (`axes=(0,)` with a 0-d int32 counter — the
  DYNAMIC index step; `slice` is static-int-only and unusable here),
  `reshape` (leading 1-dim), `broadcast` + `scatter` (fixed-size stacking —
  a grow-by-`concatenate` stack would change the carried type per
  iteration, which a typed `while` op cannot carry). The registry has NO
  `expand_dims`. Step 0 runs pre-loop via a static `slice` at index 0;
  `length == 0` raises `TraceError` (v1).

`scan` v1 scope: STATIC length only (an explicit static `int`, or derived
from xs's static leading dim). An explicit static `length` SMALLER than a
statically-known leading dim is allowed and runs a PREFIX scan over the
first `length` elements (stacked outputs have shape `(length, ...)`); only
an explicit `length` LARGER than a statically-known leading dim raises
`TraceError`. Symbolic length → `TraceError` (dynamic-scan region ops
reserved; no silent fallback).

## Caching & performance (what the 2025-08 cleanup did, where the time goes)

Measured after the cleanup (same machine, `$TMPDIR/etl_trace_bench.py`):
chain (~100-op elementwise+reduce defn) 47.4 ms/trace (was 48.7), MLP
5-input 1.33 ms/trace (was 1.35), control-flow while+cond 3.30 ms/trace
(was 3.32). Modest — expected, because tracing is NOT the bottleneck.

Where tracing time actually goes (~89% on the chain benchmark):
`inspect.stack()` inside `etl/ops/_utils.py::get_location` (~0.48 ms per
op). OUT OF SCOPE for this module — it is ops' source-location capture. The
escape hatch `ETL_DISABLE_LOCATIONS=1` exists (set it to skip location
capture entirely); a `sys._getframe`-based walk or a per-trace location
cache would remove most of it — flagged to the package root.

Caching that EXISTS in this module (all semantic-neutral dedup):
- `_classify_operands` (`cond`): every container operand is flattened
  EXACTLY once per `cond` call (was ~5×: validation + operand list + once
  per branch); `_leaf_registered_flags` is computed once per `cond` call.
- `Graph.unflatten_outputs`: single pass interleaving tensor results and
  static records (was O(n·s) repeated list inserts).
- `_return_terminator`/`_run_in_region`: one canonical spelling each for
  block-ending and region-running (dedup, not speed).

Caching deliberately NOT added: per-`Graph` input-validation plans
(`flatten_inputs` plan caching). The dominant per-call cost is the input
pytree walk itself, which a plan cannot remove, and `etl.pipeline`
already caches input/output plans per `Executable`/`BoundExecutable` — a
second cache here would just duplicate that. Traces themselves are never
cached (every `trace()` call builds a fresh Graph — that is the contract).

## Error cases (all public errors derive from `core.ETLError`)

`Defn.__call__` (always), `current_builder()` outside trace, invalid trace
input/output leaves, branch tree/arity mismatch or non-scalar-bool `pred`/
`cond_fn` result, static-kwargs that are not static, symbolic scan length
(v1) — all → `core.TraceError` with pytree path / graph location where
applicable. `flatten_inputs`: `ShapeError` (shape vs DimExpr), `DTypeError`,
`DeviceError`, `TraceError` (tree/static mismatch). `Graph.load` mismatch →
`PersistenceError` (from persist). `verify()` → `VerificationError`.
Error messages are byte-stable (the test suite matches substrings) — when
editing messages, keep the wording of existing paths intact.

## Constraints

- Imports: `etl.core` and `etl.ir` ONLY (plus stdlib/numpy). `etl.persist`
  is imported lazily inside `Graph.save/load` (persist does not import
  trace). NEVER import `etl.ops` — ops imports trace for
  `current_builder()`; control flow builds raw ops via `ir.opdef`.
- Files < ~1000 lines; current layout: `defn.py`, `builder.py`, `_tree.py`,
  `graph.py`, `trace.py`, `control_flow.py`, `__init__.py`.
- All `trace` outputs are ordinary ops — backends need no control-flow
  runtime support (numpy interpreter selects regions/loops natively).
- No caching, no eager execution, no silent fallbacks anywhere in this
  module.

## Design decisions

- **`_TraceSession` owns all trace state.** The alternative (module-level
  globals, or threading a state object through every helper) either leaks
  state across nested/reentrant traces or buries it in signatures. The
  dataclass makes the trace lifecycle visible: `open`/`run`/`finish` map
  1:1 onto the documented algorithm steps.
- **The leaf policy is a parameter, not a copy.** `trace.py` and
  `control_flow.py` previously shipped separate pytree walkers that drifted
  apart (they disagreed about which objects are leaves). `_tree.py`
  parameterizes the ONLY thing that differed. The two knobs
  (`leaf_spec` + `plain_leaf_type`) are explicit because the leaf
  conventions are load-bearing for `etl.pipeline` (`_is_tensor_leaf_spec`
  reads `TreeSpec.type`).
- **A fully-static dataclass config object passed as a trace argument is
  legitimate static specialization, not an error.** Dataclass instances are
  pytree containers: the tracer descends into them, and a config whose
  fields are all static Python values specializes statically at trace time —
  each static leaf is recorded in `static_values` and validated at run time
  via `Graph.flatten_inputs`. (A dataclass containing non-static,
  non-`TensorSpec` leaves still raises `TraceError` per the trace algorithm
  above.)
- **Markers are plain classes, not dataclasses.** `_TensorSpecLeaf` /
  `_SymbolicLeaf` are deliberately not dataclasses (no `__eq__`): identity
  is enough, and `TreeSpec` equality never has to compare their payloads.

## Known warts

- **`get_location` is ~89% of trace time** (`etl/ops/_utils.py`,
  `inspect.stack()` per op). Escape hatch: `ETL_DISABLE_LOCATIONS=1`. Fix
  belongs in ops/ (see "Caching & performance").
- **`Graph.input_specs` is a TreeSpec, not "specs".** The name predates the
  input_tree concept; it is kept for API stability. New code should read
  `tensor_specs` + `static_values` for leaf-level data.
- **`StaticValue` (this module) collides in name with
  `etl.block.decl.StaticValue`.** Different types, both public-ish; do not
  import one where the other is expected. (No import conflict — separate
  modules.)
- **Marker-leaf privates are cross-module imports.** `etl.pipeline`
  imports `_TensorSpecLeaf`/`_SymbolicLeaf` by name — they must stay at
  `etl.trace.trace` with those names (see "Module layout").
- **`etl.trace` is the function, not the submodule**, after `import etl`
  (the package re-exports the function under that attribute). Use
  `from etl.trace import ...` (works — the import system consults
  `sys.modules` first) or `sys.modules["etl.trace"]` for the module object.
- **`core.TreeSpec.context` (registered-node aux) is invisible to
  `core.first_mismatch_path`** — aux-bearing registered nodes get no
  automatic run-time aux-equality validation via `Graph.flatten_inputs`;
  check aux explicitly or model it as static leaves.

## How to extend (hack-ability)

- **Add a new traceable op**: nothing to do in trace/ — `etl.ops` builds IR
  via `current_builder()`, and control flow needs no changes. (The op's
  shape/dtype rules live in ops/ir.)
- **Add a new static leaf type** (e.g. a config object): extend
  `_is_static_value` in `_tree.py` (the single copy) and check that the
  type survives `Graph.save/load` (persist codec in `etl/persist/`).
- **Add a new control-flow construct**: copy the `_run_in_region` pattern —
  build a detached region via `builder.build_region(...)`, classify operands
  (`_classify_operands` for cond-style, the carry rules for while-style),
  run callables inside the region, emit the `return` via
  `_return_terminator`, `create` the region op with explicit `result_types`
  (or `shape_fn`), pop in `finally`. Follow the `scan` desugar pattern for
  anything loop-shaped.
- **Change how static values are recorded**: the single construction site
  is `graph._static_record`; the classification sits in
  `trace._flatten_specs` / `trace._classify_outputs`. The recording format
  (`StaticValue.index/path/value/kind`) is consumed by `graph.py` and
  encoded by `etl.persist` — change all three together.
- **Add an input/output plan cache**: see "Caching deliberately NOT added"
  before doing it.

## Test strategy

pytest; mirror under `../tests/test_trace/` (or per-module files):
- `defn`: calling a Defn raises `TraceError` with the staging-direction
  message; bare/options decorator forms; idempotence.
- `trace`: static values specialize (python `if` over an int works at trace
  time); concrete-tensor and unknown-leaf inputs raise with path; nested
  structures (tuple/list/dict/namedtuple/dataclass) round-trip;
  SymbolicTensor inputs have no `.numpy()` (SymbolicTensor purity);
  closure-captured `Tensor` fails via ops' `TraceError`; output static
  values re-inserted; zero-input and zero-result graphs.
- `current_builder` raises outside trace; ops route into nested regions.
- `cond`/`while_loop`/`scan`: region ops appear in the IR (inspect module),
  loop-carried shapes, branch-tree mismatch errors, scan static-length
  round-trip vs equivalent while_loop, symbolic-length `TraceError`.
- `Graph`: flatten/unflatten validation (ShapeError/DTypeError/DeviceError/
  static mismatch), save/load round-trip via persist (integrity,
  `PersistenceError` on mismatch).
- Spec-compliance coverage lives in `../tests/test_spec_compliance.py`
  (owned by the package root): staging explicitness, bind-as-sugar,
  vmap≡vectorize, etc.

## Status

Implemented and green. The 2025-08 cleanup refactor (commit 507e42a):
extracted `_tree.py` (shared walker + static helpers), introduced
`_TraceSession`, unified region execution (`_run_in_region`) and block
termination (`_return_terminator`), deduped `cond` operand flattening and
output unflattening. Public API unchanged; error messages byte-identical;
full suite 5665 passed / 0 failed (skips environmental: torch/xla/GPU);
`etl.bench` conformance 97/97.
