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

## API Surface

Exported from `__init__.py` (the public contract — exact names):

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

## Trace algorithm (binding contract — Phase 2 must implement exactly)

1. Unwrap `Defn`/`__etl_defn__` → plain fn.
2. Treat `specs` (positional args) as ONE pytree; flatten via
   `core.TreeSpec`. Leaves must be `TensorSpec` (tensor input; shape may
   hold `Dim`/`DimExpr`, `None` = dynamic) or static Python values
   (`None`/bool/int/float/complex/str/`Enum`/numpy `dtype`/`slice` — see
   `_is_static_value`). Anything else (concrete `Tensor`, ndarray,
   `SymbolicTensor`, …) → `TraceError` with the pytree path. Capturing a
   concrete tensor is NEVER silently allowed.
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
   anything else → `TraceError`. Emit the function `return` terminator
   with the symbolic leaves (zero-result graphs are legal).
7. Return `Graph(...)` — NOT verified automatically (staging explicit;
   `etl.build` verifies). Every `trace` call yields a NEW Graph.

Static-value snapshot semantics: closure static values are naturally
snapshotted because the function body runs once at trace time — the values
are read when ops are built, and only explicit tensor values ever reach the
IR. Run-time graph specialization is validated by `Graph.flatten_inputs`
(static leaves must match recorded type+value; dtype/shape/device checked
against specs with `DTypeError`/`ShapeError`/`DeviceError`; tree mismatch →
`TraceError`). numpy arrays are accepted at run time and wrapped via
`core.from_numpy` (documented convenience).

## Graph layout

Entry `ir.Function` "main": block args = tensor inputs in `tensor_specs`
order; terminator `return` yields results in output-tree SymbolicTensor-leaf
order. `input_specs`/`output_tree` carry the structured I/O contract
(tuple/list/dict/namedtuple/dataclass — pytree machinery from `core`).
Static leaves live ONLY in `static_values`/`output_static_values` records,
never in the IR.

## Builder context

`current_builder()` / `with_builder()` (contextvars-backed tuple stack in
`builder.py` — fully implemented, thread/async-safe). `etl.ops` op functions
query `current_builder()` and build into the function body or the innermost
control-flow region. `control_flow.py` swaps builders per region so user
branch/body callables target the right region.

## Control-flow region conventions (coordination contract with `etl/ir`)

Region ops are built via `ir.opdef(name)` + the region `Builder` — this
module NEVER imports `etl.ops` (ops imports trace — keep the DAG acyclic).
Assumed ir API (coordinate with the ir architect; objective-sanctioned):

- Ops have a `regions` attr: `list[Region]`; block args carry loop-carried
  values. `ir.Builder` exposes creation of blocks-with-args inside a region
  and a generic `create(opdef_name, operands=, attrs=, result_types=,
  regions=)` (see `_region_builder` helper).
- `if` op ("if", effects none, 2 regions): no operands; regions = then,
  else — one block each, NO block args; each block's `return` terminator
  yields the branch outputs (n SymbolicTensors, identical trees, unified
  dtypes/shapes across branches); op results = selected branch's outputs.
  Branch callables run once each at trace time under `with_builder(...)`.
- `while` op ("while", effects none, 2 regions): operands = n initial
  carried values; regions = condition, body — one block each with n block
  args (the carried values). Condition region returns ONE scalar 0-d bool;
  body region returns n next-carried values. Op results = final carried
  values. cond_fn/body_fn run once each at trace time.
- `return` op: terminator-only (operands = yielded values), valid only as a
  block's last op.
- Required registry op defs for `scan` desugaring: `add`, `less`, `slice`
  (dynamic index), `expand_dims`, `concatenate` — built raw from
  `ir.opdef(...)`. FINAL NAMES are owned by ir's op registry; Phase 2 must
  verify against the implemented registry and adjust if names differ.

`scan` v1 scope: STATIC length only (int, or derived from xs's static
leading dim). Symbolic length → `TraceError` (dynamic-scan region ops
reserved; no silent fallback).

## Error cases (all public errors derive from `core.ETLError`)

`Defn.__call__` (always), `current_builder()` outside trace, invalid trace
input/output leaves, branch tree/arity mismatch or non-scalar-bool `pred`/
`cond_fn` result, static-kwargs that are not static, symbolic scan length
(v1) — all → `core.TraceError` with pytree path / graph location where
applicable. `flatten_inputs`: `ShapeError` (shape vs DimExpr), `DTypeError`,
`DeviceError`, `TraceError` (tree/static mismatch). `Graph.load` mismatch →
`PersistenceError` (from persist). `verify()` → `VerificationError`.

## Constraints

- Imports: `etl.core` and `etl.ir` ONLY (plus stdlib/numpy). `etl.persist`
  is imported lazily inside `Graph.save/load` (persist does not import
  trace). NEVER import `etl.ops` — ops imports trace for
  `current_builder()`; control flow builds raw ops via `ir.opdef`.
- Files < ~1000 lines; current layout: `defn.py`, `builder.py`, `graph.py`,
  `trace.py`, `control_flow.py`, `__init__.py`.
- All `trace` outputs are ordinary ops — backends need no control-flow
  runtime support (numpy interpreter selects regions/loops natively).
- No caching, no eager execution, no silent fallbacks anywhere in this
  module.

## Routing table

Flat module — no child directories. All tracing work lands in the five
modules above (`trace.py` = the tracer + input classification,
`control_flow.py` = region building, `graph.py` = Graph I/O validation +
persistence delegation, `builder.py` = the ops hook, `defn.py` = marker
decorator).

Sibling: `../tests/` → test suite (read-only from here; escalate test
writes to the package root).

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

Architecture complete (skeletons + contracts). Phase 2 (delegated to a
Manager): implement `trace` algorithm, control-flow region building, Graph
I/O + persistence delegation per the contracts above.
