# etl/dist — explicit distributed collectives

## Intent

Graph-time, explicit collective operations under the SPMD local-tensor model (root `../CONTEXT.md`, principle 5): every tensor is a local physical tensor, collectives appear explicitly in the program, and the compiler may optimize but never invent communication. `dist` owns:

- **`Group`** — static, named sets of ranks a collective operates over,
- **the six collectives** — frontend functions that each build ONE IR op with effect kind `collective` into the active trace builder (same discipline as `etl/ops`: `trace.current_builder()` hook, SymbolicTensor-in / SymbolicTensor-out, `TraceError` on concrete tensors / outside a trace),
- **`rank()` / `world_size()`** — scalar int64 graph values (effect `read`) resolved from the runtime execution context by backends,
- **the collective-executor hook** (`set/get_collective_executor`).

`dist` performs **no concrete computation** (no eager numerical duplication — root principle 9): the default single-process identity executor lives in the numpy backend and is installed via the hook at backend import time.

**Boundary:** explicit data-preparation helpers (`split_tensor`, `replicate_tensor`) are NOT here — they live in `etl/core` as concrete, eager utilities. `dist` = graph-time ops only.

## API Surface

```python
# groups (group.py)
group(name: str, ranks: tuple[int, ...], backend=None) -> Group
class Group                          # attrs: name, ranks, backend; is_world, size(world_size=None), __contains__, __eq__/__hash__
WORLD_GROUP: Group                   # Group("world", None) — default group=None of every collective

# collectives (collectives.py)
all_reduce(tensor, op="sum", group=None) -> SymbolicTensor
all_gather(tensor, axis=0, group=None) -> SymbolicTensor
reduce_scatter(tensor, op="sum", axis=0, group=None) -> SymbolicTensor
all_to_all(tensor, split_axis, concat_axis, group=None) -> SymbolicTensor
broadcast(tensor, src_rank=0, group=None) -> SymbolicTensor
collective_permute(tensor, source_target_pairs, group=None) -> SymbolicTensor

# execution context + executor hook (context.py)
rank() -> SymbolicTensor             # scalar int64, shape ()
world_size() -> SymbolicTensor       # scalar int64, shape ()
RankContext(rank: int, world_size: int)     # frozen dataclass, validated
set_collective_executor(executor | None) -> None
get_collective_executor() -> CollectiveExecutor
```

## Group semantics

- Groups are **static Python values** (value model, root CONTEXT.md): resolved at trace time, they specialize the graph. Immutable + hashable; equality over `(name, ranks, backend)`; treat as frozen after construction.
- `ranks` is validated at construction: non-empty, unique, non-negative ints (bools rejected) → `ValueError`. `ranks=None` ⇔ the **world group** (`WORLD_GROUP`, name `"world"`): all ranks of the runtime execution context — membership only known at run time. `Group.size(world_size)` resolves it.
- Collective ops record the group in attributes **by name + size** (`group: str` + `group_size: int | None`, `None` for world) — serialization is self-describing, no Python objects in artifacts.
- **Coordination:** `etl/trace` static-value snapshotting must handle `Group` by duck-typing its `name`/`ranks`/`backend` attributes — trace MUST NOT import dist (dist imports trace; a trace→dist import would cycle).

## Local-shape semantics (worked examples)

All examples: explicit 4-rank group `g = group("data", (0, 1, 2, 3))`, local tensor `x` per rank. Results are LOCAL shapes; **no global tensor type ever appears in the IR** (spec-compliance asserts this).

| collective | local input | local output |
|---|---|---|
| `all_reduce(x, "sum", g)` | `[4, 8]` | `[4, 8]` (elementwise sum of all 4 ranks) |
| `all_gather(x, axis=0, g)` | `[256, 1024]` | `[1024, 1024]` |
| `all_gather(x, axis=1, g)` | `[256, 1024]` | `[256, 4096]` |
| `reduce_scatter(x, "sum", axis=0, g)` | `[1024, 1024]` | `[256, 1024]` |
| `all_to_all(x, 0, 1, g)` | `[512, 64]` | `[128, 256]` |
| `broadcast(x, src_rank=0, g)` | `[256, 1024]` | `[256, 1024]` (rank 0's copy on every rank) |
| `collective_permute(x, ((0,1),(1,2),(2,3),(3,0)), g)` | `[256, 1024]` | `[256, 1024]` (rank i receives rank (i−1)%4's tensor) |

Shape rules (explicit groups → trace-time DimExpr arithmetic; world group → axis dims are `None`, runtime-dynamic, validated by the executor):

- `all_reduce` / `broadcast` / `collective_permute`: shape unchanged.
- `all_gather`: `out[axis] = in[axis] * group_size`.
- `reduce_scatter`: `out[axis] = in[axis] // group_size` (explicit groups: must divide evenly, else `ShapeError`).
- `all_to_all`: `out[split_axis] = in[split_axis] // group_size`, `out[concat_axis] = in[concat_axis] * group_size` (equal axes ⇒ unchanged; explicit groups: split-axis dim must divide evenly).
- All axes normalize Python-style (negative wraps); out of range → `ShapeError`.

## IR op attribute layout (coordination with etl/ir)

Canonical op defs live in `etl/ir/op_defs/collective.py` — **the registry wins over this file**. Op names have NO `dist.` prefix. All six communication collectives: effect kind **`collective`**, 1 operand, 1 result, and always carry `group: str` (required) + `group_size: int | None` (required — `None` for the world group).

| ir op name | extra attrs | result shape |
|---|---|---|
| `all_reduce` | `reduce_op: str` ("sum"\|"max"\|"min"\|"prod") | same local shape/dtype |
| `all_gather` | `axis: int` (default 0) | axis dim × group size (or `None`) |
| `reduce_scatter` | `reduce_op: str`, `axis: int` (default 0) | axis dim ÷ group size (or `None`) |
| `all_to_all` | `split_axis: int`, `concat_axis: int` | split dim ÷ g, concat dim × g (or `None`/`None`) |
| `broadcast_collective` | — (the shape op `broadcast` owns its name) | same local shape/dtype |
| `collective_permute` | `source_target_pairs: tuple[(int, int), ...]` | same local shape/dtype |
| `rank` | `group: str` (required — dist passes `"world"`) | scalar int64, shape `()` |
| `world_size` | `group: str` (required — dist passes `"world"`) | scalar int64, shape `()` |

`rank` / `world_size` carry effect **`read`** (not `pure`) — backends MUST resolve them at run time from the execution context and MUST NOT constant-fold them at lower/compile time.

Result shapes come from the op defs' `shape_fn`s (`infer_identity`, `infer_all_gather`, `infer_reduce_scatter`, `infer_all_to_all`, `infer_scalar_int64` in `etl/ir/inference.py`) applied automatically by `Builder.create()` — dist reads result dtypes/shapes from the result value's type and never recomputes them. dist still performs its own explicit divisibility checks (reduce_scatter / all_to_all, static int dims, explicit groups) before building, for clear `ShapeError`s.

## Collective executor hook contract

```
CollectiveExecutor (runtime_checkable Protocol, context.py) — one method per
collective op (the numpy backend's SingleRankCollectiveExecutor structurally
satisfies it):
  all_reduce(tensor: Tensor, group: Group, op) -> Tensor
  all_gather(tensor: Tensor, axis: int, group: Group) -> Tensor
  reduce_scatter(tensor: Tensor, axis: int, group: Group) -> Tensor
  all_to_all(tensor: Tensor, axis: int, group: Group) -> Tensor
  broadcast(tensor: Tensor, src_rank: int, group: Group) -> Tensor
  collective_permute(tensor: Tensor, mapping, group: Group) -> Tensor
```

- `tensor`: the local concrete `Tensor`; `group` / `mapping` / `op` are graph-constant objects (the `dist.Group` descriptor / permutation mapping / reduction kind).
- Returns the LOCAL result tensor. Executors may coordinate across ranks in-process (multi-rank simulation in tests) or delegate to a real transport.
- `RankContext(rank, world_size)` is exported for multi-rank simulations; it is NOT part of the v1 protocol parameters.
- Hook is process-global and starts **unset** (a clean `import etl` installs nothing): `get_collective_executor()` → `BackendError` ("no collective executor registered; load a backend that provides one (e.g. numpy_backend)"). `set_collective_executor(None)` resets; anything else that is not structurally a `CollectiveExecutor` raises `TypeError` (runtime-checkable).
- **Coordination:** `etl/backends` (numpy) installs its default single-process identity executor at import time — `etl/backends/numpy/__init__.py` (module tail) calls `dist_context.set_collective_executor(SingleRankCollectiveExecutor())`; the interpreter's collective kernels (`etl/backends/numpy/kernels/collective.py`) dispatch all eight ops through `dist.context.get_collective_executor()` (body-level import of `etl.dist.context` — acyclic). The numpy interpreter resolves `rank`/`world_size` from a per-execution `RankContext` (default rank 0 / world size 1, `etl/backends/numpy/exec_context.py`) and accepts an optional per-`run` override (`NumpyExecutable.run(..., rank_context=...)`, thread-local) so multi-rank tests can resolve the scalars per simulated rank.

## Error behavior

| condition | error |
|---|---|
| `group()`: empty/duplicated/negative/non-int ranks, empty name | `ValueError` |
| collective with concrete `Tensor` operand, or outside a trace | `TraceError` (mirrors `etl/ops`; error messages carry source location) |
| unknown reduction `op` | `ValueError` |
| axis out of range for operand rank | `ShapeError` |
| explicit group: reduce_scatter / all_to_all axis dim not divisible by `len(ranks)` | `ShapeError` |
| `broadcast`: `src_rank` negative or not in explicit group | `ValueError` |
| `collective_permute`: empty/malformed pairs, duplicate src or dst, rank outside explicit group | `ValueError` |
| `get_collective_executor()` with nothing installed | `BackendError` |
| world-group runtime validation (divisibility, src membership) | `BackendError` at execution, raised by the executor |

## Constraints

- **Imports:** `_op_utils.py` imports `etl.core`, `etl.ir`, and `etl.dist.group`; `collectives.py` / `context.py` import `_op_utils`, `etl.core`, and `from etl.trace import current_builder`. The trace import MUST be the direct submodule form — `from etl import trace` yields the trace *function* attribute (shadowed in `etl/__init__.py`), not the module. No `backends`/`pipeline`/`persist` imports. `backends` may import `dist` (hook install); `trace` and `persist` must NOT (see Group coordination note — use duck typing).
- **No concrete computation in dist** — no numpy numerics; executor implementations live in backends (or tests).
- **v1 collectives are single-tensor** (1 operand → 1 result); multi-tensor collectives are future extensions.
- Files < ~1000 lines; new collectives = ir op def + frontend function + shape rule — the executor hook protocol stays unchanged.

## Test strategy

pytest under `../tests/dist/` (sibling — read-only from here; escalate test writes to root):

- `test_group.py`: validation errors, equality/hash, world-group semantics, static-value-ness.
- `test_collectives.py`: graph construction — op attrs (group name/ranks, params), effect kind `collective`, local shapes per the worked-examples table, `TraceError` on eager calls, world-group `None` dims.
- `test_context.py`: `rank()`/`world_size()` scalar int64 ops (effect `read`, attr `group="world"`); hook unset → `BackendError`; install/clear round-trip.
- **Multi-rank simulation:** install a custom in-process executor; run the same executable once per simulated rank with distinct local inputs; the executor buffers contributions (keyed by op instance + group) until all ranks contributed, then computes numpy semantics. Assert all_gather concatenation, all_reduce sum/max/min/prod, reduce_scatter chunking, all_to_all transposition, broadcast, collective_permute routing, plus rank/world_size resolution per simulated rank.
- Default numpy backend: identity semantics (rank 0 / world 1 → copy).
- Serialization: graph save/load round-trips group name+ranks attrs; no Python objects in artifacts.
- Spec-compliance (in `../tests/test_spec_compliance.py`): collectives are explicit `collective`-effect ops; no global tensor type anywhere; local shapes only; `rank()`/`world_size()` are graph values, not Python ints.

## Routing table

| Path | Area |
|---|---|
| `./_op_utils.py` | private shared helpers: operand normalization (`_require_symbolic_tensor`), location capture (`_get_location`), result wrapping (`_wrap_result`), group/reduction/axis/pairs validation, `REDUCTIONS` |
| `./group.py` | `Group`, `group()`, `WORLD_GROUP` — static group values + validation |
| `./collectives.py` | the six collectives + `REDUCTIONS` (re-exported from `_op_utils`) — graph-time op builders |
| `./context.py` | `rank()`, `world_size()`, `RankContext`, executor hook (`CollectiveExecutor` protocol + `set/get_collective_executor`) |
| `./__init__.py` | re-exports the public surface |

Siblings (read-only — escalate writes to parent): `../core/` (SymbolicTensor, Tensor, errors), `../ir/` (op defs, builder, shape inference), `../trace/` (active-builder hook, static snapshotting), `../ops/` (frontend-op discipline reference), `../backends/` (installs the default executor), `../../tests/` (test suite).

## Status

Implementation complete: all six collectives build canonical IR ops (registry op names, `reduce_op`/`group`/`group_size` attrs, effect `collective`), `rank()`/`world_size()` build scalar int64 `read`-effect ops, and the executor hook slot (`CollectiveExecutor` protocol + `set/get_collective_executor`) is live. Validated via throwaway scripts (explicit-group and world-group shapes/attrs, serialization round-trip, error paths, executor hook).
