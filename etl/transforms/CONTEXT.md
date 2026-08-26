# etl/transforms — graph-to-graph transformations

## Intent

Frontend graph transformations: `vectorize` (the batching primitive), `vmap` (transparent function-side sugar over vectorize), and automatic differentiation (`grad` / `jvp` / `vjp`). Everything here is **graph → graph**: the result of any transform is an ordinary `Graph` of ordinary `etl.ops` ops — backends need NO vectorize/autodiff runtime support (binding: `../../CONTEXT.md` design principle 7, `../CONTEXT.md` transforms bullet).

There is deliberately **no execution here**: transforms never import `etl.backends` / `etl.pipeline` and never run tensors. When a transform is given a plain function or `Defn` (rather than a `Graph`), it cannot know input shapes without specs, so it returns a `TransformCallable` mapping `TensorSpec`s → transformed `Graph` (see **Spec-callable convention** below). Calling that callable with concrete `Tensor`s raises `TraceError` — explicit staging applies here as everywhere.

## API surface (public — re-exported from `__init__.py`)

- `vectorize(graph, axes) -> Graph` — THE primitive. `graph` must be a traced `Graph` (a callable/`Defn` raises `TypeError` pointing at `etl.vmap`). `axes`: an `int`, `None`, or a pytree thereof matching the graph's input structure; an `int` maps that input's leading axis (v1: only 0), `None` leaves it unmapped (mandatory for static inputs). Every op is rewritten via its batching rule; mapped values carry an explicit leading batch dim. No rule ⇒ `core.TransformError` naming the op — never a Python-loop fallback.
- `vmap(fn_or_graph, in_axes=0, out_axes=0) -> Graph | TransformCallable` — function-side sugar over the same machinery. Graph input: `vectorize(graph, in_axes)` + output-axis rearrangement per `out_axes`. Callable/`Defn` input: returns a `TransformCallable` `(*batched_specs) -> Graph` that strips the leading mapped dim from each mapped spec, traces the function exactly ONCE via `etl.trace`, then vectorizes + rearranges.
- `grad(fn_or_graph, argnums=None) -> Graph | TransformCallable` — reverse-mode gradients via the shared VJP machinery. Graph output must be exactly one scalar tensor (`ShapeError` otherwise). `argnums`: int or tuple/list of ints indexing the flattened **tensor** inputs (static values excluded); `None` = all tensor inputs. Selected inputs must be floating/complex (`TransformError` otherwise). Result graph: same inputs → one gradient tensor (int argnum) or a tuple of gradient tensors.
- `jvp(fn_or_graph, tangents) -> Graph | TransformCallable` — forward-mode. `tangents`: pytree of `TensorSpec` (or `None` = zero tangent) matching the **input** structure; shapes/dtypes must match the primal specs. Result graph inputs = primal inputs followed by tangent inputs (in flattened order); outputs = `(primal_outputs, tangent_outputs)`.
- `vjp(fn_or_graph, cotangents=None) -> Graph | TransformCallable` — reverse-mode. `cotangents`: pytree of `TensorSpec` (or `None`) matching the **output** structure; `None` default = scalar-one cotangent `TensorSpec((), out_dtype)`, valid only for a single scalar output. Result graph inputs = primal inputs followed by cotangent inputs; outputs = `(primal_outputs, input_cotangents)`. Shares VJP rules with `grad`.
- Rule registries (public mutable dicts + register functions): `batching_rules` / `register_batching_rule(name, fn)`, `jvp_rules` / `register_jvp_rule(name, fn)`, `vjp_rules` / `register_vjp_rule(name, fn)`. Custom blocks register under the namespace **`block:<block_name>`** — that is exactly what `BlockOp.batching_rule/jvp_rule/vjp_rule(fn)` in `../block/` do (block imports transforms for registration; transforms never imports block, keeping the import graph acyclic).
- `TransformCallable` — the object returned for fn/`Defn` inputs; `__call__(*specs) -> Graph` (fresh graph per call); `kind` attribute (`"vmap" | "grad" | "jvp" | "vjp"`).

## The vectorize core (design, binding for implementation)

`vectorize(graph, axes)` walks the graph's function blocks in **topological order** and builds a **new** module/function (never mutates the input graph; `Graph`'s prebuilt-module constructor is used). Per value, the rewriting env records batching metadata (see **MappedAxes** below); per op, the machinery:

1. Looks up the operand metadata from the env (`ValueEnv`).
2. Fetches the op's rule: `batching_rules[op.def.name]`; for `block_call` ops the key is `block:<name>` taken from the op's block-name attribute. Missing rule ⇒ `TransformError` naming the op/block.
3. Pushes its own `ir.Builder` onto the trace builder stack and invokes the rule; rules build replacement ops with ordinary `etl.ops.*` functions (resolved via `trace.current_builder()`).
4. Records the returned values + metadata; seeds input metadata from `axes` (mapped input specs gain a fresh symbolic `Dim` named `batch`, `batch_1`, … as the leading dim; rank stays known).

The result `Graph` preserves `output_tree` and `static_values`; mapped outputs carry the leading batch dim; unmapped outputs are unchanged. Source locations of original ops are attached to replacement ops.

## The vmap ⇔ vectorize equivalence contract (binding, tested)

1. **Graph case**: `vmap(graph, in_axes, out_axes=0)` must produce the **same IR** as `vectorize(graph, in_axes)` — same ops, same input/output specs. Tests compare `ir.serialize_module` / `pretty_print` output for exact equality.
2. **Function case**: `vmap(f)(*batched_specs)` must produce the same IR as `vectorize(etl.trace(f, *stripped_specs), in_axes)` with `out_axes=0`, where `stripped_specs` removes the leading mapped dim from each mapped spec (dtype/device/name preserved).
3. **Rearrangement is post-processing**: `out_axes` handling (transpose of a mapped axis, insertion of a size-one axis) happens AFTER vectorize, as ordinary ops added to the already-vectorized graph — never inside batching rules. With `out_axes=0` there is no extra op at all.
4. **Once-only tracing**: the wrapped function is traced exactly once per `TransformCallable` invocation (tested with a call counter in `f`).
5. **Numerical check**: building + running the vmap graph on a batch equals running the unvectorized graph per row and stacking, for a fixed set of inputs.

`out_axes` semantics (v1): entry `0` ⇒ output must be mapped — its mapped axis stays leading; if the output is unmapped, a size-one axis is inserted at 0 (explicitly requested, allowed). Entry `None` ⇒ output must be unmapped; a mapped output raises `TransformError` (axis mismatch — a batch axis can never be silently dropped). Non-zero entries are deferred (see v1 scope).

## Rule-call signatures (binding)

Rules are **pure graph builders**: they emit ops into the active transform builder (the machinery has pushed its `ir.Builder` onto the trace builder stack while a rule runs) and must never loop over batch elements or swallow errors.

**Proxy-op convention (implemented in `autodiff.py`, binding for JVP/VJP rules)**: the AD machinery rebuilds the full primal computation into a new module, so the ORIGINAL op's values are NOT the values rules must build on. Rules therefore receive a **proxy `ir.Op`** — a fresh dataclass instance never inserted into any block — carrying the original op's `name`/`id`/`attributes` (shallow-copied dict; leaf objects shared, read-only) and `location`, but whose `operands` and `results` are the RECREATED SSA values of the transformed graph (same types as the originals). `op.opdef`/`op.effect`/`op.result` resolve normally by name. Rules must treat the proxy as read-only and must NOT follow `value.owner`/`value.defining_op` off proxy values (those still point at the original op). The `constant` op is handled by the machinery itself (fixed data: output tangent = `ZeroTangent`; zero operands propagate nothing) — it has NO registry entry.

- **Batching**: `rule(op, operands, axes) -> (new_values, new_axes)` — `operands: tuple[ir.Value, ...]` are the original operand values, `axes: tuple[MappedAxes, ...]` their metadata (aligned); returns replacement SSA values aligned with `op.results` plus their metadata. A rule may emit any number of ops (e.g. a reduction rule shifts its axis argument; a dot rule builds a batched matmul).
- **JVP**: `rule(op, tangents) -> output_tangents` — `tangents` is a tuple of `ir.Value | None | ZeroTangent` aligned with `op.operands`; primal values are available as `op.operands` / `op.results`. Returns a tuple aligned with `op.results`.
- **VJP**: `rule(op, cotangents, primals) -> input_cotangents` — `cotangents` aligned with `op.results`; `primals` = the op's operand values. Returns a tuple of `ir.Value | None | ZeroTangent` aligned with `op.operands`; `None`/`ZeroTangent` mean zero gradient, materialized (as a zeros op) only when a real tensor is required (e.g. cotangent accumulation), keeping transformed graphs free of dead zero ops.

## Axis metadata (MappedAxes)

Per-value batching metadata in a transformed graph: a tuple of **leading** mapped axis indices — `(0,)` after one vectorization, `(0, 1)` after nesting two, `()` = unmapped (`UNMAPPED`). A value of shape `(a, b, c)` with axes `(0,)` has batch extent `a` and per-row dims `b, c`. Metadata lives in a `ValueEnv` keyed by value identity; rules receive operand metadata as arguments and never touch the env. Static operands (attributes) carry no metadata — they specialize, they don't batch.

## AD semantics (binding)

- `grad`: exactly one scalar output (`ShapeError` otherwise). Reverse mode = one backward sweep: per-op VJP rules applied in reverse topological order; cotangents of multiply-used values accumulate via `add`. The sweep is seeded with a scalar-one cotangent of the output dtype. `vjp` returns primal outputs + input cotangents; `grad` returns only the selected input gradients. `jvp` is a forward sweep seeded with the tangent inputs.
- `stop_gradient`: gradient/tangent = zero (`ZeroTangent`) in both directions.
- Boolean/int-producing ops (`equal`, `less`, `greater`, `argmax`, `argmin`, …) have builtin rules yielding `ZeroTangent` — their outputs cannot backpropagate, which is NOT an error.
- Ops with **no rule** — `runtime_call`, collectives, `block_call` of a block without a registered rule — raise `TransformError` naming the op/block. No silent fallback, ever.

## Spec-callable convention (design decision)

Transforms are graph→graph, but `vmap(f)/grad(f)/jvp(f)/vjp(f)` on a bare function cannot know input shapes at transform time, and importing pipeline/backends is forbidden. Therefore fn/`Defn` inputs return a `TransformCallable`: `tf(*specs) -> Graph`. Each public module's docstring states the exact expansion (trace → transform → rearrange), keeping every composition transparent — no hidden staging, no hidden execution. `etl.build(tf(*specs))` is the documented way to run the result.

## v1 scope vs deferred

- **v1 supported**: straight-line op graphs (all ops with registered rules); nested vmap via composition (rule support permitting); `stop_gradient`; custom blocks via `block:<name>` rules.
- **Deferred → `TransformError`**: vectorizing region-bearing control-flow ops (`cond`/`while_loop`/`scan`) — no batching rules are registered for them in v1; non-zero `in_axes`/`out_axes` entries (mapped axes must be leading; transpose-based normalization is planned); vectorizing `runtime_call` (callbacks are per-value Python).
- **Non-goals here**: no caching of transformed graphs (design principle 3), no backend awareness, no lowering.

## Constraints

- **Imports (binding)**: only `etl.core`, `etl.ir`, `etl.trace`, `etl.ops`, and this package's own submodules. NEVER `etl.backends` / `etl.pipeline` / `etl.dist` / `etl.block`. Block rules arrive through the registries (key `block:<name>`), keeping `block → transforms` one-directional.
- Files < ~1000 lines: machinery in `batching.py`/`autodiff.py`, builtin rules in `rules.py` (split per op category if it grows), entry points one file each.
- **Errors**: `TransformError` (missing rule, axis mismatch, bad out_axes, non-differentiable argnums), `ShapeError` (grad of non-scalar output), `TraceError` (concrete tensors passed to a `TransformCallable`), `TypeError` (wrong argument kinds). Messages name the offending op/block and include source locations where available.

## Routing table (flat module — concerns → files)

| File | Area |
|---|---|
| `vectorize.py` | public `vectorize` + axes validation/normalization |
| `batching.py` | batching rule registry + core vectorize rewriting algorithm |
| `vmap.py` | public `vmap`, spec derivation, output-axis rearrangement |
| `autodiff.py` | JVP/VJP rule registries, `ZeroTangent`, forward/backward sweeps |
| `grad.py`, `jvp.py`, `vjp.py` | public AD entry points |
| `rules.py` | builtin rules for the standard op set (registered at import) |
| `_metadata.py` | `MappedAxes`, `ValueEnv` |
| `_wrappers.py` | `TransformCallable` (spec → Graph convention) |

Sibling: `../../tests/transforms/` → test suite for this module (read-only from here — escalate test-related writes to root).

## Test strategy

`../../tests/transforms/`: `test_vectorize.py` (per-op batching, metadata propagation, `TransformError` for missing rules), `test_vmap.py` (sugar semantics, once-only tracing, out_axes incl. size-one insertion and `None` mismatch, nested vmap), `test_grad.py` / `test_jvp.py` / `test_vjp.py` (numerical checks vs finite differences), `test_vmap_vectorize_equivalence.py` (IR equality per the equivalence contract), `test_block_rules.py` (`block:<name>` namespacing). Spec-compliance additions live in `../../tests/test_spec_compliance.py` (vmap≡vectorize sugar, unsupported-op errors, transformed graphs contain only ordinary ops). CPU only, pytest.

## Notes for agents

- Rules must NEVER fall back to Python loops or silently no-op.
- Skeleton stubs reference sibling modules (`core`/`ir`/`ops`/`trace`) by their **contracted export names only** (from `../CONTEXT.md`); during implementation, reconcile concrete signatures (e.g. `Graph`'s prebuilt-module constructor, the function terminator op, `Value` identity) with the landed APIs of those modules.
- When adding a public name here, update `__init__.py`, this CONTEXT.md, and the parent contract `../CONTEXT.md` together.
- `etl/__init__.py` (parent-owned) must re-export this surface: `vectorize`, `vmap`, `grad`, `jvp`, `vjp`, the three registries + register functions, `TransformCallable`.
