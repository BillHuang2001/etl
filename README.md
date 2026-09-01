# etl — EvoX Tensor Library

A minimal, explicit, compiler-neutral tensor graph runtime for Python.

> **Make computation explicit, require large tensors to be explicit inputs, keep
> `vmap` as transparent function-side sugar over `vectorize`, keep binding as
> transparent argument-passing sugar, keep communication and multi-device tensor
> preparation explicit, expose and persist the graph, and let compilers do the
> compilation.**

etl provides an ergonomic tensor language for constructing, transforming,
inspecting, caching, loading, and executing computational graphs. Optimization,
kernel scheduling, memory planning, and code generation are left to external
compilers (IREE, XLA via PJRT, TVM).

```python
import etl
import etl.numpy as enp

@etl.defn
def f(x, w):
    return etl.dot(x, w) + 0.5

# Explicit staging: trace → lower → compile → load → run
graph = etl.trace(f, etl.TensorSpec((4, 8), etl.float32), etl.TensorSpec((8, 4), etl.float32))
lowered = etl.lower(graph)
artifact = etl.compile(lowered)
exe = etl.load(artifact, device="cpu")
y = etl.run(exe, etl.ones((4, 8)), etl.zeros((8, 4)))

# Convenience sugar (documented shorthand only):
exe = etl.build(f, etl.TensorSpec((4, 8), etl.float32), etl.TensorSpec((8, 4), etl.float32))
```

Features:
- Explicit graph construction — `@etl.defn` / `etl.trace`; no implicit tracing, no eager/graph mode switching
- Graph transforms (graph→graph): `etl.vectorize` / `etl.vmap`, `etl.grad` / `etl.jvp` / `etl.vjp`
- In-graph control flow: `etl.cond` / `etl.while_loop` / `etl.scan`
- Explicit collectives: `etl.dist.all_reduce(...)` — the compiler may optimize, never invents, communication
- Sparse tensors: `etl.sparse` (COO/CSR/CSC) — 16 graph ops, works with transforms and control flow
- Key-based functional RNG: `etl.random.key(seed)` — deterministic, stateless, multi-algorithm
- Custom kernels: `etl.external_call` + `etl.register_external_kernel` (30-second demo below); custom graph ops via `etl.block`
- Reference CPU backend (numpy interpreter, default) + StableHLO MLIR export
- Pluggable compiler adapters — IREE, XLA (via PJRT), TVM — optional extras, lazily
  activated (`etl.lower(graph, backend="iree")`); the library runs with none installed
- Symbolic shapes (`B = etl.dim("B")`), pytree-structured inputs/outputs, DLPack interop
- Persistence/caching: serializable `Graph` / `LoweredProgram` / `CompiledArtifact`

```python
# Same graph, real compilers (needs: pip install etl[iree] / etl[tvm];
# the xla backend needs a user-provided PJRT plugin .so — ETL_PJRT_PLUGIN):
exe = etl.build(f, etl.TensorSpec((4, 8), etl.float32), etl.TensorSpec((8, 4), etl.float32),
                backend="iree", device="cpu")
```

See `CONTEXT.md` for the architecture and `etl/CONTEXT.md` for the public API contract.

## Custom kernels in 30 seconds

`etl.external_call` plugs an opaque, named kernel into a `defn` graph — no etl
op needs to know it, and the kernel is just numpy arrays in/out (a triton/cuda
kernel written against this interface plugs in the same way). Canonical worked
example: `etl/bench/examples/op_external.py`.

```python
import numpy as np
import etl

x, w = etl.ones((4, 8)), etl.ones((8, 4))      # concrete runtime Tensors

# 1. Declare the call inside a defn graph — name + declared output specs
@etl.defn
def model(x, w):
    h = etl.external_call("row_softmax", x, result=etl.TensorSpec((4, 8), etl.float32))
    return etl.dot(h, w)                        # downstream ops consume it like any SSA value

# 2. Register a plain-numpy fallback (ndarrays in, ndarray out) — then it just works
def row_softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

handle = etl.register_external_kernel("row_softmax", row_softmax)   # default slot
y = etl.evaluate(model, x, w)                   # numpy backend dispatches the kernel by name

# 3. Per-backend kernels coexist with the fallback — no replacement
@handle.impl("iree")                            # a triton/cuda kernel registered under iree...
def row_softmax_iree(x): ...
y = etl.evaluate(model, x, w, backend="iree", target_backends=["llvm-cpu"])

# 4. Portability: graph-decomposition fallback for backends without host-dispatch (xla/tvm)
@handle.portable
@etl.defn
def row_softmax_graph(x):
    m = etl.max(x, axes=[-1], keepdims=True)
    e = etl.exp(etl.subtract(x, m))
    return etl.divide(e, etl.sum(e, axes=[-1], keepdims=True))

# 5. Composability: explicit transform rules — else the portable auto-fallback
@handle.batching_rule
def _batching(op, operands, axes): ...          # -> (new_values, new_axes)
@handle.vjp_rule
def _vjp(op, cotangents, primals): ...          # -> input cotangents
```

Registration is a run-time, process-global concern: graphs carry only the
kernel name, so a process that runs a saved graph must re-register its kernels.
The numpy backend validates count/dtype/shape against the declared specs and
errors loudly on unknown names; xla/tvm (no host-dispatch) fall back to the
portable decomposition with a `UserWarning`, and without any kernel or portable
registered the error is explicit and actionable.

## Install

```bash
pip install -e .          # numpy only — runs fully with the default numpy backend
pip install -e ".[dev]"   # + pytest
pip install -e ".[interop]"  # + torch (DLPack interop tests)
pip install -e ".[compilers]"  # + IREE, TVM adapters (xla: user-provided PJRT plugin .so, no pip dep)
```

## Test

```bash
pytest
```
