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
- `@etl.defn` / `etl.trace` — explicit graph construction (no implicit tracing, no eager/graph switching)
- Symbolic shapes (`B = etl.dim("B")`) and pytree-structured inputs/outputs
- Graph transforms: `etl.vectorize` / `etl.vmap`, `etl.grad` / `etl.jvp` / `etl.vjp`
- Explicit collectives: `etl.dist.all_reduce(...)`
- Custom ops via `etl.block` (portable impls, per-backend kernels, batching/derivative rules)
- Reference CPU backend (numpy interpreter, default) + StableHLO MLIR export
- Pluggable compiler adapters — IREE, XLA (via PJRT), TVM — optional extras, lazily
  activated (`etl.lower(graph, backend="iree")`); the library runs with none installed
- Serializable, cacheable pipeline objects (`Graph`, `LoweredProgram`, `CompiledArtifact`)
- DLPack interoperability with PyTorch / CuPy / other runtimes

```python
# Same graph, real compilers (needs: pip install etl[iree] / etl[xla] / etl[tvm]):
exe = etl.build(f, etl.TensorSpec((4, 8), etl.float32), etl.TensorSpec((8, 4), etl.float32),
                backend="iree", device="cpu")
```

See `CONTEXT.md` for the architecture and `etl/CONTEXT.md` for the public API contract.

## Install

```bash
pip install -e .          # numpy only — runs fully with the default numpy backend
pip install -e ".[dev]"   # + pytest
pip install -e ".[interop]"  # + torch (DLPack interop tests)
pip install -e ".[compilers]"  # + IREE, XLA (PJRT), TVM adapters
```

## Test

```bash
pytest
```
