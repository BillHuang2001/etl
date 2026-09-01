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
op needs to know it, and the kernel is just numpy arrays in/out. The canonical
worked example is `etl/bench/examples/op_external.py`; here is the real thing:
a Triton/CUDA kernel hosted by IREE.

```python
import etl

x, w = etl.ones((4, 8)), etl.ones((8, 4))      # concrete runtime Tensors

# 1. Declare the call inside a defn graph — name + declared output specs
@etl.defn
def model(x, w):
    h = etl.external_call("triton_add", x, result=etl.TensorSpec((4, 8), etl.float32))
    return etl.dot(h, w)                        # downstream ops consume it like any SSA value

# 2. The default (numpy) slot is mandatory — the idiomatic default is a stub
#    that raises: a custom kernel targets specific hardware, so faking a CPU
#    path is worse than failing loudly.
def _no_cpu_fallback(*_):
    raise etl.BackendError(
        "'triton_add' is a Triton/CUDA kernel: no numpy fallback. "
        "Run with backend='iree' on a CUDA device, or register a real "
        "numpy implementation in the default slot."
    )
handle = etl.register_external_kernel("triton_add", _no_cpu_fallback)

# 3. The kernel — triton/torch are the kernel author's own toolchain: etl
#    remains numpy-only, triton is not an etl dependency, and the kernel can
#    be written against this interface from a separate repo.
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

# 4. Register it in the "iree" backend slot — numpy in, numpy out. The iree
#    host-dispatch stages operands to host numpy at each graph boundary (v1:
#    operands/results cross host↔device) and validates the result against the
#    declared specs.
@handle.impl("iree")
def triton_add_iree(x):
    n = x.size
    xt = torch.from_numpy(x).cuda()
    yt = xt
    out = torch.empty_like(xt)
    add_kernel[(triton.cdiv(n, 1024),)](xt, yt, out, n, BLOCK=1024)
    return out.cpu().numpy()

y = etl.evaluate(model, x, w, backend="iree", target_backends=["cuda"])
```

Registration is a run-time, process-global concern: graphs carry only the
kernel name, so a process that runs a saved graph must re-register its kernels.
Every backend resolves the default slot when no exact slot exists — omitting
it makes the numpy backend raise loudly anyway (exact backend slot → default
slot → loud `BackendError`). The numpy backend validates count/dtype/shape
against the declared specs and errors loudly on unknown names. For backends
without host-dispatch (xla/tvm), `@handle.portable` over an `@etl.defn` graph
is the fallback with a `UserWarning`, and `@handle.batching_rule` /
`@handle.vjp_rule` provide explicit transform rules — else the portable
auto-fallback applies.

## etl vs torch, jax, nx

etl is the only one of the four with **no eager mode and no implicit tracing**:
the graph is the program, staging is explicit, and compilers are external — no
compiler-owned optimization or resharding semantics. Torch traces implicitly
(`torch.compile`/`export`), jax traces at every `jit` call, and nx pairs
explicit `defn` graphs with eager execution. etl makes every stage an explicit
step, and its sugar is documented shorthand only.

| | PyTorch | JAX | Nx (Elixir) | etl |
|---|---|---|---|---|
| Programming model | imperative eager + `torch.compile` | functional, transforms over JIT'd functions | explicit `defn` graph language, `block`/`vectorize` | explicit `@etl.defn` graphs, no eager mode at all |
| Graph / staging | implicit tracing (`torch.fx`/`export`) | `jit` traces at call | `defn`/`trace` explicit | every stage explicit (`trace` → `lower` → `compile` → `load` → `run`), sugar is documented shorthand only |
| Transforms | `torch.autograd`, `torch.func.vmap` | `grad`/`vmap`/`jit` as function transforms | `grad`/`vectorize` in-graph | graph→graph `vectorize`/`vmap`/`grad`/`jvp`/`vjp` (backends never see `vmap`) |
| In-graph control flow | `torch.cond`/Python loops | `lax.cond`/`while_loop`/`scan` | `cond`/`while_loop`/`scan` | `etl.cond`/`while_loop`/`scan`; Python control flow over symbolic tensors fails clearly |
| Compilation / backends | own kernels + Inductor/Triton | XLA | pluggable backends (EXLA…) | compiler-neutral — numpy reference backend + StableHLO export + pluggable adapters (IREE/XLA-PJRT/TVM), no optimization/kernel ownership |
| Custom kernels | `torch.utils.cpp_extension`, `triton.jit` with torch tensors | `custom_call`/Pallas | `Nx.Defn.Kernel` | kernel-agnostic `etl.external_call` + name-keyed registry (numpy in/out, per-backend slots) and `etl.block` custom ops |
| RNG | stateful `torch.Generator` | key-based functional | stateful backend RNG | key-based functional, deterministic, multi-algorithm (splitmix64 default, threefry, philox) |
| Sparse | `torch.sparse` (separate layout world) | `jax.experimental.sparse` | none in core | first-class `etl.sparse` (COO/CSR/CSC) that flows through transforms and control flow |

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
