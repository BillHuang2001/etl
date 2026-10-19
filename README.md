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

## Run a HuggingFace model compiled (`etl_hf`)

`etl_hf` is a separate top-level package built **on** etl (etl core stays free of
layers/loaders): it reads a real checkpoint — `config.json` plus
`model.safetensors` / `pytorch_model.bin` — and runs the **whole model forward
pass as ONE etl graph**, so the model goes through the same explicit
`trace → lower → compile → load → run` pipeline as any other graph. It is
numpy-only: the checkpoint readers are pure numpy (`safetensors`, sharded
indexes and both `pytorch_model.bin` encodings), and `torch`/`transformers`/
`iree` are never imported at module scope.

The API is **functional** and follows etl's own value model — three layers that
never blur:

- **checkpoint IO produces DATA.** `load_config(ckpt)` returns a frozen static
  `ModelConfig` (one config = one graph), `load_params(ckpt, cfg)` the canonical
  `dict[str, np.ndarray]` of host weights, and `load_hf(ckpt)` both at once as a
  plain `LoadedModel(config, params, family, model_dir)` record with no methods.
  Nothing here is traced, compiled or cached.
- **`forward_fn` is the COMPUTE FUNCTION.** A pure `@etl.defn`
  `(params, input_ids) -> logits` that captures **no** weights: every parameter
  is an ordinary graph **input** (never `etl.constant`, so nothing is silently
  inlined into the program), and the static specialisation
  `(family, config, seq_len, batch_size, pack_weights)` is all the graph depends
  on (`build`'s cache key adds the resolved backend, device and options).
- **`build` / `run` are the explicit staging stages.** `build` *is*
  `trace → lower → compile → load` (via `build_cached`, memoised, so an
  identical specialisation returns the identical `Executable`); `run` is
  validation + `etl.run`; `predict` is the one documented host-facing shorthand.

```python
import numpy as np
import etl, etl_hf

ckpt = "/path/to/checkpoint"                     # a directory: config.json + weights

# 1. checkpoint IO -> DATA (no graph exists yet; nothing traced or compiled)
loaded = etl_hf.load_hf(ckpt)                    # LoadedModel: .config, .params, .family, .model_dir
cfg, params = loaded.config, loaded.params       # (or load_config(ckpt) / load_params(ckpt, cfg))

# 2. the COMPUTE FUNCTION: one pure @etl.defn graph, weights are explicit INPUTS
fwd = etl_hf.forward_fn(cfg, seq_len=32)         # (params, input_ids) -> logits, captures no weights
ids = np.zeros((1, 32), np.int32)                # static (batch, seq_len)

# 3. STAGING: the explicit trace -> lower -> compile -> load -> run pipeline
exe = etl_hf.build(fwd, cfg)                     # numpy interpreter (default)
logits = etl_hf.run(exe, params, ids)            # -> etl.Tensor, on the executable's device
exe = etl_hf.build(fwd, cfg, backend="iree")     # compiled once (StableHLO -> IREE)
logits = etl_hf.run(exe, params, ids)
host = etl_hf.predict(loaded, ids, backend="iree")          # host-facing shorthand -> np.ndarray
oracle = etl_hf.reference_logits(cfg, params, ids)          # independent pure-numpy oracle

# 4. device-resident recipe: place weights + ids ONCE, then run with zero host round-trips
device = etl.core.Device("rocm", 0)              # or etl.core.Device("cuda", N)
exe = etl_hf.build(fwd, cfg, backend="iree", device=device)
out = etl_hf.run(exe, etl_hf.place_params(params, device), etl_hf.place_ids(ids, device))
```

Outputs are `etl.Tensor`s on the executable's device — `logits.numpy()` for the
cpu instance, and for a non-cpu one the explicit
`out.to(etl.core.Device("cpu", 0)).numpy()`; nothing is copied to the host
behind your back. Opt-in weight packing is `forward_fn(cfg, pack_weights=True)`
+ `build(..., pack_weights=True)` + `pack_params(pack_plan(param_specs(cfg)), params)`.

- **Install & optionality:** `etl_hf` uses **only** `numpy` + etl (plus the
  standard library) and is a **separate top-level package used from the repo
  checkout** — it is deliberately **not installed by `pip install etl`**
  (`pyproject.toml`'s package find matches only `etl` / `etl.*`, so neither a
  wheel nor an sdist of `etl` contains `etl_hf`). `pip install etl[hf]` pulls
  only the optional `torch`/`transformers` reference oracles used to validate and
  benchmark `etl_hf` against the real HuggingFace implementation — never a
  runtime requirement. `pip install etl[iree]` (or `[tvm]` / a PJRT plugin `.so`
  for `xla`) enables the compiled path. `import etl` never imports `etl_hf`,
  `torch` or `transformers`.
- **Families:** `llama` (also `mistral` / `qwen2` configs — RMSNorm, RoPE, GQA,
  SwiGLU) and `gpt2` (also `distilgpt2` — LayerNorm, learned positions,
  `gelu_new`, tied head). Weights are always **explicit graph inputs** (never
  `etl.constant`), one graph per static `(config, seq_len, batch)` — memoised
  with `build_cached`.
- **Parity:** validated against real HuggingFace (`torch`/`transformers`)
  logits on five checkpoints (`tiny-random-llama`, `tiny-random-mistral`,
  `tiny-gpt2`, `distilgpt2`, `gpt2`): ≤1.5e-4 max-abs on the 124M models
  (≈1e-6 relative), fp32 accumulation-order only.
- **Compiled, including on GPU:** a 12-layer/124M graph (870 IR ops, 171 KB
  StableHLO) compiles on iree in a few seconds and runs **device-resident** on
  an AMD MI300X (iree/rocm): outputs stay on the device, weights are placed
  once with `place_params` and fed back input-to-output in a same-device loop
  (re-placing per call measured 10–56× slower). Whole-forward latency:
  `tiny_llama` 0.20 ms, `distilgpt2` 1.24 ms, `gpt2` (124M, packed) 7.07 ms —
  4–14× faster than the numpy interpreter and up to ~200× faster than
  iree/llvm-cpu. On CPU the numpy/OpenBLAS path is still the fast one (the
  compiler's value here is the GPU).
- **Compared to PyTorch** (measured; full tables in `./bench_logs/hf_bench_results.md`):
  on an AMD Instinct MI300X VF (gfx942, ROCm 7.1) across five checkpoints
  (`tiny_llama`, `tiny_mistral`, `tiny_gpt2`, `distilgpt2`, `gpt2` 124M; `seq_len` 32
  for all of them, 128 for the two real ones), batch 1, the whole forward as ONE
  graph in the device-resident same-device loop (zero `asdevicearray` per call):
  **`xla`/rocm beats torch-rocm in 6/6 measured cells (7.18×–9.99×)** and is
  21.6×–35.6× faster than torch-cpu on the real models; **`iree`/rocm beats
  torch-rocm in 5/6 (0.73×–9.13×)**, its one loss being `gpt2` at `seq_len=128`.
  XLA's default fp32 `dot` is TF32-class, so the opt-in `dot_precision="HIGHEST"`
  leg is the bit-accurate one (rel-to-absmax 1.5e-6–2.9e-6, argmax agreement 1.0,
  for 3%–29% more latency). On CPU the numpy/OpenBLAS interpreter still wins the
  large matmuls (`gpt2` s32: 28.9 ms interpreted vs 403 ms on llvm-cpu) — the CPU
  and GPU stories are never conflated. Legs that cannot run fail explicitly, never
  silently: `tiny_mistral` on `iree`/`xla` (its `gelu` needs `etl.erf`, a StableHLO
  export v1 deferral), and `distilgpt2`/`gpt2` at `seq_len=128` on iree-llvm-cpu
  (upstream `iree-compile` stack-allocation limit). There is no XLA CPU plugin on
  that host, so the xla legs are ROCm-only — and they are the first `etl_hf` runs
  on the XLA backend anywhere.
- **Opt-in weight packing (`pack_weights=True`)** collapses the whole weight
  tree into ~4 graph inputs by host-side concatenation plus static in-graph
  slices (bit-identical to unpacked). It also works around a real upstream
  limit: IREE's Python runtime SIGSEGVs on llvm-cpu graphs that **consume**
  more than ~161 distinct inputs, which an unpacked 124M model exceeds (196
  weights) — packed, it runs compiled and stable (13/13 identical runs).
- **Deferred by design (explicit errors):** exact-`erf` GELU configs
  (`hidden_act: "gelu"`) are numpy-backend-only (`erf` is a StableHLO v1
  deferral); dynamic shapes, kv-cache/incremental decode, and generation loops
  are out of scope — this is the forward graph, not a serving stack.

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
pip install -e ".[bench]"    # + torch (etl.bench's PyTorch comparison column only)
pip install -e ".[iree]"     # + IREE compiler/runtime adapters
pip install -e ".[tvm]"      # + apache-tvm, jaxlib (its bundled MLIR bindings; `jax` itself is never imported)
pip install -e ".[compilers]"  # + IREE and TVM adapters (xla: user-provided PJRT plugin .so, no pip dep)
pip install -e ".[hf]"       # + torch, transformers — reference oracles for the etl_hf benchmarks only
```

Extras are additive and each enables only its own capability; `numpy` stays the only
hard dependency (`xla` has no extra at all — it loads a user-provided PJRT C API
plugin `.so`, configured with `ETL_PJRT_PLUGIN` or the backend's `plugin_path`
option). None of them installs `etl_hf`: it is a separate top-level package used
from the checkout (see the `etl_hf` section above).

## Test

```bash
pytest
```
