# etl/backends/stablehlo — StableHLO MLIR export utility (v1: export-only)

## Intent

Turn verified EvoXIR (`etl.ir`) programs into **StableHLO MLIR text** so an external compiler can take over: `iree-compile model.mlir -o model.vmfb`.
This module is an **export utility ONLY** — it is NOT a compilable `Backend`, is NOT registered in the backend registry, and never lowers/compiles/loads/runs anything.
The compiler adapters that consume this export — `"iree"`, `"xla"` (PJRT C API via ctypes), `"tvm"` — live in `../adapters/`.

**This file is the current-state contract. All measurements, probe matrices, per-line inventories and host gotchas live in the companion [`./EMISSION-NOTES.md`](./EMISSION-NOTES.md)** — read it before re-measuring or re-litigating a design decision.

Binding contracts: `../../../CONTEXT.md` (staging pipeline, error strategy, non-goals) and `../../CONTEXT.md` (backends package).
The shared backend-option framework — `KNOWN_OPTIONS` union validation, the `ETL_*` env-var table, precedence rules, the `opt_level` machinery — is OWNED by `../../CONTEXT.md` ("Backend options override") and is not restated here; this file documents only the exporter's own option surface.

## API Surface

`etl.backends.stablehlo.export(graph_or_module, options=None) -> str` — the ONLY public name (`__all__ = ["export"]`).

- Accepts a `trace.Graph` (duck-typed via `.module`) or an `etl.ir.Module` directly; anything else ⇒ `TypeError`.
- Calls `module.verify()` FIRST — failures surface as `core.VerificationError`; MLIR is never emitted from invalid IR.
- Returns StableHLO MLIR text (`str`) — compiler input for external tools.
- A deferred or unmapped op ⇒ `core.BackendError` NAMING THE OP, with a message suggesting decomposition or a future adapter; never a silent skip, never partial output.
- Export-only: not registered as a Backend; performs no lower/compile/load/run.
- `options` is a plain dict: unrecognized keys are ignored, recognized keys are VALUE-validated with a loud `core.BackendError` naming the option, the offending value and the accepted forms.

## v1 mapping (data lives in `./ops.py`; binding table from `../../CONTEXT.md`)

Direct mnemonics (etl op → StableHLO, emitted with the `stablehlo.` prefix).
Lookup order (`_SEARCH_ORDER`): `ELEMENTWISE_MAP` → `SHAPE_MAP` → `CONSTANT_MAP` → `CONTROL_FLOW_MAP` → `COLLECTIVE_MAP` → `COMPARISON_MAP` → `DECOMPOSITIONS`.

| Table | etl op(s) | StableHLO |
|---|---|---|
| `ELEMENTWISE_MAP` | add/subtract/multiply/divide/power/remainder/maximum/minimum | `stablehlo.add/subtract/multiply/divide/power/remainder/maximum/minimum` |
| `ELEMENTWISE_MAP` | abs/negate/sqrt/sign | `stablehlo.abs/negate/sqrt/sign` |
| `ELEMENTWISE_MAP` | exp/log/log1p/sin/cos/tan/acos/tanh/floor/ceil | `stablehlo.exponential/log/log_plus_one/sine/cosine/tan/acos/tanh/floor/ceil` |
| `ELEMENTWISE_MAP` | round | `stablehlo.round_nearest_even` (banker's rounding, numpy `round` semantics) |
| `ELEMENTWISE_MAP` | sigmoid | `stablehlo.logistic` |
| `ELEMENTWISE_MAP` | bitwise_and/or/xor; logical_and/or/not | `stablehlo.and/or/xor/not` |
| `ELEMENTWISE_MAP` | bitwise_left_shift | `stablehlo.shift_left` |
| `ELEMENTWISE_MAP` | bitwise_right_shift | `stablehlo.shift_right_arithmetic`; the writer picks `shift_right_logical` when the RESULT dtype is unsigned |
| `ELEMENTWISE_MAP` | cast | `stablehlo.convert` |
| `COMPARISON_MAP` | equal/not_equal/less/less_equal/greater/greater_equal | `stablehlo.compare` + `comparison_direction` attr (EQ/NE/LT/LE/GT/GE) |
| `SHAPE_MAP` | select/broadcast/reshape/transpose/slice/concatenate/pad | `stablehlo.select/broadcast_in_dim/reshape/transpose/slice/concatenate/pad` |
| `SHAPE_MAP` | reduce_sum/max/min/prod | `stablehlo.reduce` (the reduction kind comes from the op's attrs) |
| `SHAPE_MAP` | dot / conv | `stablehlo.dot_general` / `stablehlo.convolution` |
| `CONSTANT_MAP` | constant | `stablehlo.constant` (dense elements attr — see the constant-emission note) |
| `CONTROL_FLOW_MAP` | `if` / `while` | `stablehlo.if` / `stablehlo.while` — the IR names; frontend `cond`/`while_loop` never reach IR under their own names |
| `COLLECTIVE_MAP` | all_reduce/all_gather/reduce_scatter/all_to_all/`broadcast_collective`/collective_permute | `stablehlo.all_reduce/all_gather/reduce_scatter/all_to_all/collective_broadcast/collective_permute` |

The dist collective's IR op name is `broadcast_collective` (NOT `broadcast`), so it does not collide with the shape op `broadcast`; dispatch is purely by IR op name, no effect-kind disambiguation.

## Dedicated emitters (`SPECIAL_EMITTERS` — writer routines, not a single mnemonic)

`gather` · `scatter` · `sort` · `argsort` · `argmax`/`argmin` (family key `arg_reduce`) · `tile` · `eigh` · `diag` · the random family (`random_key_mix`/`uniform`/`normal`/`randint`/`permutation`, key `random`).

- **`gather`** → `stablehlo.gather`, single-axis numpy `take` semantics; the index is reshaped to full rank and `collapsed_slice_dims`/`start_index_map` are built from the gather axis (a NON-leading axis is emitted as the equivalent leading-axis gather via a transpose — see the `_eigh_take` note).
- **`scatter`** → `stablehlo.scatter` with an `update_computation` region; numpy `put_along_axis` semantics — REPLACEMENT, not accumulation.
- **`sort`/`argsort`** → a `stablehlo.sort` of `(value, iota)` with an LT comparator region (ascending; descending = `stablehlo.reverse` after, which also reverses ties — a descending argsort's tie order is pinned as `np.argsort(x)[::-1]`, NOT numpy's stable order).
  The alternative **count** composition computes argsort WITHOUT `stablehlo.sort`: two broadcast-compare rank sums + an iota tie-break + `reduce(add)` over the k axis, the inverse permutation via a pos-EQ mask + select-sentinel + `reduce(min)`, and the sorted keys via masked select + `reduce(min)` with ±inf/`iinfo` sentinels.
  Selection is the exporter option **`sort_emission`** = `"pair"` (default) | `"count"` | `"auto"` (see the option surface).
  The count composition is the DEFAULT for cuda targets because the 1-operand `stablehlo.sort` (which the VALUES sort emits) lowers on iree-cuda to a single-threaded O(n²) kernel; count-mode values and argsort are bit-exact there and ~120–134× faster, while llvm-cpu's count values-composition is ~1.6× SLOWER than the 1-operand sort above n≈10000, hence iree's `"auto"` upgrade applies to cuda targets only.
  Count mode is O(n²) parallel — beyond n≈100k prefer an argmin-style shortcut or an upstream iree sort-codegen fix.
- **`argmax`/`argmin`** → a two-operand `stablehlo.reduce` over `(value, iota)` with an index tie-break comparator (first occurrence on ties, matching `np.argmax`/`np.argmin`).
  The comparator is NaN-AWARE for float keys: `wins = (isnan(a) ∧ ¬isnan(c)) ∨ (a ≷ c) ∨ ((a = c ∨ (isnan(a) ∧ isnan(c))) ∧ idx_a < idx_c)`, `isnan(x)` = `compare NE(x, x)`; NaN therefore beats every non-NaN and the smallest index wins among NaNs, so ANY reduction-tree association (single-stage or split) yields numpy's first-NaN index.
  Never express the equal-class term as `NOT(a NE c)` — two NaNs are NE.
  `nan_aware` is derived from the POST-promotion key dtype: float keys take the NaN-aware form, int/bool keys keep the plain comparator (their text is unaffected by the option).
  One comparator REGION factory serves both the single-stage and both split stages: `split_reduce._arg_reduce_region(writer, key_elem, direction, nan_aware=...)`.
- **`tile`** → reshape + `broadcast_in_dim` + reshape decomposition.
- **`diag`** → rank-2: flatten reshape + constant-index single-axis gather (the main diagonal); rank-1: iota-EQ mask + `select` (the diagonal matrix).
  Dtype preserved in both directions including complex (the mask compares iotas, never data); EXACT vs numpy.
  Any dynamic dim ⇒ explicit `BackendError` (the constant index offsets need static shapes). There is no `stablehlo.diag` op.

### `eigh` — two cyclic-Jacobi `while` compositions

StableHLO 1.0 removed `stablehlo.eigh`/`qr`/`svd` (iree 3.11 ships only `cholesky`), so `eigh` is emitted as a symmetric eigensolver: ONE `stablehlo.while` plus a SHARED post-loop tail (`_emit_eigh_finish`: a stable pair-sort for the ASCENDING eigenvalue order, then a gather column reorder of `V`).

- **`eigh_algorithm="jacobi_batched"` (default)** — `./eigh_batched.py` (`EighBatchedMixin`): the loop carries `(a, v, k)` with cond `k < sweeps·L`; each iteration applies ALL `m` mutually disjoint rotations of one round of the odd-even ("circle") round-robin schedule from `eigh_batched_schedule(n) -> (L, m, fixed)`.
  Even `n` ⇒ `L = n-1`, fixed vertex `n-1`, `m = n/2` slots; odd `n` ⇒ `L = n`, no fixed vertex, `m = (n+1)/2`, slot 0 of every round a BYE (a forced-identity pseudo-rotation with `p == q == r`), so `L` rounds always yield exactly `n(n-1)/2` real rotations per sweep.
  The round is expressed through partner gathers plus per-slot/per-index vectors derived arithmetically from hoisted iotas, so the body contains NO scatter region and its op count is constant in `n`.
  The round counter reduces with `stablehlo.remainder` on provably non-negative numerators, never with a scatter region.
- **`eigh_algorithm="jacobi"`** — `writer._emit_eigh_jacobi`: the loop carries `(a, v, p, q, k)` with cond `k < sweeps·n(n-1)/2`; one iteration applies ONE `(p, q)` rotation, with dynamic `(p, q)` indexing via gathers (never `stablehlo.slice` — its start indices must be static).
- **Selection** — batched iff the option is `"jacobi_batched"` AND `n >= 2`; `n < 2` has no off-diagonal and keeps the sequential composition. The rank/square/dtype/dynamic-dim validation is SHARED by both paths with unchanged wording.
  `jacobi_batched` is the default because it is faster at every measured size (llvm-cpu 4.0×/29.2×/74.4× at n=10/50/100; xla/rocm 5.7–65× up to n=100) and it is the only composition iree/rocm can run — the sequential one SEGFAULTs or hangs there.
- **Sweep schedule** — adaptive, dimension-based: 3–7 sweeps for f32 by size, 8 for f64, in both compositions.
- **Layout** — the carried matrices live transposed in `(n, n, batch...)`; A takes a dim-1 then a dim-0 half-step `X' = c·X + s·take(X, partner, axis)`; the carried `V` is `Vᵀ` (a dim-1 V step would break the eigenvector pairing) — which is why the post-loop tail is shared unchanged.
- **`_eigh_take`'s transpose dance is REQUIRED, not stylistic**: a gather with a non-leading `collapsed_slice_dims`/`start_index_map` makes iree-compile fail at every size with a `vector.transfer_read` mask-type mismatch; the take transposes the target axis to the front, gathers with `collapsed_slice_dims=[0]`, and transposes back. Do not simplify it away.
- **dtype rule** — mirrors the numpy kernel: int/bool → f64; f32/f64 pass through; complex/f16 ⇒ explicit `BackendError`. Dynamic dims ⇒ explicit `BackendError`.
- **Parity contract** — fp32-tolerance, NOT bit-exact. Tests assert `w` ascending, `A ≈ V diag(w) Vᵀ` and `VᵀV ≈ I`, and `w` vs `numpy.linalg.eigvalsh` within the documented fp32/f64 bounds; they NEVER compare elementwise `v` (eigenvectors of a degenerate eigenspace are basis-dependent).
- **Text property** — O(1) in `n` for both compositions: the loop body is O(1) ops, so the line/op count is constant within each parity (`eigh_algorithm` mode × even/odd `n`); only the hoisted O(n²) eye (and, under `eigh_early_exit`, the i≠j mask) grows in bytes. The runtime stays O(n⁴).
- **Constants/iotas are hoisted OUTSIDE the loop** — iree 3.11 hoists a region-local `stablehlo.constant` into an undefined `util.global` (`'util.global.load' op undefined global: @__hoisted_*`); see the region contract.
- Under the `eigh_early_exit=True` exporter option (default **False**) either composition additionally carries an i1 `done` flag; the cond region always stays the plain `k < total` compare.
  It defaults OFF because it is a NET LOSS on random matrices in both compositions and it makes iree-cuda crash at invoke (dealloca `INVALID_ARGUMENT` at n=10, SIGSEGV at n=50), so it is opt-in and effectively llvm-cpu-only; its calibration tolerance is f32 3e-5 / f64 1e-13.

### Random ops — algorithm-aware expansion

`random_key_mix`/`random_uniform`/`random_normal`/`random_randint`/`random_permutation` emit via `./random_export.py`.

- `algorithm="splitmix64"` (the frontend default) → the inline SplitMix64 i64 subgraph; uniform/randint/permutation are EXACT vs the numpy kernels.
- `algorithm="threefry2x32"` / `"philox4x32_10"` → TWO paths, chosen PER ALGORITHM by the `rng_bit_generator` exporter option: NATIVE `stablehlo.rng_bit_generator` (`rng_algorithm = THREE_FRY` / `PHILOX`, key-first state layout) when the algorithm name is in the option set, else the bit-exact INLINE i32/ui32 cipher expansion (20-round Threefry2x32 via shift/or rotate composition; 10-round Philox4x32-10 with an i64 mulhilo chain mirroring the kernel's cyclic 4-word round-key schedule).
- The word-stream contract matches the numpy kernels BIT-FOR-BIT: per-op pi-hex salts XOR-folded onto the key words (threefry → words 0/1, philox → 0/1 and 2/3), counter `(p, 0)` per cipher call/block, element `i` ← flat word `i`, normal pairs words `2i`/`2i+1`; 32-bit-word uniform scaling is the FULL-word `w · 2⁻³²` (the `(w >> 11) · 2⁻⁵³` form is splitmix64-only).
- `random_normal` with an f32 output takes a documented f32 Box–Muller fast path (~1e-6 scale budget vs the f64 numpy kernel, same-key bit-identical across runs); f64/f16 output keeps the exact f64 math.
- NEVER `stablehlo.rng` — its implementation-defined algorithm would break the same-key ⇒ same-values determinism contract.
- `random_multinomial` is DEFERRED for ALL algorithms (cumulative-search decomposition not wired).
- `scan` has **no IR OpDef**: `etl.scan` desugars at trace time into while + gather + scatter (all v1), so scan graphs export and run on iree-llvm-cpu.

## Decompositions (`DECOMPOSITIONS` — expansion into ordinary sub-ops)

| etl op | Expansion |
|---|---|
| `square` | `multiply(x, x)` |
| `relu` | `maximum(x, 0)` |
| `stop_gradient` | identity passthrough (emit the operand directly) |
| `reduce_mean` | reduce-sum, then divide by the element count (divisor via a static constant or a `dynamic_broadcast_in_dim` when non-reduced dims are dynamic) |

## Deferred in v1 (`DEFERRED_OPS` — ⇒ `core.BackendError` naming the op)

`runtime_call`, `external_call`, `block_call`, `rank`, `world_size` (dist graph scalars), `erf`, `gelu`, `diagonal`, `cholesky`, `qr`, `matrix_rank`, `svd`, `matrix_exp`, `random_multinomial`, and the 16 `sparse_*`/`dense_dot_sparse` ops.

- `erf`/`gelu`: no `stablehlo.erf` exists (erf is CHLO-only); gelu's erf-based decomposition defers with it — no silent approximation. `cholesky`/`qr`/`svd` have StableHLO counterparts but are not wired; `matrix_rank`/`matrix_exp` need SVD-cutoff / Padé decompositions.
- `external_call` defers at the EXPORTER level by design: the iree adapter SPLITS external-call graphs BEFORE export (see `../CONTEXT.md`); direct-export users see the deferral.
- Unmapped names (`call`, `tril`, `triu`, `cumsum`, `solve`, …) also count as deferred via `status()` — there is no v1 mapping for them.
- Complex-number elementwise beyond `cast` is additionally deferred (a dtype check in the writer, not an op name).
- `argmax`/`argmin` are NOT deferred — they are composed (see the emitters). Mnemonics were verified against the StableHLO spec at implementation time; an op with no spec mnemonic is deferred rather than emitted with an invented one.

## Dynamic dims — per-op allow/reject matrix (v1; THIS file is the authority cited by `../../CONTEXT.md`)

Symbolic dims render as `?` (int dims literally; `Dim`/`DimExpr`/`None` → `?`, e.g. `tensor<?xNxf32>`). Rank is always concrete, so the emitted rank is exact.
Every REJECT raises at export/`lower()` time with a message naming the op, the shape, the offending dims, and containing "dynamic" — never invalid MLIR reaching a compiler.

**ALLOWED:** elementwise (incl. the `_equalize_operand` / `_scalar_constant_for` dynamic-broadcast paths), `cast`, `select`, `compare`, bitwise/logical, reductions over dynamic dims WITHOUT keepdims reshape, `reduce_mean` whose REDUCED dims are static (dynamic non-reduced dims use the dynamic-broadcast divisor), `transpose`, `concatenate`, `if`/`while`, `constant`, `dot_general` with matched batch structure, and the dynamic-broadcast path.

**REJECTED:** `reshape` with ANY dynamic dim (including the keepdims reshapes inside the reduce/reduce_mean emitters) · `conv` with any dynamic dim in x/w/result · `slice` and `pad` with dynamic dims (iree-compile accepts the MLIR but the runtime ABORTs at every concrete size) · `reduce_mean` whose REDUCED dim is dynamic (element count not statically known) · `broadcast` with a dynamic result and no full-shape source ("dynamic broadcast") · `eigh`/`diag` with any dynamic dim · `dot` batch structure that cannot be emitted (no shape source for the required dynamic broadcast / unprovable symbolic merge).

The rejection helper is `Writer._reject_dynamic_dims(op, shape, what, op_name=None)`; `slice`/`pad` joined the list empirically.
iree/tvm declare `dynamic_shapes=True` but support is PER-OP partial — the exporter is the gate. This is why vmap of softmax/layernorm/attention-style per-sample functions defers on compiler backends.

## Type map (`DTYPE_MAP`; keys are numpy dtype objects, normalized via `numpy.dtype(...)`)

`float16`→`f16`, `float32`→`f32`, `float64`→`f64`, `int8`→`i8`, `int16`→`i16`, `int32`→`i32`, `int64`→`i64`, `uint8`→`ui8`, `uint16`→`ui16`, `uint32`→`ui32`, `uint64`→`ui64`, `bool`→`i1`, `complex64`→`complex<f32>`, `complex128`→`complex<f64>`.

## Exporter option surface

`export()` options are a plain dict; the writer validates VALUES (loud `core.BackendError` naming the option, the value and the accepted forms). `CompilerBackend._exporter_options` forwards exactly the six LOWER-stage keys in the "adapter-plumbed" column and validates option KEYS per adapter `KNOWN_OPTIONS` — a key no adapter declares is rejected there, and the per-stage env spellings live in `../../CONTEXT.md`.

| Option | Default | Values / validation | Reach |
|---|---|---|---|
| `rng_bit_generator` | the caller's `Capabilities.rng_bit_generator` set (numpy `frozenset()`, iree `{"threefry2x32"}`, xla `frozenset()`, tvm `frozenset()`) | bool OR a collection of canonical algorithm names: `True` → both ciphers (legacy), `False`/`None` → empty, a collection is used as-is and unknown names are ignored | adapter-plumbed: iree, xla, tvm |
| `sort_emission` | `"pair"` (Writer); iree overrides the adapter default to `"auto"` and upgrades `"auto"` → `"count"` when the resolved `target_backends` contains `"cuda"` | `"pair"` \| `"count"` \| `"auto"`, case-insensitive, whitespace-stripped | adapter-plumbed: **iree only** |
| `while_init_rewrite` | `True` | bool, coerced by `options.normalize_bool_option` so an env-style `"0"` cannot become `True` | adapter-plumbed: **iree only** |
| `dot_precision` | `"DEFAULT"` | `"DEFAULT"` \| `"HIGH"` \| `"HIGHEST"`, case-insensitive; `"DEFAULT"` emits NO `precision_config` (byte-identical output); `"HIGH"`/`"HIGHEST"` stamp `precision_config = [#stablehlo<precision …>, #stablehlo<precision …>]` on every `dot_general`. Only `"HIGHEST"` disables XLA/ROCm's TF32-class fp32 `dot` (`"HIGH"` does not); iree/tvm parse it and stay bit-identical | adapter-plumbed: all three |
| `eigh_algorithm` | `"jacobi_batched"` | `"jacobi"` \| `"jacobi_batched"`, case-insensitive, whitespace-stripped; `None`/absent ⇒ default; anything else ⇒ `BackendError` naming the accepted values | adapter-plumbed: all three |
| `split_reduction` | `True` | bool, coerced by `options.normalize_bool_option` | adapter-plumbed: all three |
| `split_reduction_threshold` | `1024` | int ≥ 2 (a value below 2 is rejected) | **`export`-only** (no env var, no adapter `KNOWN_OPTIONS` entry) |
| `split_reduction_non_exact` | `False` | bool | **`export`-only** |
| `eigh_early_exit` | `False` | bool | **`export`-only** |

- `sort_emission="pair"` remains the Writer default and the xla/tvm behaviour; only the iree adapter upgrades to `"auto"`/`"count"`, and an explicit per-call value always wins.
- `"auto"` means "pair for sorted-axis extent < 32, count otherwise" (32 is the extent at which the iree-cuda pair-emission bufferization bug bites).
- The three `export`-only options are deliberately not adapter-plumbed: they are diagnostic/experimental knobs, and `split_reduction_threshold`/`split_reduction_non_exact` change numerics or cost in ways the pipeline does not surface.
- Adding a new exporter option: Writer flag + normalizer in `Writer.__init__`, `export()` docstring, an `_exporter_options` entry in `../compiler.py` (+ an optional `default_<option>` `ClassVar`), and each consuming adapter's `KNOWN_OPTIONS["lower"]`.

## Design decisions

- **Verify before export** — `export()` runs `module.verify()` first, so emission never works from invalid IR.
- **Program-order emission** — ops emit in block order, so program order IS effect order (write/read/collective/callback semantics preserved).
- **Data/code separation** — mapping tables live in `ops.py` (auditable against the StableHLO spec without touching emission code); `writer.py` holds only emission logic.
- **Error style** — deferred/unmapped op ⇒ `core.BackendError` naming the op; never a silent skip, never partial output.
- **Batched `dot` emission matches numpy `matmul`** (`_emit_dot`, 5-path dispatch): the frontend guarantees rank ≥ 2, batch dims broadcast per `infer_dot` (equal or size-1).
  Matched batch structure emits directly (byte-identical to the plain form); a plain-matrix rhs with fully static shapes emits a non-batched `dot_general`; static broadcast targets use `broadcast_in_dim`; symbolic targets use `dynamic_broadcast_in_dim` whose output-dimensions vector is built per dim from `get_dimension_size` on the batch source plus static constants (the other operand is NOT a valid full-shape source when its non-batch dims differ).
  A provably all-size-1 rhs batch is squeezed to a plain matrix ONLY when BOTH operands are fully static — iree 3.11 cannot legalize the `dynamic_reshape` its import inserts for a non-batched dynamic `dot_general`, so dynamic shapes fall through to the batched dynamic-broadcast path.
  An unprovable symbolic merge raises `BackendError` containing "dynamic". The result shape is always exactly the declared `infer_dot` result. Rank-1 operands (IR-level only) emit valid non-batched forms (`v·v` → scalar, `M·v`, `v·M`).
- **Promoted-accumulator rule for `sum`/`prod`/`mean` (binding).** numpy's accumulator dtype is fixed by the OPERAND dtype, and `_reduce_core_with_split(x, dims, kind, accum_dtype)` calls `_promote_reduce_operand(...)` — ONE `stablehlo.convert` — ONCE, BEFORE the split plan is computed and before any reduce, so integer `sum`/`prod` never accumulate in a narrow operand dtype (`prod(arange(1,14))` in i32 gives 13! = 6227020800, not the wrapped value; large i32 sums do not wrap).

  | kind | operand dtypes | accumulator |
  |---|---|---|
  | `sum`, `prod` | bool, i8, i16, i32 | `i64` |
  | `sum`, `prod` | int64 | `i64` |
  | `sum`, `prod` | u8, u16, u32, u64 | `ui64` (never a signed narrow type) |
  | `mean` | every int/bool operand | `f64` (the numpy backend sums in int64 before dividing) |
  | `max`, `min` | all | operand dtype (the i1 `any`/`all` guards keep their i1 path) |
  | float kinds | all | unchanged (no promotion, no convert) |

  The trailing convert is a documented unreachable safety net. A consequence: int/bool `mean` at reduced extents ≥ the threshold does NOT take the split path (accumulator f64, and float `sum` is not split by default).
- **`split_reduction` family — two-stage reduction geometry (contract).** A reduction is ELIGIBLE iff `split_reduction` is on AND every operand dim is STATIC (symbolic dims never split — a non-error) AND the reduced dims are non-empty AND `prod(reduced extents) >= split_reduction_threshold` AND `prod(non-reduced extents) <= 1024` (rank-0 or otherwise tiny) AND the kind's exactness class allows it.
  Emission: pad the PIVOT axis (the largest reduced dim) up to `R·C` with `stablehlo.pad` + the reduction identity when the extent is not factorable, `stablehlo.reshape` the pivot extent `N` into `(R, C)`, let stage 1 reduce the inner dims (its result keeps `R`, so it is parallelizable) and stage 2 reduce the `R` axis.
  `(R, C)` = the exact divisor of `N` closest to `sqrt(N)` when one exists, else `R` = the power of two nearest `sqrt(N)` clamped to `[16, 256]` with `C = ceil(N/R)` (`SPLIT_ROWS_MIN`/`SPLIT_ROWS_MAX`/`SPLIT_FACTOR_LIMIT`/`SPLIT_RESULT_ELEMS_LIMIT`). Op count is CONSTANT in the extent.
  `argmin`/`argmax` keep the composite `(value, index)` reduce with the SAME comparator in both stages; the index operand becomes `iota(pivot)·C + iota(pivot+1)` — the ORIGINAL flat index for real elements and `>= N` for pad slots — so the `idx LT` tie-break can never pick a pad.
  **Exactness classes (binding):** `min`/`max` (incl. the i1 bool-guard shapes), `argmin`/`argmax`, and INTEGER/bool `sum`/`prod` split BY DEFAULT (grouping-invariant, bit-identical to the single-stage emission — an all-`±inf`/all-equal result's ±0 SIGN may differ, and `-0.0 == 0.0`). FLOAT `sum`/`prod` — and therefore `reduce_mean`/`var`/`std` — are NOT split by default (reassociation changes bits) and need `split_reduction_non_exact=True`.
  `split_reduction=False` is a HARD byte-identity contract: the emitted text is exactly the single-stage emission.
  **Signed narrow-int guard (`UNSAFE_MIN_EXTENT_MULTIPLE = 64`):** for `min` over a signed int narrower than 64 bits, an upstream iree/ROCm miscompile returns `-1` when a reduce extent is a multiple of 64, so the row count is re-picked until BOTH stage extents escape that class (keeping the single-stage text when nothing escapes). Do not remove or generalize the guard.
  Why the default is ON: IREE's GPU backend lowers any reduce whose RESULT is rank-0 to ONE workgroup of ONE thread; split reduction is 18.1× on an evox-shaped fitness monitor on iree/rocm and 7.9–8.1× per rank-0 op, while being ~1.0× on llvm-cpu/xla.
  Scope: wired into `_emit_reduce`, `_emit_reduce_mean` and `_emit_arg_reduce` ONLY — the internal reduces (count-argsort, the `eigh` convergence energy, `while_init_rewrite`'s zero source) keep their single-stage text.
- **Constant emission path (no size gate)** — the IR `constant` op dispatches to `Writer.render_constant`, which reads the FULL payload from `op.attributes["value"]` and inlines it via `Writer._constant_text`: a recursive `dense<[...]>` string join with no size threshold, no chunking, no streaming, no separate-file serialization, and no path that turns a large constant into a runtime function input. A ~4 MB f32 constant therefore becomes one multi-MB MLIR line, and the whole module text is built in memory. The only value-dependent constant handling in the writer is the `while_init_rewrite` zero-init rewrite, which is size-independent. `etl.constant`'s large-constant warning (`ETL_LARGE_CONSTANT_BYTES`) lives in `etl/ops`, outside this node.
- **Reduce semantics vs numpy (live facts).** `np.min`/`np.max` PROPAGATE NaN, and `stablehlo.minimum`/`maximum` on iree match `np.minimum`/`np.maximum` pairwise exactly, so min/max agree with numpy on NaN and ±inf.
  Signed zeros DIVERGE: iree's `stablehlo.reduce` is IEEE-ordered (a `min` of any mix containing `-0.0` → `-0.0`; a `max` containing `+0.0` → `+0.0`) while numpy returns the last-visited operand — stable across single- and two-stage emission, so re-association adds no new ±0 classes but does not fix the existing one.
  `np.min`/`np.max` of an EMPTY reduction raise `ValueError` while the emitted reduce returns its init (`∓inf` / `iinfo.min|max`); `np.sum`/`np.prod` of empty give the identity and match.

### Region contract for `if` / `while` regions (four independent rules, all binding)

1. **Invariant lift + dominance** — a region's region-invariant constants/iotas are LIFTED to the enclosing scope (`_split_region_invariants`/`_region_lines`), because iree 3.11 hoists a region-LOCAL `stablehlo.constant` into an undefined `util.global` (`'util.global.load' op undefined global: @__hoisted_*`). A unit that touches a region-local or sibling-region SSA name (a region block arg, anything defined in this block or the sibling region, anything a nested region defines) STAYS in the region; lifting bubbles outward one level at a time, so an `eigh` inside a cond inside a cond ends with its constants at function scope.
2. **No `stablehlo.sort` in a region or an inline branch scope** — a 2-operand `stablehlo.sort` there is mis-executed (the index column is correct but the gather of the same operand comes back UNSORTED), so every sort/argsort emitted at `_region_depth > 0` OR `_select_inline_depth > 0` uses the COUNT composition (`_count_argsort_active`/`_count_sort_active`). A flat top-level block keeps the mode's own `sort_emission` choice, so bare exports are unaffected.
3. **No `stablehlo.while` inside `stablehlo.if`** — a `cond` whose branch regions contain a while-emitting op (`_WHILE_EMITTING_OPS = frozenset({"while", "eigh"})` — `eigh`'s while is SYNTHESIZED inside its emit text, so it has no IR region) is emitted by `_emit_if_select`, NOT `_emit_if`: both branch bodies are emitted INLINE at the enclosing scope and each result becomes `stablehlo.select(broadcast_in_dim(pred) -> tensor<...xi1>, true_i, false_i)` (the broadcast is skipped for rank-0 results). Both whiles therefore sit at function/enclosing-block level — the shape a bare `eigh` emits, which iree executes correctly. Region-free and reduce-only conds keep the `stablehlo.if` emission byte for byte (a `reduce`/`sort` comparator region inside an `if` is NOT affected). The trigger is derived from the IR, so no writer state flag is involved, and `_region_depth` is NOT incremented on the inline path.
4. **Consequences (one line each)** — the count composition costs ~+37 lines / +2.7–2.9 KB of text for a region-nested sort and ~+2–4 % run time for a while-body `eigh`; the select fallback makes a while-bearing `cond` cost the SUM of both branches (both execute unconditionally, ~2× a bare `eigh`) because every op reaching the exporter is pure — effectful ops defer — so only the selected value reaches consumers.
   The root cause is iree 3.11's `local-task` driver mis-executing a `stablehlo.while` nested in a `stablehlo.if`: the SAME compiled module is correct under `local-sync`; capture topology is NOT the cause.

## Known upstream crash classes (current state)

These are LIVE facts about iree 3.11 (and one 3.9.0 class below); the probe evidence and per-variant rates are in `./EMISSION-NOTES.md` §16–§17.

- **iree 3.11 `stablehlo.while` SEGV — AffinityAnalysis DFX solver recursion** (`CompilerToolError "Error code: -11"` inside `IREE::Stream::AffinityAnalysis::run()`). Two independent triggers:
  **T1 = a dense rank≥1 constant as a while-loop INIT operand.** Handled at the exporter level by the `while_init_rewrite` option (default ON): `_emit_while` replaces every all-zero rank≥1 `stablehlo.constant` init with COMPUTED zeros — `not(x)` / `and(x, not(x))` over a NON-constant pre-loop value plus a dtype-changing `convert` (plus a `reduce` when the source is one axis larger). Never `x - x` and never a broadcast of scalar 0 — iree's canonicalizer folds both back into constants.
  Source preference (`_find_while_zero_source`) is SAFETY-ordered, not cosmetic: (1) the REDUCE path — an input-derived bool/int value one axis larger than the target, reduced along the extra axis — is the ONLY class validated safe at real-graph scale; (2) exact-shape sources and (3) the `iota` fallback are small-graph-safe only. Non-zero vector constants and rank-0 inits are left untouched (a rank-0 init is unaffected, and a non-zero vector constant has no valid replacement — such loops remain broken on iree).
  **T2 = a `broadcast_in_dim` whose operand is a CONVERT of a loop-carried value, consumed directly by a non-convert op.** No exporter fix: loops with that shape need user-side re-expression (a dtype-changing convert BETWEEN the broadcast and its consumer, or the unrolled fixed-point / xla route).
  Companion bug (distinct, 100 % reproducible): any constant tensor referenced from a while body/cond fails compile deterministically with `'util.global.load' op undefined global: @__hoisted_*`, even when loop-carried — while loops must reference only captured SCALARS or carried values.
- **iree 3.9.0 `stablehlo.while` SEGV — the loop-carried-COUNT class.** A while loop's crash-proneness scales with its CARRY COUNT and with `select` ops in its body, independently of the T1/T2 trigger shapes above. Live facts: the NSGA3 tell graph (5 whiles: 4/4/5/5/7 carries, selects in every body) SIGSEGVs; 3-carry whiles (NSGA2) compile, including inside full modules; standalone 3- and 4-carry whiles with one select compile; a 7-carry + 2-select while SEGVs standalone.
  In the full NSGA3 module the 4-carry front-peel while is the unique NECESSARY while (removing it alone kills the SEGV; the same loop standalone compiles — module context is part of the trigger), and rewriting its `select` to arithmetic does NOT dodge it.
  `--iree-stream-affinity-solver-max-iterations=1` converts the SEGV into a clean deterministic `failed to solve for affinity analysis` (a fail-loud diagnostic, NOT a working compile).
  The exporter is carry-count-agnostic: `_emit_while` emits every operand positionally with no count gate, so this is not exporter-fixable by a carry-count threshold — the routes are the unrolled fixed-point rewrite or the xla adapter. Candidate exporter workarounds (arithmetic-`select` rewrite of while bodies; same-dtype-group carry packing) are documented in `./EMISSION-NOTES.md` §17 and are NOT implemented.

### Loop-carried-count crash observation (upstream iree AffinityAnalysis)

This is the subsection the `etl/trace/CONTEXT.md` "Loop-CARRIED-COUNT crash observation" citation points at; the same facts appear in the bullet above.
The carrier here is `_emit_while`'s operand list, and the exporter applies no carry-count gate — what separates a 3-carry loop that compiles from a ≥4-carry loop that SEGVs is inside iree's `AffinityAnalysis` solver, not in the emitted text.
Current observed boundary: 3-carry whiles compile (NSGA2, including inside full modules); the ≥4-carry front-peel while SEGVs inside a large module (the NSGA3 tell graph) though it compiles standalone, so module context is part of the trigger; a 7-carry + 2-select while SEGVs standalone.
Reducing carries therefore means USER-SIDE graph changes (e.g. recompute a derived carry such as `dominates = (count == 0)` locally in the cond/body instead of carrying it) — there is no CSE/DCE/region simplification anywhere in etl that would do it for you.
A carry-rewrite could only live ABOVE the exporter (in `etl.transforms` or `CompilerBackend.lower`), because the writer accepts packed and unpacked whiles alike; the probe matrices, per-variant crash rates and the two unimplemented candidate workarounds are in `./EMISSION-NOTES.md` §17.
- **`sort`/`argsort` on iree-cuda** — the pair emission's `stablehlo.sort` fails bufferization deterministically when the SORTED AXIS SIZE ≥ 32 (no multi-operand sort variant works there), and the 1-operand VALUES sort lowers to a single-threaded O(n²) kernel at any axis size. Both are handled by `sort_emission` (count composition), not by a code change here.
- **iree/rocm `while`/`scan`** — the HIP HAL aborts at run time on certain loop shapes (loop carry initialized from a graph CONSTANT; condition reading a float carry). Same graphs are llvm-cpu-clean. See `../CONTEXT.md` Known Issues.
- **Sequential `eigh` at n ∈ {4, 8, 12, 16}** crashes at compile on both llvm-cpu and cuda (`command_buffer_validation.c … OUT_OF_RANGE`, vectorization-threshold-dependent); the batched composition builds and runs clean at all four sizes. Sizes outside that family work in both compositions.

## Constraints (binding)

- Top-level imports restricted to `etl.core` and `etl.ir`; NEVER import `etl.pipeline` (cycle). `trace.Graph` is duck-typed via `.module` (no `etl.trace` import). numpy is allowed (`DTYPE_MAP` keys). `writer.py` uses `TYPE_CHECKING` imports for `etl.ir` annotations.
- Export-only: never registers with the backend registry; no lower/compile/load/run.
- CPU-neutral: MLIR text only, no device interaction.
- Files < ~1000 lines, with verified exceptions that are legitimately long and must not be split without a plan: `writer.py` **4896 lines** (the single emission file — mapping data lives in `ops.py`, and the batched `eigh` composition is already extracted into `eigh_batched.py` as a mixin), `random_export.py` **1445 lines** (per-algorithm cipher expansions; coherence is lost if it is fragmented), `split_reduce.py` **801 lines**, `eigh_batched.py` **649 lines**, `ops.py` **376 lines** (data only), `__init__.py` **183 lines**.

## Test strategy

`../../../tests/backends/` (sibling — read-only from here; test-related writes escalate to root).

| File | Collected | Pins |
|---|---|---|
| `test_stablehlo.py` | 121 | golden-text exports for the whole v1 table (elementwise incl. the shifts' signed/unsigned split, comparisons + direction attr, reduce, dot/conv, if/while, collectives + `collective_broadcast`); symbolic-dims rendering; the four decompositions; every deferred op ⇒ `BackendError` naming the op; the `eigh` compositions (batched default golden, sequential reference golden, early-exit goldens, `n < 2` fallback, dtype/`dynamic`-dims rejection) and `diag`; `verify()` failure ⇒ `VerificationError`; non-Graph/non-Module ⇒ `TypeError` |
| `test_options_override.py` | 96 | the exporter option surface end-to-end: defaults, case/whitespace spelling, invalid values, and each option's reach through `lower(...)` on the adapters |
| `test_split_reduction.py` | 67 | the three `split_reduction*` defaults/validators, the eligibility gates (dynamic dims, below/inclusive threshold, large result, the 1024 boundary, no reduced dims), the pad/reshape/two-stage markers and per-shape goldens, the `split_reduction=False` byte-identity contract, and that the exact class splits by default while float `sum`/`mean` does not |
| `test_iree_split_reduction_parity.py` | 127 | numpy parity through iree-llvm-cpu for the split geometry and both exactness classes |
| `test_iree_emitters_parity.py` | 50 | parity vs numpy on iree-llvm-cpu for gather/sort/argsort/argmin/argmax/tile/scatter, the shifts, and the random draws (uniform/randint/permutation bit-exact, f32 normal tolerance-based, same-key determinism), plus scan-through-while graphs; 5 GPU-guarded iree-cuda smoke tests |
| `test_iree_eigh_diag_parity.py` | 24 | `diag` EXACT vs numpy (rank-1 / square / rect); `eigh` contract checks (ascending `w`, `A ≈ V diag(w) Vᵀ`, `VᵀV ≈ I`) + `w` vs numpy within tolerance, at 3×3 / 10×10 / (2,3,3) batched / n=25 f32, f64 3×3, int32→f64, plus the dim-45 early-exit contract and ON≡OFF equivalence |
| `test_count_argsort.py` | 9 | `sort_emission`: count mode emits NO `stablehlo.sort`, `"auto"` picks count iff the sorted-axis extent ≥ 32 (including via the iree adapter default), llvm-cpu parity, and a GPU-guarded axis-64 argsort on iree-cuda |
| `test_while_init_rewrite.py` | 4 | the `while_init_rewrite` pin: the rewrite chain in the MLIR, 10/10 zero-SEGV llvm-cpu compiles + bit-exact run on a const-init while graph, the real evox `non_dominate_rank` shape bit-exact, and the `while_init_rewrite=False` opt-out |
| `test_nsga2_xla_route.py` | 2 (gate-skipped without a PJRT plugin) | the xla route: an axis ≥ 32 argsort builds and runs bit-exact on the xla adapter |

**Region-nested emission pins** (`test_iree_eigh_diag_parity.py`): `test_eigh_inside_cond_contract[n3|n10|n25]` covers the while-bearing-cond select fallback end-to-end (both branch directions), `test_eigh_inside_while_body_contract` the while-body case, `test_eigh_inside_cond_runtime_predicate` the runtime-bool-predicate path, and `test_eigh_inside_cond_nested` cond-inside-cond. A regression in the region rules surfaces as an iree compile error (`@__hoisted_*`, dominance) or mis-executed values.

CPU only, pytest, numpy-only deps. Anything requiring a real GPU or a PJRT plugin is guarded and skips.

## Routing table

| Path | Area |
|---|---|
| `./ops.py` | v1 mapping tables (data) + `lookup_mapping`/`status`/`is_supported`/`mlir_dtype` — the auditable mapping source of truth, plus the `_SEARCH_ORDER` that resolves the `broadcast`/`broadcast_collective` name split |
| `./writer.py` | StableHLO MLIR text emission: `Writer(EighBatchedMixin, SplitReductionMixin)` — the emission dispatch, the dynamic-broadcast path, batched-dot batch alignment, `_reject_dynamic_dims` and the per-op dynamic-dims validation, the gather/scatter/sort/argsort/arg_reduce/tile/diag emitters, the count-sort compositions, `_emit_eigh_jacobi` + the shared `_emit_eigh` dispatch + the `eigh_early_exit` sweep check, the `_reduce_core_with_split`/`_emit_split_arg_reduce` hooks, the promote-before-reduce helper, every option normalizer, and the REGION machinery (`_region_depth`/`_select_inline_depth`, `_block_body_lines`/`_split_region_invariants`/`_region_lines`, `_emit_if_select`, `_emit_while_zero_init`) |
| `./__init__.py` | `export()` — the only public entry point (plus its `options` docstring) |
| `./eigh_batched.py` | the BATCHED cyclic-Jacobi `eigh` composition: `EighBatchedMixin` (`_emit_eigh_batched`, the shared post-loop tail `_emit_eigh_finish`, the shared `_emit_eigh_cond_region`, the rotation-map/index/gather/half-step/take helpers) + `eigh_batched_schedule(n) -> (L, m, fixed)`; mixed into `Writer`; imports `etl.core` + numpy only |
| `./split_reduce.py` | the two-stage (split) reduction emission: `SplitReductionMixin` (`_reduce_core_with_split`, `_promote_reduce_operand`, `_emit_split_arg_reduce`, the pad/reshape/two-stage helpers), the `_arg_reduce_region` NaN-aware comparator factory, the `normalize_split_reduction*` validators, and the geometry helpers (`split_extent`/`split_stage_extents`/`split_extent_avoiding_multiple`) with the `UNSAFE_MIN_EXTENT_MULTIPLE` guard; imports `etl.core` + numpy + stdlib only |
| `./random_export.py` | algorithm-aware random-op expansion: SplitMix64 inline i64 subgraphs, the threefry2x32/philox4x32_10 inline i32/ui32 cipher expansions, and the native `rng_bit_generator` path — all bit-exact vs the numpy kernels |
| `./EMISSION-NOTES.md` | companion: measurement logs, probe matrices, per-line inventories, repro-artifact paths, candidate (unimplemented) workarounds, and the host gotchas — including §17 "Loop-carried-count crash observation (upstream iree AffinityAnalysis)" |

## Notes for agents

- **Six `"stablehlo.reduce"` emission sites, not one** — any reduction change must decide per site: `_emit_reduce_core` (shared by `_emit_reduce` and `_emit_reduce_mean`), `_emit_arg_reduce`, the count-argsort reduces `_emit_count_reduce` and `_emit_count_reduce_min`, the `eigh` convergence energy `_emit_eigh_reduce`, and `_emit_while_zero_init`'s zero source. `_emit_reduce_region` is NOT one of them (it builds only the collective reduce regions). Current line numbers are in `./EMISSION-NOTES.md` §13 — they drift, so re-grep `"stablehlo.reduce"` rather than trusting them.
- **Several non-frontend reduce sites can cross the "reduced extent ≥ threshold" gate too** — count-argsort reduces over the k axis (extent = the sorted-axis size) and the `eigh` energy reduce over n² elements.
- **Reduce emission order is fixed:** promote → split plan → emit. A split plan computed from the un-promoted dtype picks the wrong class for int/bool `sum`/`prod`/`mean`.
- **`scan` has no IR OpDef** — do not look for one; it desugars in `etl.trace`.
- **iree 3.11 cannot legalize `dynamic_reshape`** for a non-batched dynamic `dot_general`; that is why every static-only fast path in `_emit_dot` is gated on FULLY static shapes rather than on a per-dim check.
- **`_emit_dot`'s helper layout** — all `_dot*` helpers live between `_emit_dot` and `_emit_conv`; see `./EMISSION-NOTES.md` §19.
- **`iree`'s declared dtypes exclude uint64** at the function signature, so u64 values must be constructed INSIDE the graph via i64 bits + masking (as the shift u64-logical parity test does) — u64 tensors cannot be graph inputs/outputs on iree.
- **`stablehlo.sort` tie order is not guaranteed to match numpy's stable argsort** — the descending path reverses ties too; the parity tests pin the descending-argsort tie-break as `np.argsort(x)[::-1]`.
- **Host gotcha — stale `etl` from another checkout**: a validation script run from outside this worktree can silently `import` a different etl checkout unless it does `sys.path.insert(0, "<worktree path>")` first, which makes an unrelated upstream failure look current. Details in `./EMISSION-NOTES.md` §20.
- `../../` = `etl/backends/`, `../../../` = repo root (the `../../../tests/backends/` paths above are correct from this node).
