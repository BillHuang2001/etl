# etl.sparse — explicit sparse tensors

## Intent

A minimal, explicit sparse-tensor value model + graph-time frontend ops, on
the same **spec → symbolic → concrete** three-phase discipline as the dense
core — ALL phases share ONE class hierarchy rooted at `SparseTensor`
(`is_sparse(x)` catches every phase and variant). A sparse tensor is an
explicit `(structure, values)` pair with a known dense shape. There is NO
eager mode for the computation ops and NO hidden densification: every
conversion is an explicit op (or the explicit frontend auto-conversion of
CSR/CSC inputs to the computation format, COO).

## API surface (exact names)

- Value model (`value.py`): `SparseTensor` (base class AND concrete COO
  representation), `CSRTensor`, `CSCTensor`, `SparseTensorSpec`, `is_sparse`.
- Creators (CONCRETE, eager numpy — no trace): `coo(indices, values, shape)`,
  `csr(indptr, indices, values, shape)`, `csc(indptr, indices, values,
  shape)`, `from_dense(dense, format="coo")` (exact `np.nonzero` extraction;
  only `format="coo"` in v1 — others raise `ValueError`).
- Converters (POLYMORPHIC: concrete instance → eager method; symbolic →
  graph op): `to_dense(x)`, `to_csr(x)`, `to_csc(x)`, `to_coo(x)`.
- Computation ops (graph-time; symbolic sparse in): `add`, `subtract`
  (composed `add(a, negate(b))`), `multiply`, `multiply_dense(a, dense)`,
  `negate`, `sum` (alias `reduce_sum`) → DENSE result, `transpose(a,
  perm=None)`, `reshape(a, new_shape)`, `concatenate(operands, axis=0)`,
  `matmul(a, b)` — sparse@dense → `sparse_dot_dense`, dense@sparse →
  `dense_dot_sparse`, sparse@sparse → `TraceError` (v1 deferral), dense@dense
  → `etl.ops.dot` (lazy import, no cycle).
- `SparseTensorSpec.from_concrete(concrete_sparse) -> SparseTensorSpec` —
  derives the canonical spec from a CONCRETE sparse instance (used by
  `etl.pipeline.evaluate` for concrete sparse args).

## Pytree contract (binding for trace/pipeline/transforms)

ONE registered pytree base: `core.register_pytree_node(SparseTensor,
flatten_fn, unflatten_fn)` — the MRO walk makes `TreeSpec.type ==
SparseTensor` for EVERY phase/variant (spec / symbolic / concrete COO / CSR /
CSC).

**Children layout (context always `None`):**
- COO: `[indices, values, *dense_shape, dtype, "coo"]`
- CSR: `[indptr, indices, values, *dense_shape, dtype, "csr"]`
- CSC: `[indptr, indices, values, *dense_shape, dtype, "csc"]`

`dense_shape` contributes ONE leaf per dim (int, or a `core.Dim` after
vectorize), then the values `np.dtype` leaf, then the format `str` leaf. The
static leaves are snapshotted and run-validated by the existing
trace/pipeline machinery: the FORMAT leaf makes coo-vs-csr run-time mismatches
fail loudly in `Graph.flatten_inputs`, and dense_shape/dtype leaves fail on
mismatch. The layout is IDENTICAL across phases (structure-match; only the
leaf kinds differ).

`unflatten` dispatches on `children[0]`'s kind: `TensorSpec` → validated
`SparseTensorSpec`; `SymbolicTensor` → symbolic instance via
`SparseTensor.from_parts` (no validation); `Tensor`/`ndarray` → concrete
instance (canonical validation applies). The format leaf picks the variant.

## Constraints

- **Canonical form enforced in the concrete constructors** (never silent;
  `core.ShapeError`/`core.DTypeError`): COO = int64 `(nnz, ndim)` indices,
  lex-sorted row-major, strictly unique (duplicate NONZERO rows error;
  duplicate rows with one stored zero are legal), in-range per column;
  CSR/CSC = rank-2, `indptr[0] == 0`, monotone non-decreasing,
  `indptr[-1] == nnz`, per-row/column indices sorted strictly increasing and
  in-range. Integer index dtypes normalize to int64. `dense_shape` = positive
  ints (a `core.Dim` allowed ONLY at position 0 — batched/vmap outputs; its
  extent is `indices.shape[0]` for in-range checks and `to_dense`).
- **`nnz` is runtime-dynamic** in every graph leaf spec (`(None, ndim)` /
  `(None,)`); indptr specs stay STATIC `(rows+1,)` / `(cols+1,)` i64 —
  `sparse_coo_to_csc` emits the STANDARD column-pointer length `cols+1`
  (matches `infer_sparse_coo_to_csc`).
- **COO is the computation format**: any CSR/CSC SYMBOLIC input to a
  computation op (add/subtract/multiply/negate/sum/transpose/reshape/
  concatenate/matmul-sparse-side/multiply_dense) or to `to_dense` is first
  converted by emitting `sparse_csr_to_coo` / `sparse_csc_to_coo` (rank-2
  only, `ShapeError` otherwise). `to_csr`/`to_csc` emit
  `sparse_coo_to_csr`/`sparse_coo_to_csc` on COO input. Concrete converters
  are pure layout conversions (canonical COO is row-major, so COO→CSR needs
  no sort).
- **Batched sparse** = leading batch dim on the tensor leaves with
  `dense_shape` UNCHANGED at the input I/O boundary; vmap OUTPUTS get
  `dense_shape = (batch_dim, *dense_shape)` via the batched-aux-remap
  registry (`rules.py` registers `SparseTensor`'s remap). Public
  `etl.vmap`/`etl.vectorize` accept bare `in_axes=0`/`None` AND pytrees: a
  LEAF axes entry at a registered-node position broadcasts across the node's
  tensor leaves (statics → None); a container entry there raises
  `TransformError`. Vectorizing a GRAD graph over sparse hits the
  pre-existing transforms gap "cannot batch op 'gather' with dynamic index
  dims".
- Batched graph-side `sparse_from_dense`: per-batch stored-zero PADDING rows
  to a common nnz (lex-max row, zero values); duplicate-with-zero rows are
  legal (kernels validate values-aware) — documented, no special handling.
- **No sparse constant in v1**: `etl.constant(concrete_sparse)` raises
  `core.TraceError` via the existing guard in `etl/ops/constant.py`
  ("etl.constant expects a concrete core.Tensor, got SparseTensor").

## Known issues / v1 deferrals (explicit errors, never silent)

- sparse @ sparse matmul → `TraceError` (densify one operand with
  `etl.sparse.to_dense`).
- Whole-sparse `etl.bind` is not supported in v1 (bind's per-leaf
  spec validation does not cover the sparse static leaves); leaf-level bind
  works.
- Compiler backends: the 16 sparse ops are numpy-backend-only in v1;
  stablehlo export / iree / xla / tvm defer with an explicit `BackendError`
  suggesting `etl.sparse.to_dense` — see `etl/backends/CONTEXT.md` Known
  Issues.
- `scan` over sparse `xs` / sparse `y`-stacking is a v1 deferral (explicit
  `TraceError`); sparse loop CARRIES through `scan`/`cond`/`while_loop` work
  (trace-level support is generic registered-pytree handling — see
  `etl/trace/CONTEXT.md`).
- VJP deferrals (explicit `TransformError`): `sparse_concatenate`,
  `sparse_coo_to_csc` / `sparse_csc_to_coo` — see the Differentiation section.

## Differentiation & batching (rules.py — implemented)

`rules.py` registers at `import etl.sparse` via `register_batching_rule` /
`register_vjp_rule` / `register_jvp_rule` / `register_batched_aux_remap`
(etl.transforms): one shared batching rule for all 16 sparse ops, the
`SparseTensor` batched-output aux remap, VJP rules for all 16 ops (3
explicit deferrals), and explicit JVP rules for the four BILINEAR ops
(`sparse_multiply`, `sparse_multiply_dense`, `sparse_dot_dense`,
`dense_dot_sparse`); all other sparse ops derive their JVP from their VJP
rule via the adjoint (double-vjp) trick.

- **Batching**: the shared rule rebuilds the op over the batched operands
  with the SAME attributes — dense_shape attrs stay per-element; result
  batch dims come from the input types via ir inference. The output TREE's
  dense_shape gains the batch `Dim` at position 0 via the aux remap — never
  prepend the batch to attrs.
- **VJP**: the structure (indices) is never differentiated — index operands
  get `ZeroTangent`, values a FLAT values cotangent. `sparse_add` gathers
  the merged cotangent (union); `sparse_multiply` WEIGHTS it by the other
  operand's value at the matched row (intersection); merge-row lookup is
  O(nnz²) (`_row_lookup`: row-equality broadcast + argmax + full-row mask).
  Dot vjp rules accumulate dense gradients via one-hot row selection +
  dense matmul — the numpy scatter kernel is OVERWRITE, not accumulate.
- **JVP**: the four bilinear rules are two-term product rules built with the
  `etl.sparse` frontend ops; `sparse_multiply`'s is gather-based (the
  intersection merge reorders rows). Zero tangents short-circuit.
- **Cotangent guidance (binding)**: vjp/grad cotangents for sparse inputs
  are FLAT tensors per leaf (ZeroTangent for indices, a values cotangent for
  values) — there is NO sparse cotangent object; callers rebuild the sparse
  value from the returned leaf cotangents. `etl.grad(f)` on a sparse arg
  needs `argnums=(1,)` (the int64 indices leaf is not differentiable).
- **VJP deferrals (explicit `TransformError`)**: `sparse_concatenate`
  (needs dynamic slicing of the merged cotangent per operand);
  `sparse_coo_to_csc` / `sparse_csc_to_coo` (reorder values; need a
  sort-based un-permutation).

## Routing table

| Path | Area |
|---|---|
| `./value.py` | Three-phase value model: `SparseTensor`/`CSRTensor`/`CSCTensor`/`SparseTensorSpec` (+ `from_concrete`), canonical validation, concrete layout helpers, pytree registration |
| `./ops.py` | Frontend ops: creators (eager), polymorphic converters, computation ops, COO auto-conversion, three-option TraceErrors |
| `./rules.py` | Differentiation: 16 batching rules + `SparseTensor` batched-output aux remap, 16 vjp rules (3 explicit deferrals), explicit jvp rules for the 4 bilinear ops (all other sparse ops auto-derive their jvp via the adjoint trick) |
| `./_utils.py` | Internal helpers: `_get_location` (etl-frame skip), `_require_symbolic_sparse`/`_require_symbolic_dense`, `_wrap_dense`, `_raw_reshape`, `_row_lookup` (O(nnz²) row-equality lookup for the merge AD rules) |

## Notes for agents

- **Pytree leaf layout** (binding): `[tensor leaves..., *dense_shape, dtype,
  format]`; static leaves `[*dense_shape, dtype, format]` — never change it;
  trace graph, pipeline, transforms, and persist all rely on it.
- **Polymorphic unflatten**: rebuild dispatch is on `children[0]`'s kind
  (TensorSpec / SymbolicTensor / Tensor-or-ndarray); the format leaf selects
  the variant class. Never route the symbolic path through the variant
  constructors (they canonical-validate; symbolic leaves have no data).
- **Dim in dense_shape**: allowed only at position 0; its extent is
  `indices.shape[0]` (in-range checks, `to_dense`). The vectorize output
  remap (rules.py) prepends the batch Dim.
- **`operand_extents`**: `concatenate` MUST pass
  `operand_extents=tuple(op.dense_shape[axis] for op in operands)` (the
  kernel requires it; result `dense_shape[axis]` = sum) — declared on the
  `sparse_concatenate` opdef in etl/ir, works end-to-end (only the VJP
  defers).
- **Batched stored-zero padding rows**: graph-side `sparse_from_dense` and
  the merge kernels pad per-batch nnz to the common max with lex-max
  stored-zero rows; stored-zero duplicates are legal (values-aware canonical
  validation).
- **`SparseTensorSpec.from_concrete`**: flatten the concrete instance →
  per-leaf TensorSpecs with the nnz dim (dim 0 of indices/values) → `None`,
  indptr kept STATIC, static leaves passed through → `core.unflatten` (the
  polymorphic unflatten builds the validated spec).
- Ops discipline mirrors `etl.dist`: `current_builder()` first (TraceError
  outside a trace), `_get_location()`, operand normalization (concrete →
  three-option TraceError, spec → TypeError), static checks first
  (dense_shape equality → ShapeError; dtype equality → DTypeError), then
  `builder.create(op_name, operands=..., attributes=..., location=...)` with
  attrs matching the ir opdefs exactly, then result wrapping with the known
  result dense_shape.
