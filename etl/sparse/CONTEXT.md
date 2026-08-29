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
  derives the canonical spec from a CONCRETE sparse instance (used by the
  pending `etl/pipeline.py` evaluate branch).

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

`unflatten` is polymorphic on the first child's kind: `core.TensorSpec` →
validated `SparseTensorSpec`; `core.SymbolicTensor` → symbolic instance via
`SparseTensor.from_parts` (NO canonical validation — no data to check);
`core.Tensor` / `np.ndarray` → concrete instance (canonical validation
applies). The format leaf selects the variant class.

## Constraints

- **Canonical form enforced in the concrete constructors** (never silent;
  `core.ShapeError`/`core.DTypeError`): COO = int64 `(nnz, ndim)` indices,
  lex-sorted row-major, strictly unique, in-range per column; CSR/CSC =
  rank-2, `indptr[0] == 0`, monotone non-decreasing, `indptr[-1] == nnz`,
  per-row/column indices sorted strictly increasing and in-range. Integer
  index dtypes normalize to int64. `dense_shape` = positive ints (a
  `core.Dim` allowed ONLY at position 0 — batched/vmap outputs; its extent
  is `indices.shape[0]` for in-range checks and `to_dense`).
- **`nnz` is runtime-dynamic** in every graph leaf spec (`(None, ndim)` /
  `(None,)`); indptr specs stay STATIC `(rows+1,)` / `(cols+1,)` i64.
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
  `dense_shape = (batch_dim, *dense_shape)` via the transforms
  batched-aux-remap registry (`transforms.register_batched_aux_remap` —
  `rules.py` registers `SparseTensor`'s remap). vmap of sparse INPUTS
  requires an `in_axes` pytree mapping BOTH tensor leaves to 0 and the static
  leaves to `None` — bare `in_axes=0` fails (the node has two tensor leaves).
- Batched graph-side `sparse_from_dense`: per-batch stored-zero PADDING rows
  to a common nnz (lex-max row, zero values); duplicate-with-zero rows are
  legal (kernels validate values-aware) — documented, no special handling.
- **No sparse constant in v1**: `etl.constant(concrete_sparse)` raises
  `core.TraceError` via the existing guard in `etl/ops/constant.py`
  ("etl.constant expects a concrete core.Tensor, got SparseTensor").

## Known issues / v1 deferrals (explicit errors, never silent)

- **`etl.sparse.concatenate` is BLOCKED at trace time**: the
  `sparse_concatenate` opdef in `etl/ir/op_defs/sparse.py` does NOT declare
  `operand_extents`, but the numpy kernel requires it AND `builder.create`
  hard-rejects unknown attributes → `VerificationError: op
  'sparse_concatenate': unknown attribute(s) ['operand_extents']; declared:
  ['axis', 'dense_shape', 'dtype']`. The frontend passes `operand_extents`
  per spec; a one-line ir opdef amendment (AttrSpec name="operand_extents",
  type=ATTR_INTS, required) must land in etl/ir (parent's scope). Everything
  else works end-to-end.
- **Sparse values cannot flow through `cond` in v1**: `etl.trace`'s cond
  machinery requires every branch-output leaf to be a SymbolicTensor and
  rejects the sparse static leaves ("branches yield tensors only (static
  output leaves are not supported)"). `while_loop` accepts static leaves but
  rejects sparse carries whose op-produced leaf shapes carry `None` for the
  nnz dim vs the traced input's `Dim` ("body_fn output leaf N shape (None,
  2) differs from the loop-carried shape (Dim(...), 2)"). Both are
  trace-level constraints (etl/trace/control_flow.py) — parent's scope.
- sparse @ sparse matmul → `TraceError` (densify one operand with
  `etl.sparse.to_dense`).
- Whole-sparse `etl.bind` is not supported in v1 (bind's per-leaf
  spec validation does not cover the sparse static leaves).
- Compiler backends: the 16 sparse ops are numpy-backend-only in v1;
  stablehlo export / iree / xla / tvm defer with an explicit `BackendError`
  suggesting `etl.sparse.to_dense` — see `etl/backends/CONTEXT.md` Known
  Issues.
- Pending parent-level wiring (out of this node's write scope):
  `etl/__init__.py` must import `etl.sparse`; `etl/pipeline.py` `evaluate`
  must derive specs for concrete sparse leaves via
  `SparseTensorSpec.from_concrete` (currently raises `TypeError` at flat leaf
  index 2: "argument ... is int").
- Differentiation/batching rules (`rules.py`, pending): jvp/vjp rules,
  per-op batching rules, and the batched-aux remap are registered by the
  rules agent — see the `rules.py` section below.

## Routing table

| Path | Area |
|---|---|
| `./value.py` | Three-phase value model: `SparseTensor`/`CSRTensor`/`CSCTensor`/`SparseTensorSpec` (+ `from_concrete`), canonical validation, concrete layout helpers, pytree registration |
| `./ops.py` | Frontend ops: creators (eager), polymorphic converters, computation ops, COO auto-conversion, three-option TraceErrors |
| `./rules.py` | (PENDING — rules agent) vectorize/vmap batching rules, jvp/vjp rules, `register_batched_aux_remap` for `SparseTensor` |
| `./_utils.py` | Internal helpers: `_get_location` (etl-frame skip), `_require_symbolic_sparse`/`_require_symbolic_dense`, `_wrap_dense` |

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
  `operand_extents=tuple(op.dense_shape[axis] for op in operands)` (kernel
  requires it; result `dense_shape[axis]` = sum) — blocked until the ir
  opdef amendment lands (see Known issues).
- **Batched stored-zero padding rows**: graph-side `sparse_from_dense` and
  the merge kernels pad per-batch nnz to the common max with lex-max
  stored-zero rows; stored-zero duplicates are legal (values-aware canonical
  validation).
- **`SparseTensorSpec.from_concrete`**: flatten the concrete instance →
  per-leaf TensorSpecs with the nnz dim (dim 0 of indices/values) → `None`,
  indptr kept STATIC, static leaves passed through → `core.unflatten` (the
  polymorphic unflatten builds the validated spec). The pending
  `etl/pipeline.py` evaluate branch calls it for concrete sparse args.
- Ops discipline mirrors `etl.dist`: `current_builder()` first (TraceError
  outside a trace), `_get_location()`, operand normalization (concrete →
  three-option TraceError, spec → TypeError), static checks first
  (dense_shape equality → ShapeError; dtype equality → DTypeError), then
  `builder.create(op_name, operands=..., attributes=..., location=...)` with
  attrs matching the ir opdefs exactly, then result wrapping with the known
  result dense_shape.
