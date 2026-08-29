"""etl.sparse — explicit sparse tensors: value model + graph-time frontend ops.

A sparse tensor is an explicit ``(structure, values)`` pair with a known
dense shape, living in THREE phases that share ONE class hierarchy rooted at
:class:`SparseTensor` (``is_sparse`` catches every phase and variant):

1. **Spec phase** — :class:`SparseTensorSpec`: leaves are ``core.TensorSpec``;
   describes a future runtime sparse tensor (trace inputs).
2. **Symbolic phase** — instances whose leaves are ``core.SymbolicTensor``
   (graph values), assembled via ``SparseTensor.from_parts``.
3. **Concrete phase** — instances whose leaves are numpy arrays or
   ``core.Tensor``, built through the variant constructors (canonical-form
   validation, never silent).

The whole hierarchy is ONE registered pytree node. **Leaf layout (binding):**
``[tensor leaves..., *dense_shape, dtype, format]`` — COO: ``[indices,
values, *dense_shape, dtype, "coo"]``; CSR/CSC: ``[indptr, indices, values,
*dense_shape, dtype, "csr"|"csc"]``. ``dense_shape`` contributes one leaf per
dim, then the values dtype, then the format string.

**COO is the computation format (binding):** any CSR/CSC SYMBOLIC input to a
computation op (or to ``to_dense``) is first converted by emitting
``sparse_csr_to_coo`` / ``sparse_csc_to_coo``; the converted COO feeds the
computation. The csr/csc conversions are rank-2 only (``core.ShapeError`` at
trace time for rank != 2).

The creators (``coo`` / ``csr`` / ``csc`` / ``from_dense``) are concrete and
eager (numpy, no trace). The converters (``to_dense`` / ``to_csr`` /
``to_csc`` / ``to_coo``) are polymorphic: concrete instance -> eager method;
symbolic instance -> graph op. The computation ops (``add`` / ``subtract`` /
``multiply`` / ``multiply_dense`` / ``negate`` / ``sum`` (alias
``reduce_sum``) / ``transpose`` / ``reshape`` / ``concatenate`` / ``matmul``)
are graph-time only — concrete operands raise ``TraceError`` (no eager mode).
"""

from etl.sparse.ops import (
    add,
    csc,
    concatenate,
    coo,
    csr,
    from_dense,
    matmul,
    multiply,
    multiply_dense,
    negate,
    reduce_sum,
    reshape,
    subtract,
    sum,
    to_csc,
    to_coo,
    to_csr,
    to_dense,
    transpose,
)
from etl.sparse.value import (
    CSCTensor,
    CSRTensor,
    SparseTensor,
    SparseTensorSpec,
    is_sparse,
)

__all__ = [
    # value model
    "SparseTensor",
    "CSRTensor",
    "CSCTensor",
    "SparseTensorSpec",
    "is_sparse",
    # creators (polymorphic: concrete -> eager value; symbolic -> graph)
    "coo",
    "csr",
    "csc",
    "from_dense",
    # converters (polymorphic: concrete -> eager; symbolic -> graph op)
    "to_dense",
    "to_csr",
    "to_csc",
    "to_coo",
    # computation ops (graph-time)
    "add",
    "subtract",
    "multiply",
    "multiply_dense",
    "negate",
    "sum",
    "reduce_sum",
    "transpose",
    "reshape",
    "concatenate",
    "matmul",
]

# Import-time side effect: registers the 16 sparse batching rules +
# the sparse batched-output aux remap into etl.transforms (see rules.py).
from . import rules  # noqa: F401
