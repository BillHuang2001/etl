"""etl — EvoX Tensor Library.

A minimal, explicit, compiler-neutral tensor graph runtime.

The public surface is the contract documented in this package's CONTEXT.md
(read it). Everything here is an explicit staging step or a transparent
composition of explicit steps — nothing silently traces, compiles, moves, or
communicates.

Typical explicit workflow::

    @etl.defn
    def f(x, w):
        return etl.dot(x, w)

    graph = etl.trace(f, etl.TensorSpec((B, D), etl.float32), etl.TensorSpec((D, H), etl.float32))
    graph = etl.vmap(graph, ...)          # optional
    lowered = etl.lower(graph)
    artifact = etl.compile(lowered)
    executable = etl.load(artifact, device="cpu")
    y = etl.run(executable, x, w)

Convenience shorthands (documented compositions only)::

    exe = etl.build(f, *specs, device="cpu")   # trace→lower→compile→load
    y = etl.evaluate(f, x, w)                  # derive specs → build → run

Pytree utilities (pure sugar over flatten/unflatten — no new semantics):

    etl.tree_map(fn, *trees)        # map over leaves, structure-validated
    etl.tree_leaves(tree)           # all leaves in pre-order
    etl.tree_structure(tree)        # the TreeSpec of a tree
    etl.tree_flatten(tree)          # (leaves, treespec)
    etl.tree_unflatten(leaves, spec)  # rebuild a tree from leaves + spec
"""
from __future__ import annotations

__version__ = "0.1.0"

# --- Value model (etl.core) ------------------------------------------------
from . import core  # noqa: F401
from .core import (  # noqa: F401
    BackendError,
    DTypeError,
    Device,
    DeviceError,
    Dim,
    DimExpr,
    ETLError,
    PersistenceError,
    ShapeError,
    SymbolicTensor,
    Tensor,
    TensorSpec,
    TraceError,
    TransformError,
    TreeSpec,
    VerificationError,
    bool_,
    complex128,
    complex64,
    devices,
    dim,
    dtype,
    empty,
    flatten,
    float16,
    float32,
    float64,
    from_dlpack,
    from_numpy,
    full,
    int8,
    int16,
    int32,
    int64,
    ones,
    register_pytree_node,
    replicate_tensor,
    split_tensor,
    tensor,
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_structure,
    tree_unflatten,
    uint8,
    uint16,
    uint32,
    uint64,
    unflatten,
    zeros,
)

# --- Tensor operations (etl.ops) -------------------------------------------
from . import ops  # noqa: F401
from .ops import (  # noqa: F401
    abs,
    add,
    argmax,
    argmin,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    broadcast,
    cast,
    concatenate,
    constant,
    conv,
    cos,
    cumsum,
    divide,
    dot,
    equal,
    erf,
    exp,
    gather,
    gelu,
    greater,
    greater_equal,
    less,
    less_equal,
    log,
    log1p,
    logical_and,
    logical_not,
    logical_or,
    max,
    maximum,
    mean,
    min,
    minimum,
    multiply,
    negate,
    not_equal,
    pad,
    power,
    prod,
    reduce_max,
    reduce_mean,
    reduce_min,
    reduce_prod,
    reduce_sum,
    relu,
    remainder,
    reshape,
    runtime_call,
    scatter,
    select,
    sigmoid,
    sign,
    sin,
    slice,
    solve,
    sqrt,
    square,
    stop_gradient,
    subtract,
    sum,
    tan,
    tanh,
    transpose,
    tril,
    triu,
)

# --- Tracing (etl.trace) ---------------------------------------------------
# NOTE: `etl.trace` (the function) intentionally shadows the `etl.trace`
# submodule as an attribute — `import etl.trace` / `from etl.trace import ...`
# still work through the import system. Same applies to `etl.numpy` below.
from . import trace  # noqa: F401  (submodule)
from .trace import Defn, Graph, cond, defn, scan, trace, while_loop  # noqa: F401

# --- Graph transforms (etl.transforms) -------------------------------------
from . import transforms  # noqa: F401
from .transforms import grad, jvp, vectorize, vjp, vmap  # noqa: F401

# --- Backends (etl.backends) -----------------------------------------------
from . import backends  # noqa: F401
from .backends import (  # noqa: F401
    Backend,
    Capabilities,
    CompiledArtifact,
    LoweredProgram,
    numpy_backend,
)

# --- Distributed collectives (etl.dist) ------------------------------------
from . import dist  # noqa: F401

# --- Sparse tensors (etl.sparse) -------------------------------------------
from . import sparse  # noqa: F401

# --- Custom blocks (etl.block) ---------------------------------------------
from . import block  # noqa: F401  (submodule)
from .block import BlockOp, block, get_block  # noqa: F401

# --- NumPy-like namespace (etl.numpy) --------------------------------------
from . import numpy  # noqa: F401
from . import numpy as enp  # noqa: F401  (documented alias)

# --- Persistence / cache (etl.persist) -------------------------------------
from . import persist  # noqa: F401
from .persist import Cache, FileCache  # noqa: F401

# --- Execution pipeline (etl.pipeline) -------------------------------------
from . import pipeline  # noqa: F401
from .pipeline import (  # noqa: F401
    BoundExecutable,
    Executable,
    bind,
    build,
    compile,
    evaluate,
    load,
    lower,
    run,
)
