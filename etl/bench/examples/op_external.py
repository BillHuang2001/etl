"""External-kernel conformance example (category "op", tags "custom" +
"basic") — ``etl.external_call`` + ``etl.register_external_kernel``, the
kernel-agnostic custom-kernel escape hatch. This is the counterpart of
``op_custom`` (which covers ``etl.block``): a user-defined kernel plugs into
a ``defn`` graph as an OPAQUE named call — no etl op needs to know it, and
the kernel itself (a triton kernel in a future repo, plain numpy here) is
written against a tiny numpy-array interface.

Short tutorial — the full binding contract lives in ``etl/CONTEXT.md``
("External kernels") and ``etl/ops/external.py``:

1. DECLARE the call inside a ``defn`` graph with DECLARED output specs::

       @etl.defn
       def model(x, w):
           h = etl.external_call(
               "bench_row_softmax", x,
               result=etl.TensorSpec((4, 8), etl.float32),
           )
           return etl.dot(h, w)

   ``external_call(name, *operands, result=...)`` builds an opaque
   ``external_call`` IR op (effect ``callback``) carrying the STATIC kernel
   name and the result contract (single ``TensorSpec`` or a tuple of them).
   The name specializes the graph exactly like any static parameter —
   registration is NOT required at trace time. Downstream ordinary ops
   consume the kernel output like any other SSA value. v1 has no static
   (Python) parameters — inputs are tensor operands only.

2. REGISTER the kernel process-globally (a run-time concern)::

       etl.register_external_kernel("bench_row_softmax", my_softmax_kernel)

   The callable contract is ``fn(*np_arrays) -> ndarray | core.Tensor |
   tuple/list of them``; the number/dtype/shape of the returned arrays MUST
   match the declared result specs — the numpy backend validates
   count/dtype/shape at dispatch time (``BackendError`` / ``ShapeError``,
   never a silent coercion). Last registration wins (documented overwrite
   semantics — hot reload and adapter re-registration safe). The registry is
   NEVER serialized: graph artifacts carry only the name string; any process
   that runs a saved graph must re-register its kernels first (else a run-
   time ``BackendError`` naming the kernel).

3. RUN on the numpy backend — the interpreter dispatches the kernel by name
   at the op's position in block op order (effect ``callback`` anchors the
   ordering; no async execution)::

       y = etl.evaluate(model, x, w)   # backend defaults to "numpy"

4. RUN on iree (round-2 host-dispatch): the iree adapter declares
   ``Capabilities.external_calls=True``; at ``lower()`` the graph is SPLIT
   at every top-level ``external_call`` into segment programs + a kernel-call
   plan (payload ``stablehlo-segments``, artifact ``iree-vmfb-segments``),
   and at ``run()`` the segments execute strictly in plan order on the HAL
   device with the kernel's operand tensors STAGED to host numpy arrays at
   each boundary, dispatched through the SAME name-keyed registry, and the
   validated results staged back into the following segment. Carried plain
   values that are NOT kernel operands stay device-resident (zero extra host
   round-trips); per-boundary host staging is a v1 mechanism. v1 lower-time
   restrictions (explicit ``BackendError``, never silent): no
   ``external_call`` inside ``cond``/``while_loop``/``scan`` bodies (not
   splittable), STATIC integer result dims only, at least one tensor
   operand, single-function graphs::

       y = etl.evaluate(model, x, w, backend="iree",
                        target_backends=["llvm-cpu"])

5. xla/tvm and a DIRECT stablehlo export REJECT the op explicitly
   (``Capabilities.external_calls=False``; the exporter lists it in
   ``DEFERRED_OPS``) — ``BackendError`` naming the op, never a silent
   fallback. Transforms have NO rules for ``external_call`` in v1:
   ``vmap``/``grad``/``jvp``/``vjp`` raise ``TransformError`` naming the op
   (the random-op pattern). A future rule surface would register per-op
   batching/vjp rules — exactly the ``custom_l2norm`` pattern in
   ``op_custom``: a batching rule for this row-softmax kernel would
   normalize the mapped LAST axis of the batched operand (the kernel is
   per-row, so the batch axis passes through unchanged), and a vjp rule
   would express the softmax gradient with ordinary ops; the kernel itself
   stays opaque, and graphs that need derivatives must express them via the
   surrounding ordinary ops.

6. Determinism/purity contract: kernels are assumed PURE (same inputs ->
   same outputs); segment execution is strictly sequential in plan order, so
   the graph's ``callback`` effect ordering is preserved and repeat runs of
   the same graph + inputs are bit-identical (the numpy backend dispatches
   in place; the iree adapter stages sequentially).

The example itself — ``op_external``: a ROW-SOFTMAX kernel over the last
axis of a [4, 8] input (the classic fused kernel a kernel library such as
triton would ship), registered under the unique name ``"bench_row_softmax"``
and composed with a downstream ``etl.dot`` into a [4, 8] output — the
canonical "h = external_call(...); return dot(h, w)" shape from the contract
docs. The kernel is pure numpy — the mechanism is kernel-agnostic, no kernel
code lives in etl. Registration happens at MODULE SCOPE, mirroring
``op_custom``'s module-scope ``etl.block`` declarations: the kernel is
available whenever the example registry module is imported (importing
``etl.bench`` / listing examples is side-effect-safe — the registry only
maps a name to a callable; last-wins overwrite makes repeated imports
harmless).

Measured max-abs error (numpy backend, seed 0, strict defaults): 0.0 — the
kernel and the numpy reference run the IDENTICAL fp32 expression sequence
(``exp(x - max)`` / ``sum`` / ``np.matmul``), and the etl ``dot`` kernel IS
``np.matmul``, so the whole pipeline is bit-exact on the numpy backend.
"""
import numpy as np

import etl
from etl import TensorSpec, defn

from .._torch import require_torch
from .base import Example, _F32, register_all

_ROWS = 4
_DIM = 8

#: The static kernel name (specializes the graph; the run-time registry is
#: keyed by this exact string).
_KERNEL_NAME = "bench_row_softmax"


def _row_softmax(x):
    """Row-softmax over the last axis.

    Shared by the registered kernel AND the numpy reference — the identical
    fp32 expression sequence (max-subtract for stability, exp, normalized
    sum), so the reference is bit-exact against the dispatched kernel.
    """
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _row_softmax_kernel(x):
    """The registered external kernel: plain numpy in/out. The numpy backend
    validates the returned array against the declared result spec
    (dtype/shape/count) at dispatch time."""
    return _row_softmax(x)


# Module-scope registration — the established op_custom pattern (module-scope
# side effects): the kernel is available whenever the registry is imported.
etl.register_external_kernel(_KERNEL_NAME, _row_softmax_kernel)


@defn
def _ext_softmax_fn(x, w):
    """h = row-softmax(x) via the external kernel; y = h @ w via ordinary
    ops — downstream ops consume the kernel output like any SSA value."""
    h = etl.external_call(
        _KERNEL_NAME, x, result=TensorSpec((_ROWS, _DIM), _F32)
    )
    return etl.dot(h, w)


def _ext_softmax_numpy_ref(inputs):
    x, w = inputs
    return np.matmul(_row_softmax(x), w)


def _ext_softmax_torch_ref(inputs, device=None):
    torch = require_torch()
    x, w = (torch.as_tensor(t, device=device) for t in inputs)
    return torch.matmul(torch.softmax(x, dim=-1), w).cpu().numpy()


# ---------------------------------------------------------------------------
# Registry (category "op", tags "custom" + "basic")
# ---------------------------------------------------------------------------

register_all([
    Example(
        name="op_external",
        description=(
            "row-softmax via an external kernel (etl.external_call + "
            "etl.register_external_kernel) composed with a downstream dot"
        ),
        specs=(
            TensorSpec((_ROWS, _DIM), _F32),
            TensorSpec((_DIM, _DIM), _F32),
        ),
        graph=_ext_softmax_fn,
        numpy_ref=_ext_softmax_numpy_ref,
        torch_ref=_ext_softmax_torch_ref,
        category="op",
        tags=("custom", "basic"),
    ),
])
