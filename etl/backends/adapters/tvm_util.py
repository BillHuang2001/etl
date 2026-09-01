"""TVM adapter support: vendored-translator compatibility layer + VM helpers.

This module is the ONLY place in the adapter that touches TVM internals and
the vendored-translator compatibility surface (all MLIR-binding access
routes through the ``_mlir_bindings`` seam). It is split out of ``tvm.py``
for two reasons:

1. **Heavy-import rule (binding, ``../CONTEXT.md``).** ``import tvm`` and
   ``import jaxlib`` are lazy — every import lives inside a function body,
   so ``import etl`` and ``import etl.backends`` never import TVM or
   jaxlib. The ``jax`` package is never imported at all (see
   ``_install_jax_shim``).
2. **Honest accounting of a vendored-code compatibility shim.** The
   StableHLO→Relax translator vendored in TVM 0.26.0
   (``tvm.relax.frontend.stablehlo``) targets the *old* ``mlir`` python
   bindings (``ShapedType.isinstance`` classmethods, ``OpView`` as an
   ``Operation`` subclass) and the *old* TVM FFI conventions, while the
   environment pairs TVM 0.26.0 (the ``tvm_ffi`` API) with jaxlib 0.10.x
   (the *new* mlir bindings). The translator therefore crashes on basic
   programs in this pairing. ``ensure_compat()`` patches the five gaps
   below — each patch restores EXACTLY the semantics the vendored code
   assumed (pure type checks, identity-preserving key normalization, list
   attribute decoding); it never changes computation:

   * ``X.isinstance(v)`` classmethods removed from the new mlir bindings —
     restored as ``isinstance(v, X)``.
   * ``OpView`` is no longer an ``Operation`` subclass — the importer's
     ``_nodes`` dict is key-normalized so OpView keys and Operation lookups
     meet (the underlying ``.operation`` object is the identity).
   * ``relax.op.broadcast_to`` now requires a ``ShapeExpr`` for the shape —
     the vendored handler passes a plain ``list``; the patch wraps it.
   * ``DenseI64ArrayAttr`` (emitted for the modern ``array<i64: ...>``
     attribute syntax) has no vendored decoder — decoded via ``list()``.
   * Dense-elements constant decoding switches from per-element iteration
     (which raises ``TypeError: Unsupported floating-point type`` on
     float16 constants) to ``numpy.asarray``.

   ``ensure_compat()`` also EXTENDS the translator's op map
   (``convert_map``) with handlers for ops the vendored translator lacks
   but which map 1:1 onto Relax ops — including the writer's
   dynamic-broadcast plumbing (``get_dimension_size`` /
   ``dynamic_broadcast_in_dim``: the Relax VM of 0.26 cannot codegen
   ``broadcast_to`` with a symbolic shape, so the dynamic handler recovers
   the runtime shape source from the writer's deterministic
   ``output_dimensions`` chain and emits ``multiply(data,
   full_like(source, 1))`` — validated end-to-end at multiple concrete
   sizes). Every added handler is validated
   end-to-end (translate → ``relax.vm_build.build`` → run → numpy parity);
   the validated set is ``SUPPORTED_STABLEHLO_OPS`` below, which the
   compile-time pre-check enforces — anything outside it raises
   ``core.BackendError`` naming the op BEFORE the translator is invoked
   (the vendored translator fails with bare ``assert``\\s for unknown ops).

Validated translator/build/run/persistence APIs (TVM 0.26.0,
``tvm_ffi`` 0.1.13.post3, jaxlib 0.10.2 — jaxlib ONLY for its bundled MLIR
python bindings; the ``jax`` package is never used, see ``_install_jax_shim``):

- import: ``tvm.relax.frontend.stablehlo.from_stablehlo(mlir_text)``
  (parses the MLIR with ``_mlir_bindings.make_ir_context()`` — the same
  context recipe the vendored translator's ``jax_mlir.make_ir_context``
  shim delegates to)
- build: ``tvm.relax.vm_build.build(mod, target=tvm.target.Target("llvm"))``
  (note: ``tvm.relax.vm.build`` does NOT exist in 0.26 — ``vm`` resolves to
  ``tvm.runtime.vm``; the build function lives in ``relax.vm_build`` /
  re-exported as ``tvm.relax.build``)
- run: ``tvm.runtime.vm.VirtualMachine(ex, tvm.runtime.cpu())["main"](...)``
  with inputs ``tvm.runtime.tensor(np_array, tvm.runtime.cpu())`` (the
  ``tvm.nd`` namespace no longer exists in 0.26) and outputs
  ``tvm.runtime.Tensor.numpy()``
- persistence: ``VMExecutable.export_library(path)`` writes a host .so;
  ``tvm.runtime.load_module(path)`` reloads it WITHOUT recompiling
  (``Module.save_to_file`` does not exist in 0.26)

**jax-package independence.** The TVM 0.26.0 vendored StableHLO translator
executes ``from jax._src.interpreters import mlir as jax_mlir`` (inside
``from_stablehlo``) and uses exactly two things from it:
``jax_mlir.make_ir_context()`` and ``jax_mlir.ir`` (the latter a literal
re-export of ``jaxlib.mlir.ir``). ``_install_jax_shim()`` satisfies that
import with ``sys.modules`` shims — ``make_ir_context`` delegates to
``_mlir_bindings``, ``ir`` is the ``jaxlib.mlir.ir`` re-export — so the
``jax`` package is never imported and the adapter keeps working even with
``sys.modules["jax"]`` blocked (set to ``None``).
"""
from __future__ import annotations

import base64
import os
import sys
import tempfile
import types
from typing import Any

from etl import core

from . import _mlir_bindings

__all__ = [
    "SUPPORTED_STABLEHLO_OPS",
    "REDUCER_OPS",
    "check_available",
    "ensure_compat",
    "tvm_version",
    "parse_stablehlo",
    "precheck_module",
    "translate",
    "build_vm_executable",
    "export_library_base64",
    "load_virtual_machine",
    "invoke",
    "as_tvm_tensor",
]

#: StableHLO op names whose translation into Relax is validated end-to-end
#: (translate -> vm build -> run -> numpy parity). Everything else is
#: rejected at compile time with ``core.BackendError`` naming the op —
#: including ``stablehlo.convolution`` / ``stablehlo.reduce_window``, which
#: the VENDORED translator claims to handle but hardcodes NHWC/HWIO layouts
#: (etl conv is NCHW) — accepting them would silently compute wrong
#: results. Control flow (``stablehlo.if``/``while``), ``gather``,
#: ``scatter``, ``remainder`` and all collective ops have no vendored
#: handler at all and are rejected the same way.
SUPPORTED_STABLEHLO_OPS: frozenset[str] = frozenset(
    {
        # arithmetic / elementwise
        "stablehlo.add",
        "stablehlo.subtract",
        "stablehlo.multiply",
        "stablehlo.divide",
        "stablehlo.power",
        "stablehlo.maximum",
        "stablehlo.minimum",
        "stablehlo.abs",
        "stablehlo.negate",
        "stablehlo.sign",
        "stablehlo.sqrt",
        "stablehlo.rsqrt",
        "stablehlo.exponential",
        "stablehlo.log",
        "stablehlo.log_plus_one",
        "stablehlo.sine",
        "stablehlo.cosine",
        "stablehlo.tanh",
        "stablehlo.logistic",
        # bitwise / logical
        "stablehlo.and",
        "stablehlo.or",
        "stablehlo.xor",
        "stablehlo.not",
        # data movement / layout
        "stablehlo.convert",
        "stablehlo.broadcast_in_dim",
        "stablehlo.reshape",
        "stablehlo.transpose",
        "stablehlo.concatenate",
        "stablehlo.slice",
        "stablehlo.pad",
        # dynamic-shape broadcast plumbing (writer-emitted for broadcasts
        # whose result has dynamic dims; handlers in the compat shim —
        # validated end-to-end at multiple concrete sizes)
        "stablehlo.get_dimension_size",
        "stablehlo.dynamic_broadcast_in_dim",
        # reductions / linear algebra / constants / comparisons
        "stablehlo.reduce",
        "stablehlo.dot_general",
        "stablehlo.constant",
        "stablehlo.compare",
        "stablehlo.select",
        # terminators
        "func.return",
        "stablehlo.return",
    }
)

#: Reducer ops accepted inside ``stablehlo.reduce`` bodies.
REDUCER_OPS: frozenset[str] = frozenset(
    {
        "stablehlo.add",
        "stablehlo.maximum",
        "stablehlo.minimum",
        "stablehlo.multiply",
    }
)

#: Idempotency guard for ``ensure_compat`` (module state, not global state —
#: the shim must be applied once per process; re-applying wraps ``__init__``
#: repeatedly).
_shims_applied = False


def _install_jax_shim() -> None:
    """Inject ``sys.modules`` shims for the ``jax`` package — never import it.

    The TVM 0.26.0 vendored StableHLO translator executes
    ``from jax._src.interpreters import mlir as jax_mlir`` (inside
    ``from_stablehlo``) and uses exactly ``jax_mlir.make_ir_context()`` and
    ``jax_mlir.ir`` (the latter a literal re-export of ``jaxlib.mlir.ir``).
    The shims satisfy that import without the jax package: ``jax`` /
    ``jax._src`` / ``jax._src.interpreters`` are minimal ``types.ModuleType``
    namespace shells, and ``jax._src.interpreters.mlir`` exposes
    ``make_ir_context()`` (delegating to ``_mlir_bindings.make_ir_context``)
    and ``ir`` (``_mlir_bindings.ir`` — the same module object jax
    re-exports).

    Idempotent (a ``_etl_jax_shim`` marker on the ``jax`` shim
    short-circuits) and cheap. CRITICAL ORDERING: it must run BEFORE any
    ``tvm.relax.frontend.stablehlo`` import/use — hence it is called at the
    top of both ``check_available()`` and ``ensure_compat()``.

    Edge cases:
    * ``sys.modules["jax"] = None`` (a jax-blocked process): the entry is
      overwritten — the shim takes over.
    * A real jax already imported in this process: left in place (it
      satisfies the translator's import and works); the shim only occupies
      the namespace when no real jax is present.
    """
    existing = sys.modules.get("jax")
    if isinstance(existing, types.ModuleType):
        if getattr(existing, "_etl_jax_shim", False):
            return  # our shim is already installed
        return  # a real jax is already imported — leave it alone
    # sys.modules["jax"] is absent or a blocker (None) — install the shims.

    jax_mod = types.ModuleType("jax")
    jax_mod._etl_jax_shim = True
    src_mod = types.ModuleType("jax._src")
    interpreters_mod = types.ModuleType("jax._src.interpreters")

    mlir_shim = types.ModuleType("jax._src.interpreters.mlir")

    def _make_ir_context(*args, **kwargs):
        # jax's signature is (platforms=None, thread_pool_size=...) — both
        # tune jax lowering callbacks the vendored translator never uses;
        # the binding-provider factory takes none.
        return _mlir_bindings.make_ir_context()

    mlir_shim.make_ir_context = _make_ir_context
    mlir_shim.ir = _mlir_bindings.ir  # jax_mlir.ir IS jaxlib.mlir.ir

    sys.modules["jax"] = jax_mod
    sys.modules["jax._src"] = src_mod
    sys.modules["jax._src.interpreters"] = interpreters_mod
    interpreters_mod.mlir = mlir_shim
    sys.modules["jax._src.interpreters.mlir"] = mlir_shim


def _op_name(op: Any) -> str:
    """The operation name of a parsed mlir op (OpView or plain Operation)."""
    return getattr(op, "OPERATION_NAME", None) or op.name


def _op_key(obj: Any) -> Any:
    """Identity key for op bookkeeping: the underlying Operation object.

    New mlir bindings: ``OpView.operation`` is the Operation; old bindings:
    OpView IS an Operation. Either way this returns the stable identity.
    """
    op = getattr(obj, "operation", None)
    if isinstance(op, _mlir_bindings.ir.Operation):
        return op
    return obj


def ensure_compat() -> None:
    """Apply the vendor-compat shim + handler extensions (idempotent).

    See the module docstring for what each patch restores. Raises
    ``core.BackendError`` when TVM or the jaxlib mlir bindings are
    unavailable (with the pip-install hint).
    """
    global _shims_applied
    if _shims_applied:
        return
    _install_jax_shim()
    check_available()

    ir = _mlir_bindings.ir  # jaxlib.mlir.ir via the binding-provider seam
    from tvm import relax
    from tvm.relax.frontend.stablehlo.stablehlo_translator import StableHLOImporter

    # -- (1) restore the removed ``X.isinstance(v)`` classmethods ----------
    for cls in (
        ir.ShapedType,
        ir.IntegerAttr,
        ir.FloatAttr,
        ir.DenseIntElementsAttr,
        ir.DenseFPElementsAttr,
    ):
        if not hasattr(cls, "isinstance"):
            cls.isinstance = classmethod(lambda c, other: isinstance(other, c))

    # -- (2) OpView/Operation key normalization for the importer's _nodes --
    class _NormalizingDict(dict):
        def __setitem__(self, k, v):
            super().__setitem__(_op_key(k), v)

        def __getitem__(self, k):
            return super().__getitem__(_op_key(k))

        def __contains__(self, k):
            return super().__contains__(_op_key(k))

    # -- extra op handlers (validated end-to-end; see module docstring) ----
    def _first(data):
        return data[0] if isinstance(data, list) else data

    def _transpose(self, node):
        data = _first(self.retrieve_operands(node))
        perm = self._attr2value(node.attributes["permutation"])
        return self.block_builder.emit(relax.op.permute_dims(data, axes=list(perm)))

    def _compare(self, node):
        operands = self.retrieve_operands(node)
        direction = str(node.attributes["comparison_direction"])
        direction = direction.split("comparison_direction ")[-1].strip(">")
        fn = {
            "EQ": relax.op.equal,
            "NE": relax.op.not_equal,
            "LT": relax.op.less,
            "LE": relax.op.less_equal,
            "GT": relax.op.greater,
            "GE": relax.op.greater_equal,
        }[direction]
        return self.block_builder.emit(fn(operands[0], operands[1]))

    def _select(self, node):
        operands = self.retrieve_operands(node)
        return self.block_builder.emit(
            relax.op.where(operands[0], operands[1], operands[2])
        )

    def _convert(self, node):
        data = _first(self.retrieve_operands(node))
        out_dtype = self._convert_data_type(node.result.type)
        return self.block_builder.emit(relax.op.astype(data, out_dtype))

    def _unary(self, node, fn):
        data = _first(self.retrieve_operands(node))
        return self.block_builder.emit(fn(data))

    def _bitwise(self, node, fn):
        operands = self.retrieve_operands(node)
        return self.block_builder.emit(fn(operands[0], operands[1]))

    def _not(self, node):
        data = _first(self.retrieve_operands(node))
        dtype = self._convert_data_type(node.operands[0].type)
        fn = relax.op.logical_not if dtype == "bool" else relax.op.bitwise_not
        return self.block_builder.emit(fn(data))

    def _log1p(self, node):
        data = _first(self.retrieve_operands(node))
        dtype = self._convert_data_type(node.result.type)
        one = relax.const(1, dtype)
        return self.block_builder.emit(relax.op.log(relax.op.add(data, one)))

    def _concat(self, node):
        operands = self.retrieve_operands(node)
        dim = self._attr2value(node.attributes["dimension"])
        return self.block_builder.emit(relax.op.concat(tuple(operands), axis=dim))

    def _slice(self, node):
        data = _first(self.retrieve_operands(node))
        begin = tuple(self._attr2value(node.attributes["start_indices"]))
        end = tuple(self._attr2value(node.attributes["limit_indices"]))
        axes = list(range(len(begin)))
        return self.block_builder.emit(relax.op.strided_slice(data, axes, begin, end))

    def _pad(self, node):
        operands = self.retrieve_operands(node)
        data, pad_value = operands[0], operands[1]
        low = self._attr2value(node.attributes["edge_padding_low"])
        high = self._attr2value(node.attributes["edge_padding_high"])
        pad_width = [int(v) for pair in zip(low, high) for v in pair]
        if not isinstance(pad_value, relax.Constant):
            raise core.BackendError(
                "stablehlo.pad requires a scalar constant pad value"
            )
        value = float(pad_value.data.numpy())
        return self.block_builder.emit(
            relax.op.nn.pad(data, pad_width, pad_mode="constant", pad_value=value)
        )

    def _reduce(self, node):
        data = self.retrieve_operands(node)
        dimensions = self._attr2value(node.attributes["dimensions"])
        reducer_op = _op_name(node.body.blocks[0].operations[0])
        fn = {
            "stablehlo.add": relax.op.sum,
            "stablehlo.maximum": relax.op.max,
            "stablehlo.minimum": relax.op.min,
            "stablehlo.multiply": relax.op.prod,
        }.get(reducer_op)
        if fn is None:
            raise core.BackendError(
                f"stablehlo.reduce reducer {reducer_op!r} is not supported by "
                "the tvm adapter (supported: add/maximum/minimum/multiply)"
            )
        return self.block_builder.emit(fn(data[0], axis=dimensions))

    def _get_dimension_size(self, node):
        """``stablehlo.get_dimension_size`` -> scalar i32 placeholder.

        The writer emits this op ONLY inside the ``output_dimensions``
        chain of a ``stablehlo.dynamic_broadcast_in_dim``, whose tvm
        handler derives the target shape from the result type and never
        consumes this value — so a type-consistent scalar placeholder
        keeps the importer walk coherent (the value is dead by
        construction).
        """
        self.retrieve_operands(node)
        return self.block_builder.emit(relax.const(0, "int32"))

    def _dynamic_broadcast_source(self, dims_value):
        """Recover the shape-source mlir Value from the writer's
        deterministic ``output_dimensions`` chain.

        The writer emits: ``concatenate(reshape(get_dimension_size(src,
        i)), constant(tensor<1xi32>)...)`` — every dynamic dim of the
        target shape queries the SAME source tensor. Returns the Relax
        value of that source; an unrecognized chain raises
        ``core.BackendError`` (never a silent guess).
        """
        pieces = []

        def walk(value):
            op = value.owner if hasattr(value, "owner") else value
            op = getattr(op, "operation", op)
            name = getattr(op, "OPERATION_NAME", None) or getattr(op, "name", None)
            if name == "stablehlo.concatenate":
                for operand in op.operands:
                    walk(operand)
                return
            if name == "stablehlo.reshape":
                walk(op.operands[0])
                return
            if name == "stablehlo.constant":
                return  # static target dim — no runtime source needed
            if name == "stablehlo.get_dimension_size":
                pieces.append(op.operands[0])
                return
            raise core.BackendError(
                "stablehlo.dynamic_broadcast_in_dim output_dimensions chain "
                f"op {name!r} is not supported by the tvm adapter"
            )

        walk(dims_value)
        sources = {str(piece) for piece in pieces}
        if len(sources) != 1:
            raise core.BackendError(
                "stablehlo.dynamic_broadcast_in_dim with output dimensions "
                "derived from multiple tensors is not supported by the tvm "
                "adapter"
            )
        return self._retrieve_operands(pieces[0])

    def _dynamic_broadcast_in_dim(self, node):
        """``stablehlo.dynamic_broadcast_in_dim`` -> Relax broadcast.

        Static target shapes emit ``relax.op.broadcast_to`` (VM-codegenable).
        Dynamic target shapes cannot use ``broadcast_to`` (the Relax VM of
        0.26 cannot codegen it with a symbolic shape); instead the handler
        recovers the runtime shape source from the writer's
        ``output_dimensions`` chain and emits ``multiply(data,
        full_like(source, 1))`` — elementwise broadcasting computes the
        target shape at run time (validated end-to-end at multiple
        concrete sizes). Bool data is rejected explicitly (no multiply
        trick on i1).
        """
        operands = self.retrieve_operands(node)
        data = operands[0]
        target_shape = self.get_shape(node.result.type)
        if all(isinstance(d, int) for d in target_shape):
            if len(target_shape) == 0:
                return data
            return self.block_builder.emit(
                relax.op.broadcast_to(data, relax.ShapeExpr(target_shape))
            )
        dtype = self._convert_data_type(node.result.type)
        if dtype == "bool":
            raise core.BackendError(
                "stablehlo.dynamic_broadcast_in_dim of bool data is not "
                "supported by the tvm adapter"
            )
        source = _dynamic_broadcast_source(self, node.operands[1])
        ones = self.block_builder.emit(
            relax.op.full_like(source, relax.const(1, dtype), dtype)
        )
        return self.block_builder.emit(relax.op.multiply(data, ones))

    _extra_handlers = {
        "stablehlo.transpose": _transpose,
        "stablehlo.compare": _compare,
        "stablehlo.select": _select,
        "stablehlo.convert": _convert,
        "stablehlo.negate": lambda self, node: _unary(self, node, relax.op.negative),
        "stablehlo.abs": lambda self, node: _unary(self, node, relax.op.abs),
        "stablehlo.sign": lambda self, node: _unary(self, node, relax.op.sign),
        "stablehlo.logistic": lambda self, node: _unary(self, node, relax.op.sigmoid),
        "stablehlo.tanh": lambda self, node: _unary(self, node, relax.op.tanh),
        "stablehlo.log": lambda self, node: _unary(self, node, relax.op.log),
        "stablehlo.log_plus_one": _log1p,
        "stablehlo.power": lambda self, node: _bitwise(self, node, relax.op.power),
        "stablehlo.and": lambda self, node: _bitwise(self, node, relax.op.bitwise_and),
        "stablehlo.or": lambda self, node: _bitwise(self, node, relax.op.bitwise_or),
        "stablehlo.xor": lambda self, node: _bitwise(self, node, relax.op.bitwise_xor),
        "stablehlo.not": _not,
        "stablehlo.concatenate": _concat,
        "stablehlo.slice": _slice,
        "stablehlo.pad": _pad,
        "stablehlo.reduce": _reduce,
        "stablehlo.get_dimension_size": _get_dimension_size,
        "stablehlo.dynamic_broadcast_in_dim": _dynamic_broadcast_in_dim,
    }

    # -- (3) importer __init__: normalize _nodes, extend the convert map ----
    _orig_init = StableHLOImporter.__init__

    def _init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self._nodes = _NormalizingDict(self._nodes)
        self.convert_map.update(
            {k: types.MethodType(v, self) for k, v in _extra_handlers.items()}
        )

    StableHLOImporter.__init__ = _init

    # -- (4) OpView is not an Operation in the new bindings ----------------
    _orig_retrieve = StableHLOImporter._retrieve_operands

    def _retrieve_operands(self, node):
        if isinstance(node, ir.OpView):
            node = node.operation
        return _orig_retrieve(self, node)

    StableHLOImporter._retrieve_operands = _retrieve_operands

    # -- (5) broadcast_to shape must be a ShapeExpr -------------------------
    def _broadcast_in_dim(self, node):
        operands = self.retrieve_operands(node)
        data = operands[0]
        shape = self.get_shape(node.result.type)
        if len(shape) == 0:
            return data
        return self.block_builder.emit(
            relax.op.broadcast_to(data, relax.ShapeExpr(shape))
        )

    StableHLOImporter._broadcast_in_dim = _broadcast_in_dim

    # -- (6) dense-array attrs + np.asarray constant decoding ----------------
    _orig_attr2value = StableHLOImporter._attr2value

    def _attr2value(self, node):
        name = type(node).__name__
        if name.startswith("Dense") and name.endswith("ArrayAttr"):
            return list(node)
        if ir.DenseIntElementsAttr.isinstance(node) or ir.DenseFPElementsAttr.isinstance(
            node
        ):
            import numpy as np

            shape = self.get_shape(node.type)
            dtype = self._convert_data_type(node.type)
            return np.asarray(node, dtype=dtype).reshape(shape).tolist()
        return _orig_attr2value(self, node)

    StableHLOImporter._attr2value = _attr2value

    _shims_applied = True


def check_available() -> None:
    """Probe the TVM compiler dependency + the StableHLO frontend + jaxlib.

    Raises ``core.BackendError`` with the pip-install hint when TVM or the
    jaxlib MLIR python bindings the vendored translator needs are
    unavailable or too old (the APIs probed here are the exact ones the
    adapter calls: ``from_stablehlo``, ``relax.vm_build.build``,
    ``VirtualMachine``, ``tvm.runtime.tensor``, ``tvm.runtime.load_module``).
    The ``jax`` package is NOT probed — the translator's ``jax._src``
    import is satisfied by the shim installed first (``_install_jax_shim``).
    """
    _install_jax_shim()
    hint = (
        "pip install etl[tvm] (apache-tvm>=0.26 with the StableHLO frontend "
        "and jaxlib for its MLIR python bindings)"
    )
    try:
        import tvm  # noqa: F401
        from jaxlib import mlir  # noqa: F401
        from jaxlib.mlir import ir  # noqa: F401
        from tvm import relax  # noqa: F401
        from tvm.relax.frontend.stablehlo import from_stablehlo  # noqa: F401
        from tvm.relax.vm_build import build  # noqa: F401
        from tvm.runtime import load_module, tensor  # noqa: F401
        from tvm.runtime.vm import VirtualMachine  # noqa: F401
    except ImportError as exc:
        raise core.BackendError(
            f"the tvm backend requires the TVM compiler with its StableHLO "
            f"frontend and jaxlib's MLIR python bindings: {exc} — {hint}"
        ) from exc


def tvm_version() -> str:
    """The installed TVM version string (used for self-describing artifacts)."""
    import tvm

    return str(tvm.__version__)


def parse_stablehlo(mlir_text: str) -> Any:
    """Parse StableHLO MLIR text into a jaxlib mlir module.

    Uses the binding-provider seam ``_mlir_bindings`` — the same context
    factory the vendored ``from_stablehlo`` uses (registers all dialects,
    including stablehlo/chlo). Parse failures raise ``core.BackendError``
    carrying the MLIR error message (this is how an invalid etl export —
    e.g. the dynamic-shape scalar-broadcast limitation documented in Known
    Issues — surfaces: honestly, at compile time, never silently).
    """
    ensure_compat()
    try:
        return _mlir_bindings.parse_module(mlir_text)
    except Exception as exc:
        raise core.BackendError(
            f"the TVM backend could not parse the StableHLO export: {exc}"
        ) from exc


def _iter_module_ops(mlir_module: Any):
    """Yield every op in the module (entry body + nested regions)."""
    for function in mlir_module.body.operations:
        yield function
        for region in function.regions:
            for block in region.blocks:
                for op in block.operations:
                    yield op
                    for nested_region in op.regions:
                        for nested_block in nested_region.blocks:
                            yield from nested_block.operations


def precheck_module(mlir_module: Any) -> None:
    """Capability gate over the parsed StableHLO module.

    Raises ``core.BackendError`` naming the first violation — before the
    translator runs (the vendored translator fails with bare asserts):

    * more than one ``func.func`` (the vendored importer translates only
      the first function);
    * a ``func.return`` with more than one operand (the vendored importer
      keeps only the first output — a silent truncation if accepted);
    * any op outside ``SUPPORTED_STABLEHLO_OPS`` (including convolution /
      reduce_window, whose vendored handlers assume NHWC/HWIO);
    * a ``stablehlo.reduce`` reducer outside ``REDUCER_OPS``;
    * ``stablehlo.slice`` with non-unit strides;
    * ``stablehlo.pad`` with non-zero interior padding.
    """
    functions = [op for op in mlir_module.body.operations if _op_name(op) == "func.func"]
    if len(functions) != 1:
        raise core.BackendError(
            f"the TVM stablehlo translator supports single-function modules "
            f"only, got {len(functions)} functions — decompose the program "
            "or use the numpy backend"
        )
    for op in _iter_module_ops(mlir_module):
        name = _op_name(op)
        if name in ("func.func",):
            continue
        if name in ("func.return", "stablehlo.return"):
            if len(op.operands) != 1:
                raise core.BackendError(
                    "the TVM stablehlo translator keeps only the first "
                    "function output — multi-tensor-output programs are "
                    "not supported by the tvm backend (decompose the "
                    "program or use the numpy backend)"
                )
            continue
        if name not in SUPPORTED_STABLEHLO_OPS:
            raise core.BackendError(
                f"stablehlo op {name!r} is not supported by the tvm backend "
                f"(validated op set: {sorted(SUPPORTED_STABLEHLO_OPS)})"
            )
        if name == "stablehlo.reduce":
            reducer = _op_name(op.body.blocks[0].operations[0])
            if reducer not in REDUCER_OPS:
                raise core.BackendError(
                    f"stablehlo.reduce reducer {reducer!r} is not supported "
                    f"by the tvm backend (supported: {sorted(REDUCER_OPS)})"
                )
        elif name == "stablehlo.slice":
            strides = list(op.attributes["strides"])
            if any(s != 1 for s in strides):
                raise core.BackendError(
                    "stablehlo.slice with non-unit strides is not supported "
                    "by the tvm backend"
                )
        elif name == "stablehlo.pad":
            interior = list(op.attributes["interior_padding"])
            if any(i != 0 for i in interior):
                raise core.BackendError(
                    "stablehlo.pad with interior padding is not supported "
                    "by the tvm backend"
                )


def translate(mlir_text: str) -> Any:
    """StableHLO MLIR text -> Relax ``tvm.IRModule`` (shimmed vendored importer).

    Translator errors are re-raised as ``core.BackendError`` carrying the
    original message — honest, never silent.
    """
    ensure_compat()
    from tvm.relax.frontend.stablehlo import from_stablehlo

    try:
        return from_stablehlo(mlir_text)
    except Exception as exc:
        raise core.BackendError(
            f"the TVM stablehlo translator failed: {exc}"
        ) from exc


def build_vm_executable(
    relax_module: Any, target: str = "llvm", pass_configs: dict | None = None
) -> Any:
    """Build a ``tvm.relax.vm_build.VMExecutable`` for the given target.

    ``target``: the TVM target string (default ``"llvm"``; e.g. ``"llvm
    -mcpu=native"``). ``pass_configs``: an optional dict of Relax pass
    configurations forwarded to ``tvm.relax.vm_build.build`` (a non-None
    value on a TVM build that does not accept ``pass_configs`` raises
    ``core.BackendError`` — never silently dropped).
    """
    import tvm
    from tvm import relax

    try:
        build_kwargs = {}
        if pass_configs is not None:
            # pass_configs is only forwarded when explicitly given: the
            # installed TVM 0.26 ``relax.vm_build.build`` does NOT accept the
            # keyword, so a ``None`` value must not be passed.
            build_kwargs["pass_configs"] = pass_configs
        return relax.vm_build.build(
            relax_module, target=tvm.target.Target(target), **build_kwargs
        )
    except TypeError as exc:
        if pass_configs is not None and "pass_configs" in str(exc):
            raise core.BackendError(
                "the installed TVM build does not accept 'pass_configs' — "
                f"drop the 'tvm_pass_configs' compile option (target "
                f"{target!r}): {exc}"
            ) from exc
        raise core.BackendError(
            f"tvm.relax.vm.build failed for target {target!r}: {exc}"
        ) from exc
    except Exception as exc:
        raise core.BackendError(
            f"tvm.relax.vm.build failed for target {target!r}: {exc}"
        ) from exc


def export_library_base64(executable: Any) -> str:
    """Export the built VM executable as a base64-encoded host library.

    ``VMExecutable.export_library(path)`` writes a host .so (the TVM
    0.26.0 persistence API — ``Module.save_to_file`` does not exist);
    ``tvm.runtime.load_module`` reloads the file WITHOUT recompiling
    (validated end-to-end). The bytes are read back immediately and the
    temp file removed — the artifact embeds the library payload.
    """
    fd, path = tempfile.mkstemp(prefix="etl-tvm-", suffix=".so")
    os.close(fd)
    try:
        executable.export_library(path)
        with open(path, "rb") as handle:
            return base64.b64encode(handle.read()).decode("ascii")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def load_virtual_machine(library_base64: str):
    """Rebuild the ``VirtualMachine`` from the exported library payload.

    Decodes to a temp file and reloads via ``tvm.runtime.load_module`` —
    never recompiles (validated: the reloaded module runs identically).
    """
    import tvm
    from tvm.runtime import cpu, load_module
    from tvm.runtime.vm import VirtualMachine

    try:
        payload = base64.b64decode(library_base64.encode("ascii"))
    except Exception as exc:
        raise core.PersistenceError(
            f"corrupt tvm artifact: library payload is not valid base64 ({exc})"
        ) from exc
    fd, path = tempfile.mkstemp(prefix="etl-tvm-load-", suffix=".so")
    try:
        with open(path, "wb") as handle:
            handle.write(payload)
        try:
            module = load_module(path)
        except Exception as exc:
            raise core.PersistenceError(
                f"the tvm artifact library could not be loaded: {exc} — "
                "never silently recompile"
            ) from exc
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return VirtualMachine(module, cpu())


def as_tvm_tensor(array: Any):
    """Wrap a numpy array as a ``tvm.runtime.Tensor`` on the CPU device."""
    from tvm.runtime import cpu, tensor

    return tensor(array, cpu())


def invoke(vm: Any, tvm_inputs: list[Any]) -> list[Any]:
    """Invoke the VM entry function; return the outputs as numpy arrays.

    The vendored translator always names the Relax entry function
    ``"main"`` (hardcoded), so ``vm["main"]`` is the entry regardless of
    the source function's name. Multiple outputs arrive as a tuple
    (though the compile-time pre-check rejects multi-output programs);
    a program with no tensor outputs returns ``None`` -> ``[]``.
    """
    result = vm["main"](*tvm_inputs)
    if result is None:
        return []
    if isinstance(result, (tuple, list)):
        return [r.numpy() for r in result]
    return [result.numpy()]
