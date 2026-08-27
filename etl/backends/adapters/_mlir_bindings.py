"""_mlir_bindings — the MLIR binding-provider seam (jaxlib → standalone swap).

Contract
--------
The tvm adapter needs to parse StableHLO MLIR text and build MLIR contexts.
jaxlib is required ONLY for its bundled LLVM MLIR python bindings — the
``jaxlib.mlir`` namespace is a verbatim bundle of the upstream LLVM ``mlir``
python bindings that any MLIR tooling uses; it is NOT the jax API. All
MLIR-binding access in this node goes through this module, so a future
standalone ``mlir``/``stablehlo`` PyPI package can replace jaxlib with a
one-module swap here:

- ``make_ir_context()`` — the context factory (core dialects + the
  separately-shipped stablehlo/chlo dialect modules).
- ``parse_module(mlir_text)`` — parse StableHLO MLIR text into an
  ``ir.Module`` (fresh context).
- ``ir`` — lazy re-export of ``jaxlib.mlir.ir`` (PEP 562 ``__getattr__``),
  for isinstance checks and other binding-level access.

Heavy-import rule (binding, ``../CONTEXT.md``): jaxlib is imported ONLY
inside function bodies (or the lazy ``__getattr__`` hook below) — importing
this module never imports jaxlib, and ``import etl`` never imports jaxlib
at all. Top level: stdlib only.
"""

from __future__ import annotations

__all__ = ["make_ir_context", "parse_module"]

_ir = None


def make_ir_context():
    """Build an MLIR context with stablehlo/chlo/func/cf registered.

    The exact working recipe (validated against jaxlib 0.10.2; copied from
    the xla adapter's ``xla_util._make_mlir_context``): core dialects via
    ``_jax_mlir_ext.register_dialects`` + ``load_all_available_dialects``,
    the separately-shipped stablehlo/chlo .so dialects via their own
    ``register_dialect``. ``Context(load_on_create_dialects=[...])`` alone
    does NOT work (the name registry lacks stablehlo). Returns the
    ``jaxlib.mlir.ir.Context``.
    """
    from jaxlib.mlir import ir as jm_ir
    from jaxlib.mlir._mlir_libs import _jax_mlir_ext
    from jaxlib.mlir.dialects import chlo, stablehlo

    registry = jm_ir.DialectRegistry()
    _jax_mlir_ext.register_dialects(registry)
    context = jm_ir.Context()
    context.append_dialect_registry(registry)
    context.load_all_available_dialects()
    stablehlo.register_dialect(context)
    chlo.register_dialect(context)
    return context


def parse_module(mlir_text):
    """Parse StableHLO MLIR text into an ``ir.Module`` (fresh context).

    Parse errors propagate raw — callers wrap them into ``core.BackendError``
    naming the failing step (never a silent retry).
    """
    from jaxlib.mlir import ir as jm_ir

    return jm_ir.Module.parse(mlir_text, context=make_ir_context())


def __getattr__(name):
    """PEP 562 lazy attribute: ``ir`` is ``jaxlib.mlir.ir`` (cached)."""
    global _ir
    if name == "ir":
        if _ir is None:
            from jaxlib.mlir import ir

            _ir = ir
        return _ir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
