"""etl.backends.stablehlo — StableHLO MLIR export utility (v1: export-only).

Produces StableHLO MLIR text from verified EvoXIR so external compilers can
take over compilation, e.g. `iree-compile model.mlir -o model.vmfb`. This
module is NOT a `Backend`: it is not registered with the backend registry
and it never lowers, compiles, or runs anything. IREE/XLA/TVM adapters are
documented future integration points — not implemented here.

See `../../CONTEXT.md` (backends contract; "StableHLO exporter — v1 scope"
is binding) and `../../../CONTEXT.md` (root design principles).
"""

from __future__ import annotations

__all__ = ["export"]


def export(graph_or_module) -> str:
    """Export a graph or module as StableHLO MLIR text.

    Args:
        graph_or_module: a `trace.Graph` (duck-typed — its `.module`
            attribute is used; no `etl.trace` import is needed here) or an
            `etl.ir.Module` directly.

    Returns:
        The StableHLO MLIR text (a ``str``) — compiler input for external
        tools, e.g. ``iree-compile model.mlir -o model.vmfb``.

    Behavior (binding — implemented by the Manager in the implementation
    phase; this stub raises NotImplementedError):
        1. Unwrap: if the argument has a ``.module`` attribute
           (`trace.Graph`), use it; accept `etl.ir.Module` directly;
           otherwise raise ``TypeError``.
        2. Verify FIRST: call ``module.verify()`` — verification failures
           surface as `core.VerificationError`; never emit MLIR from
           invalid IR.
        3. Emit via ``writer.Writer``: functions in declaration order, ops
           in block order, symbolic dims as ``?``, constants as
           ``stablehlo.constant`` dense elements (see ``./writer.py``).
        4. Unsupported or deferred op ⇒ `core.BackendError` NAMING THE OP,
           with a message suggesting decomposition or a future adapter.
           Never silently skip an op.
        5. Export-only: this module is NOT registered as a Backend — it is
           intentionally absent from the backend registry and performs no
           lower/compile/load/run.

    Raises:
        NotImplementedError: architecture-phase stub (current behavior).
        TypeError: if ``graph_or_module`` is neither Graph-like nor an
            `etl.ir.Module`.
        core.VerificationError: if the IR fails verification.
        core.BackendError: if any op is unsupported/deferred in v1 (names
            the op).
    """
    raise NotImplementedError(
        "etl.backends.stablehlo.export is an architecture-phase stub; "
        "behavior is implemented by subagent_manager per the docstring above"
    )
