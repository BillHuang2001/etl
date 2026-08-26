"""`@etl.defn` — mark a plain Python function as a numerical graph definition.

`Defn` is NOT a JIT decorator: there is no implicit tracing, compiling, or
eager execution. It only attaches a marker so the staging pipeline
(`etl.trace`, `etl.build`, `etl.evaluate`) recognizes the function as a graph
definition, and so accidental direct calls fail loudly instead of silently
running Python semantics.

Calling a `Defn` ALWAYS raises `core.TraceError` (even with `TensorSpec`
inputs) — the message directs the user to `etl.trace(defn, *specs)` (build a
`Graph`) or `etl.evaluate(defn, *args)` (build + run in one explicit step).
This is the spec-compliant "no implicit eager" behavior.

See `./trace.py` for the tracer and `./CONTEXT.md` for the full staging
contract.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from etl import core

__all__ = ["Defn", "defn"]


class Defn:
    """A marked graph definition. Never callable — raises `TraceError`.

    Attributes:
        fn: The wrapped Python function. A `Defn` passed to the constructor
            is unwrapped, so applying `@defn` twice is idempotent.
        options: Optional dict reserved for future defn-compiler-style
            configuration (e.g. preferred backend/compiler hints). v1 ignores
            its contents; unknown keys are not validated here (the pipeline
            validates when it learns to consume them).
    """

    #: Marker checked by `trace()` / `etl.evaluate` via
    #: `hasattr(obj, "__etl_defn__")`.
    __etl_defn__ = True

    def __init__(
        self,
        fn: Callable[..., Any],
        options: Optional[dict] = None,
    ) -> None:
        self.fn = fn.fn if isinstance(fn, Defn) else fn
        self.options = dict(options) if options is not None else {}
        # Marker lives on both the Defn and the underlying function so
        # `trace` accepts either form.
        setattr(self.fn, "__etl_defn__", True)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise core.TraceError(
            "`@etl.defn`-marked functions are graph definitions, not "
            "callables: they cannot be executed eagerly. Trace a graph with "
            "`etl.trace(defn, *specs)` (one `TensorSpec` per tensor input, "
            "static Python values for graph specialization) or build and run "
            "in one explicit step with `etl.evaluate(defn, *args)`."
        )

    def __repr__(self) -> str:
        name = getattr(self.fn, "__qualname__", repr(self.fn))
        return f"<Defn {name}>"


def defn(fn: Optional[Callable[..., Any]] = None, **options: Any) -> Any:
    """Mark `fn` as a graph definition. Usable bare or with options.

    Usage::

        @etl.defn
        def model(x): ...

        @etl.defn(compiler_hint="iree")   # reserved for the future
        def model(x): ...

    Idempotent: applying `defn` to an existing `Defn` returns it unchanged.
    """
    if fn is None:

        def wrap(f: Callable[..., Any]) -> Defn:
            return Defn(f, options)

        return wrap
    if isinstance(fn, Defn):
        return fn
    return Defn(fn, options)
