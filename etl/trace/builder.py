"""Active-builder context: the hook `etl.ops` uses to know WHERE to build IR.

`etl.ops` op functions never receive a builder argument. Instead they query
the innermost active `ir.Builder` via `current_builder()`. The tracer
(`./trace.py`) and the control-flow region machinery (`./control_flow.py`)
install a builder for the duration of the user function / branch function
call using the `with_builder` context manager.

Implementation is a `contextvars.ContextVar` holding an immutable tuple stack
(thread- and async-safe: each context gets its own stack, and the tuple is
never mutated — push/pop copy it, so a child task can never corrupt a
parent's stack). Fully implemented here (trivial pure code) — this is
infrastructure, not tracing behavior.

Also owns `_return_terminator`: the single canonical way `trace.py` and
`control_flow.py` end a block (the `return` terminator op). It lives here
because it is purely a Builder convention, and keeping it in one place means
the "terminator is appended, never created at the insertion point" rule is
documented once.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Optional, Tuple, Type

from etl import core
from etl import ir

__all__ = ["builder_stack", "current_builder", "with_builder"]

#: Immutable tuple stack of active builders, outermost first. Never mutated
#: in place — `with_builder` copies it on push and restores the saved parent
#: on exit, so nested contexts (and async tasks) never interfere.
_builder_stack: ContextVar[Tuple["ir.Builder", ...]] = ContextVar(
    "etl_trace_builder_stack", default=()
)


def current_builder() -> "ir.Builder":
    """Return the innermost active `ir.Builder`.

    Raises:
        core.TraceError: When no trace (or control-flow region) is active —
            i.e. an op function was called outside `etl.trace` / `etl.cond` /
            `etl.while_loop` / `etl.scan` bodies. The message names all
            three entry points (`@etl.defn`-traced definitions, `etl.trace`,
            `etl.evaluate`).
    """
    stack = _builder_stack.get()
    if not stack:
        raise core.TraceError(
            "No active trace: tensor ops can only be called while tracing "
            "a graph definition (inside an `@etl.defn` function traced via "
            "`etl.trace`, or inside the callables of `etl.cond` / "
            "`etl.while_loop` / `etl.scan`). Call `etl.trace(fn, *specs)` "
            "or `etl.evaluate(fn, *args)` to run a graph definition."
        )
    return stack[-1]


class with_builder:
    """Context manager installing `builder` as the active builder for ops.

    Nestable (LIFO): pushes onto the stack on enter, restores the saved
    parent stack on exit (even on error). The stack is a COPY-on-push
    immutable tuple (see the module docstring — thread/async safety).
    """

    def __init__(self, builder: "ir.Builder") -> None:
        self._builder = builder
        self._parent: Optional[Tuple["ir.Builder", ...]] = None

    def __enter__(self) -> "ir.Builder":
        self._parent = _builder_stack.get()
        _builder_stack.set(self._parent + (self._builder,))
        return self._builder

    def __exit__(self, exc_type: Type, exc_val, exc_tb) -> bool:
        _builder_stack.set(self._parent if self._parent is not None else ())
        return False


def _return_terminator(builder: "ir.Builder", values: Tuple["ir.Value", ...]) -> "ir.Op":
    """Emit the `return` terminator op in `builder`'s current block.

    The single canonical way `trace`, `cond`, `while_loop`, and `scan` end a
    block: delegates to `Builder.set_terminator`, which APPENDS the op as the
    block's last op (a terminator is never emitted via `create` at the
    insertion point). `values` = the yielded results (flat `ir.Value`s;
    empty tuple = a zero-result block, which is legal).
    """
    return builder.set_terminator(builder.current_block, "return", tuple(values))


#: Alias of `with_builder` — the "builder stack" context manager named in the
#: package contract; both spellings are supported.
builder_stack = with_builder
