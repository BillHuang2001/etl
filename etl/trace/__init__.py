"""etl.trace — tracing machinery: `@etl.defn`, `etl.trace`, `Graph`, control flow.

The tracer executes a Python function ONCE under an active `ir.Builder`;
tensor ops (in `etl.ops`) discover where to build via `current_builder()`.
Static Python values keep Python semantics and specialize the graph; runtime
tensor control flow is explicit (`cond` / `while_loop` / `scan`, traced into
IR regions). See `./CONTEXT.md` for the full contract.

Note on naming: inside the package, `etl.trace` the *function* (re-exported
below) shadows `etl.trace` the *submodule* at attribute level after
`etl/__init__.py` imports it. Module attributes remain reachable via
`import etl.trace as trace_mod` or `from etl.trace import current_builder`
(the import system resolves `etl.trace` as the submodule).
"""

from .builder import builder_stack, current_builder, with_builder
from .control_flow import cond, scan, while_loop
from .defn import Defn, defn
from .graph import Graph, StaticValue
from .trace import trace

__all__ = [
    "Defn",
    "Graph",
    "StaticValue",
    "builder_stack",
    "cond",
    "current_builder",
    "defn",
    "scan",
    "trace",
    "while_loop",
    "with_builder",
]
