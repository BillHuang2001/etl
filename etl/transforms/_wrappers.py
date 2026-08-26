"""Internal `TransformCallable` — the spec→Graph convention for fn inputs.

When a public transform (`vmap` / `grad` / `jvp` / `vjp`) is given a plain
function or `Defn` instead of a `Graph`, input shapes are unknown at transform
time and transforms never execute tensors. The transform therefore returns a
`TransformCallable`: calling it with a structure of `TensorSpec`s (and static
values) builds and returns the transformed `Graph`. This is transparent sugar
— each public module's docstring states the exact expansion. See
`./CONTEXT.md` ("Spec-callable convention").
"""

from __future__ import annotations

from typing import Callable, Optional

from etl.core import Tensor, TraceError, flatten
from etl.trace import Graph


class TransformCallable:
    """Returned by transforms applied to a callable/`Defn`.

    `tf(*args)` where `args` is a structure of `TensorSpec`s (+ static values)
    matching the wrapped function's inputs returns the transformed `Graph`
    (a fresh graph per call; the wrapped function is traced exactly once per
    call). Passing concrete `Tensor`s raises `TraceError` — transforms never
    execute; build the returned graph explicitly (`etl.build`, `etl.run`).
    """

    def __init__(
        self,
        build: Callable[..., Graph],
        kind: str,
        doc: Optional[str] = None,
    ) -> None:
        self._build = build
        #: One of "vmap" | "grad" | "jvp" | "vjp".
        self.kind = kind
        if doc:
            self.__doc__ = doc

    def __call__(self, *args) -> Graph:
        self._check_no_concrete_tensors(args)
        return self._build(*args)

    def _check_no_concrete_tensors(self, args) -> None:
        flat, _ = flatten(args)
        for arg in flat:
            if isinstance(arg, Tensor):
                raise TraceError(
                    f"calling an etl.{self.kind} result with concrete Tensors is "
                    f"not supported: transforms build Graphs from TensorSpecs and "
                    f"never execute. Pass TensorSpecs to obtain the transformed "
                    f"Graph, then build/run it explicitly (etl.build / etl.run)."
                )

    def __repr__(self) -> str:
        return f"<etl.transforms.TransformCallable kind={self.kind!r}>"
