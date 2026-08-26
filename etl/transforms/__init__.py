"""etl.transforms — graph-to-graph transformations.

Frontend transforms over traced graphs: `vectorize` (the batching primitive),
`vmap` (transparent function-side sugar over vectorize), and automatic
differentiation (`grad` / `jvp` / `vjp`). Every transform produces an ordinary
`Graph` of ordinary ops — backends need no transform-specific runtime support.
No execution happens here: transforms never import backends/pipeline.

Read `./CONTEXT.md` for the binding design: rule-call signatures, the
vmap⇔vectorize equivalence contract, AD semantics, and v1 scope.
"""

from etl.transforms.batching import batching_rules, register_batching_rule
from etl.transforms.autodiff import (
    jvp_rules,
    vjp_rules,
    register_jvp_rule,
    register_vjp_rule,
)
from etl.transforms.vectorize import vectorize
from etl.transforms.vmap import vmap
from etl.transforms.grad import grad
from etl.transforms.jvp import jvp
from etl.transforms.vjp import vjp
from etl.transforms._wrappers import TransformCallable

# Registers builtin rules for the standard op set (stubs in this phase).
from etl.transforms import rules as _rules  # noqa: E402,F401

__all__ = [
    "vectorize",
    "vmap",
    "grad",
    "jvp",
    "vjp",
    "TransformCallable",
    "batching_rules",
    "register_batching_rule",
    "jvp_rules",
    "register_jvp_rule",
    "vjp_rules",
    "register_vjp_rule",
]
