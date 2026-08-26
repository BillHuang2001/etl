"""Effect kinds for IR operations.

The effect of an op is declared in its ``OpDef`` (see ``op_defs``) and
controls what transforms and backends may do with it. See the directory
CONTEXT.md ("Effect model") for the binding ordering rules.

* ``pure``       — no observable side effects; may be CSE'd/reordered freely,
                   but never moved across an effectful op nor out of its region.
* ``read``       — observes mutable state (e.g. process rank, block internals);
                   may not be reordered across writes, duplicated, or eliminated.
* ``write``      — mutates state; reserved for future stateful ops (none in v1).
* ``collective`` — performs communication between devices; ordered with respect
                   to other collectives and effectful ops by program order.
* ``callback``   — executes a Python callback at runtime; never reordered,
                   duplicated, or eliminated.

Effect is **op-level**, not value-level: there are no public effect tokens in
v1. Ordering is positional — op order inside a ``Block`` IS program order for
effectful ops.
"""

EFFECT_PURE = "pure"
EFFECT_WRITE = "write"
EFFECT_READ = "read"
EFFECT_COLLECTIVE = "collective"
EFFECT_CALLBACK = "callback"

EFFECT_KINDS = frozenset(
    {EFFECT_PURE, EFFECT_WRITE, EFFECT_READ, EFFECT_COLLECTIVE, EFFECT_CALLBACK}
)
