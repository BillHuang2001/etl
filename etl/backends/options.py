"""Backend options contract: the shared unknown-option validator.

etl's design principle — "we only took care of the graph, provide a clean
pipeline, and everything else should be hack-able" — requires that users can
pass ARBITRARY per-backend flags/options at every pipeline stage (lower /
compile / load / run and the build/evaluate sugar) without etl hardcoding or
forcing compiler flags. This module defines the shared validation half of
that contract; the option names and their value types are declared per
backend via the ``KNOWN_OPTIONS`` class attribute (see below).

Contract (binding, see also ../CONTEXT.md "Backend options contract"):

- Every backend declares ``KNOWN_OPTIONS: dict[str, frozenset[str]]`` mapping
  a stage name (``"lower"`` / ``"compile"`` / ``"load"`` / ``"run"``) to the
  frozenset of option names it understands at that stage.
- Every stage method validates its options dict with
  ``validate_options(options, self.KNOWN_OPTIONS, self.name, stage)`` BEFORE
  doing anything else.
- Validation is against the UNION of all stages' sets: a key valid for
  another stage of the same backend is ACCEPTED AND IGNORED at this stage
  (the ``build``/``evaluate`` sugar forwards one options dict to several
  stages — an option meant for ``compile`` must not fail at ``lower``).
- A key valid for NO stage of the backend raises ``core.BackendError``
  listing the known options per stage — loud, never silent (the root error
  strategy forbids silently swallowing unknown options).
- Option VALUES are deliberately NOT validated here beyond the adapters' own
  minimal type checks: arbitrary flag values pass through to the compiler,
  which validates them (compiler diagnostics are surfaced as
  ``core.BackendError``). etl never guesses which flag values are legal.
- Precedence (resolved by the pipeline, ``etl/pipeline_options.py``):
  explicit kwarg/option > environment variable > etl default.
- The numpy backend is the REFERENCE interpreter and deliberately declares
  no validation (documented ignore — cross-backend scripts pass e.g.
  ``target_backends`` with a numpy backend; strict validation is the
  compiler adapters' job).

Import acyclicity: this module imports ``etl.core`` ONLY.
"""
from __future__ import annotations

from etl.core import BackendError

__all__ = ["STAGES", "validate_options"]

#: The canonical pipeline stage names (option scope keys).
STAGES = ("lower", "compile", "load", "run")


def validate_options(
    options: dict | None,
    known_by_stage: dict[str, frozenset[str]],
    backend_name: str,
    stage: str,
) -> None:
    """Raise ``core.BackendError`` for option keys unknown to ``backend_name``.

    ``known_by_stage`` is the backend's ``KNOWN_OPTIONS`` class attribute
    (stage -> frozenset of option names). Validation is against the UNION of
    all stages' sets: keys valid for another stage pass (accepted-and-ignored
    at this stage, so the build/evaluate sugar can forward one options dict
    to several stages); keys valid for NO stage raise ``core.BackendError``
    listing the per-stage known sets. ``None``/empty options pass silently.
    """
    if not options:
        return
    union: frozenset[str] = frozenset().union(*known_by_stage.values())
    unknown = sorted(set(options) - union)
    if not unknown:
        return
    listing = []
    for s in STAGES:
        names = sorted(known_by_stage.get(s, frozenset()))
        listing.append(f"{s}: {', '.join(names) or '(none)'}")
    raise BackendError(
        f"the {backend_name} backend does not recognize the {stage} option "
        f"{', '.join(repr(key) for key in unknown)} — known options by "
        f"stage: {'; '.join(listing)}; options valid for other stages are "
        "accepted and ignored at this stage"
    )
