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

``opt_level`` is the FIRST etl-defined + etl-validated common option: unlike
raw compiler flags (pass-through, validated by the compiler), etl defines
what ``opt_level`` means and translates its value per backend, so its VALUE
is validated here — ``normalize_opt_level`` is the shared entry point, also
consumed by the pipeline env-var resolution for ``ETL_OPT_LEVEL``. The
per-backend translation (mapping the normalized int to each compiler's own
optimization setting) happens in the adapters.

Import acyclicity: this module imports ``etl.core`` ONLY.
"""
from __future__ import annotations

from etl.core import BackendError

__all__ = ["STAGES", "validate_options", "normalize_opt_level", "OPT_LEVEL"]

#: The canonical pipeline stage names (option scope keys).
STAGES = ("lower", "compile", "load", "run")

#: The canonical option key for the optimization level — the family-architecture
#: anchor: future common (etl-defined) options slot in next to it.
OPT_LEVEL = "opt_level"


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


def normalize_opt_level(value) -> int:
    """Normalize an ``opt_level`` value to an int 0..3 (etl-validated).

    ``opt_level`` is the one option etl validates because etl defines and
    translates its value (the per-backend translation happens in the
    adapters) — raw compiler flags remain pass-through. Accepted forms:

    - ``"O0"``..``"O3"`` strings, case-insensitive, whitespace-stripped
      (so ``"o3"`` and ``" O2 "`` work);
    - digit strings ``"0"``..``"3"``;
    - ints 0..3 (but NOT bool — ``True``/``False`` are rejected explicitly,
      since bool is an int subclass).

    ANY other value (``None``, floats, ``"O4"``, ``"4"``, ``-1``, ``3.5``,
    ``"banana"``, ``"O"``, lists, ...) raises ``core.BackendError`` naming
    the option and the accepted forms — never a silent fallback.
    """
    if isinstance(value, str):
        stripped = value.strip().upper()
        if stripped in {"O0", "O1", "O2", "O3"}:
            return int(stripped[1])
        if stripped in {"0", "1", "2", "3"}:
            return int(stripped)
    elif type(value) is int:  # `type is int`, not isinstance: bool is excluded
        if 0 <= value <= 3:
            return value
    raise BackendError(
        f"invalid {OPT_LEVEL!r} value {value!r} — accepted forms: "
        "'O0'..'O3' (case-insensitive), '0'..'3', or an int 0..3"
    )
