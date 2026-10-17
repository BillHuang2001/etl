"""Pipeline env-var option resolution for per-backend options (the
options-override contract — see ``etl/backends/options.py`` for the backend
half).

Design principle: "we only took care of the graph, provide a clean pipeline,
and everything else should be hack-able". Every per-backend compiler flag /
option must be settable at every pipeline stage — explicitly as a kwarg to
``lower``/``compile``/``load``/``run`` (or the ``build``/``evaluate`` sugar),
or, when not passed explicitly, from an environment variable. Precedence
(binding): **explicit kwarg/option > env var > etl default**.

The env-option table (backend, stage) -> (env var, option key, parser):

| Env var                     | Backend | Stage   | Option key            | Format                                        |
|-----------------------------|---------|---------|-----------------------|-----------------------------------------------|
| ``ETL_IREE_COMPILE_ARGS``   | iree    | compile | ``iree_compile_args`` | space-separated flag list (shlex syntax)      |
| ``ETL_IREE_RUNTIME_ARGS``   | iree    | load    | ``iree_runtime_args`` | space-separated flag list (shlex syntax)      |
| ``ETL_XLA_COMPILE_OPTIONS`` | xla     | compile | ``xla_compile_options`` | base64 of a serialized CompileOptionsProto  |
| ``ETL_TVM_TARGET``          | tvm     | compile | ``tvm_target``        | TVM target string                             |
| ``ETL_TVM_PASS_CONFIGS``    | tvm     | compile | ``tvm_pass_configs``  | JSON object (must parse to a dict)            |
| ``ETL_OPT_LEVEL``           | iree/tvm/xla | compile | ``opt_level``      | 'O0'..'O3' (case-insensitive), '0'..'3', or 0..3 (validated — etl-defined value) |
| ``ETL_EIGH_ALGORITHM``      | iree/tvm/xla | lower | ``eigh_algorithm``    | ``jacobi`` or ``jacobi_batched`` (whitespace-stripped; the value is validated by the stablehlo Writer) |
| ``ETL_SPLIT_REDUCTION``     | iree/tvm/xla | lower | ``split_reduction``   | bool: ``1``/``0``/``true``/``false``/``yes``/``no``/``on``/``off`` (case-insensitive; coerced by ``normalize_bool_option``) |

Naming convention: ``ETL_<BACKEND>_COMPILE_ARGS`` / ``ETL_<BACKEND>_RUNTIME_ARGS``
for flag lists; backend-specific names for non-list options. An option that
every compiler adapter accepts IDENTICALLY gets ONE backend-agnostic spelling
(``ETL_OPT_LEVEL``, ``ETL_EIGH_ALGORITHM``, ``ETL_SPLIT_REDUCTION``) rather
than one variable per backend — such a row is repeated under each backend's
table key. The env vars are read LAZILY at call time (no import-time
snapshot) and applied ONLY for option keys absent from the caller's dict. An
empty/whitespace value is treated as unset. A malformed value raises
``core.BackendError`` naming the variable, the value, and the parse error —
never a raw shlex/base64/json exception, never silent.

Option VALUES are not re-validated here beyond TYPE COERCION — the consumer
validates them (the compiler, or the stablehlo Writer for the reserved
exporter options): an unknown/illegal value raises a loud ``core.BackendError``
from the consumer naming the accepted forms. The exceptions are the options
whose value type etl itself defines:

- ``opt_level`` — normalized and validated by
  ``etl.backends.options.normalize_opt_level`` via ``_parse_opt_level``;
- BOOL-valued options (``ETL_SPLIT_REDUCTION``) — coerced to a real ``bool``
  by the shared ``etl.backends.options.normalize_bool_option`` via
  ``_parse_bool_option``, so an env string can never fall into the
  ``bool("0") is True`` trap, and a non-bool spelling is rejected loudly.

Both keep the same precedence and error contract: explicit > env > default,
malformed → ``BackendError`` naming the variable and value.

The ``lower``-stage rows were added with the reserved exporter options
``eigh_algorithm`` / ``split_reduction`` (declared by all three compiler
adapters in ``KNOWN_OPTIONS["lower"]`` and forwarded to the exporter by
``CompilerBackend._exporter_options``); ``etl.pipeline.lower`` calls
``apply_env_options(backend.name, options, "lower")``, so these rows take
effect immediately. ``split_reduction_threshold`` / ``split_reduction_non_exact``
deliberately have NO env rows: they are ``stablehlo.export``-only Writer
knobs, NOT forwarded by ``CompilerBackend._exporter_options`` and NOT in any
adapter's ``KNOWN_OPTIONS``, so an env-supplied value would make
``lower(..., backend="iree")`` fail key validation — pass them directly to
``stablehlo.export(graph, options=...)`` instead.

The staging-level env vars ``ETL_BACKEND`` / ``ETL_DEVICE`` /
``ETL_TARGET_BACKENDS`` are resolved separately by
``etl.pipeline._resolve_backend_device`` (build/evaluate only, unchanged).
This module never imports ``etl.backends`` at module level (env resolution
must not trigger adapter imports); the exceptions are the ``opt_level`` and
bool parsers' lazy function-body imports of ``etl.backends.options`` — by the
time ``apply_env_options`` runs, ``etl`` is fully imported, and that module
pulls only core/ir/numpy, never the optional adapters.
"""
from __future__ import annotations

import base64
import json
import os
import shlex

from etl.core import BackendError

__all__ = ["apply_env_options", "ENV_OPTION_TABLE"]


def _parse_flag_list(var: str, value: str) -> list[str]:
    """Parse a space-separated flag list (shlex syntax — quoting supported)."""
    try:
        parts = shlex.split(value)
    except ValueError as exc:
        raise BackendError(
            f"invalid {var}={value!r}: not a parseable space-separated flag "
            f"list ({exc})"
        ) from None
    if not parts:
        raise BackendError(
            f"invalid {var}={value!r}: expected at least one flag (e.g. "
            "'--iree-llvmcpu-target-cpu=native')"
        ) from None
    return parts


def _parse_base64_bytes(var: str, value: str) -> bytes:
    """Parse a base64-encoded byte payload (e.g. a serialized proto)."""
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as exc:
        raise BackendError(
            f"invalid {var}={value!r}: not valid base64 (a serialized "
            f"CompileOptionsProto): {exc}"
        ) from None


def _parse_json_object(var: str, value: str) -> dict:
    """Parse a JSON object (must decode to a dict)."""
    try:
        decoded = json.loads(value)
    except Exception as exc:
        raise BackendError(
            f"invalid {var}={value!r}: not valid JSON ({exc})"
        ) from None
    if not isinstance(decoded, dict):
        raise BackendError(
            f"invalid {var}={value!r}: expected a JSON object (dict), got "
            f"{type(decoded).__name__}"
        ) from None
    return decoded


def _parse_opt_level(var: str, value: str) -> int:
    """Parse a backend-agnostic optimization level to a normalized int 0..3.

    Accepts 'O0'..'O3' (case-insensitive), '0'..'3', or the ints 0..3 — the
    one etl-defined option, so its value is validated here (unlike raw
    compiler flags). ``normalize_opt_level`` is imported lazily inside the
    body: this module must not import ``etl.backends`` at module level (env
    resolution must not trigger adapter imports), and by the time
    ``apply_env_options`` runs ``etl`` is fully imported.
    """
    from etl.backends.options import normalize_opt_level

    try:
        return normalize_opt_level(value)
    except BackendError as exc:
        raise BackendError(f"invalid {var}={value!r}: {exc}") from None


def _parse_bool_option(var: str, value: str) -> bool:
    """Parse a bool-valued option (``"1"``/``"0"``/``"true"``/``"false"``/...).

    Delegates to the shared ``etl.backends.options.normalize_bool_option`` —
    the ONE bool coercion in etl (so an env-style string never falls into the
    ``bool("0") is True`` trap and there is no second bool parser to keep in
    sync). The import is lazy inside the body for the same reason as
    ``_parse_opt_level``: this module must not import ``etl.backends`` at
    module level. The re-raise names the env VARIABLE and the raw value in
    addition to ``normalize_bool_option``'s accepted-forms message.
    """
    from etl.backends.options import normalize_bool_option

    try:
        return normalize_bool_option(var, value)
    except BackendError as exc:
        raise BackendError(f"invalid {var}={value!r}: {exc}") from None


#: (backend name, stage) -> tuple of (env var, option key, parser) entries
#: (a stage may have several env-var options, e.g. tvm compile).
ENV_OPTION_TABLE: dict[tuple[str, str], tuple[tuple[str, str, object], ...]] = {
    ("iree", "compile"): (
        ("ETL_IREE_COMPILE_ARGS", "iree_compile_args", _parse_flag_list),
        ("ETL_OPT_LEVEL", "opt_level", _parse_opt_level),
    ),
    ("iree", "load"): (
        ("ETL_IREE_RUNTIME_ARGS", "iree_runtime_args", _parse_flag_list),
    ),
    ("xla", "compile"): (
        ("ETL_XLA_COMPILE_OPTIONS", "xla_compile_options", _parse_base64_bytes),
        ("ETL_OPT_LEVEL", "opt_level", _parse_opt_level),
    ),
    ("tvm", "compile"): (
        ("ETL_TVM_TARGET", "tvm_target", lambda var, value: value.strip()),
        ("ETL_TVM_PASS_CONFIGS", "tvm_pass_configs", _parse_json_object),
        ("ETL_OPT_LEVEL", "opt_level", _parse_opt_level),
    ),
    # The reserved stablehlo-exporter lower options (all three compiler
    # adapters declare them in KNOWN_OPTIONS["lower"] and forward them to the
    # exporter, so ONE backend-agnostic spelling per option — the env row is
    # repeated under each adapter's key). ``eigh_algorithm``'s VALUE is
    # validated by the stablehlo Writer (loud BackendError listing the
    # accepted values) — the framework validates KEYS only; only the
    # whitespace is stripped here, like ``ETL_TVM_TARGET``.
    ("iree", "lower"): (
        ("ETL_EIGH_ALGORITHM", "eigh_algorithm", lambda var, value: value.strip()),
        ("ETL_SPLIT_REDUCTION", "split_reduction", _parse_bool_option),
    ),
    ("xla", "lower"): (
        ("ETL_EIGH_ALGORITHM", "eigh_algorithm", lambda var, value: value.strip()),
        ("ETL_SPLIT_REDUCTION", "split_reduction", _parse_bool_option),
    ),
    ("tvm", "lower"): (
        ("ETL_EIGH_ALGORITHM", "eigh_algorithm", lambda var, value: value.strip()),
        ("ETL_SPLIT_REDUCTION", "split_reduction", _parse_bool_option),
    ),
}


def apply_env_options(
    backend_name: str | None, options: dict, stage: str
) -> dict:
    """Return a COPY of ``options`` with env-supplied keys added when absent.

    Precedence: an explicit option key always wins over the env var. A
    backend/stage with no table entry (e.g. the numpy backend, the run stage
    in v1) is a no-op. The env vars are read lazily at call time;
    an empty/whitespace value is unset; a malformed value raises
    ``core.BackendError`` naming the variable and value. The caller's dict
    is never mutated.
    """
    resolved = dict(options)
    if backend_name is None:
        return resolved
    entries = ENV_OPTION_TABLE.get((backend_name, stage))
    if not entries:
        return resolved
    for env_var, option_key, parser in entries:
        if option_key in resolved:
            continue  # explicit wins
        raw = os.environ.get(env_var)
        if not raw or not raw.strip():
            continue  # unset
        resolved[option_key] = parser(env_var, raw)
    return resolved
