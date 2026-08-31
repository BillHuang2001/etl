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

Naming convention: ``ETL_<BACKEND>_COMPILE_ARGS`` / ``ETL_<BACKEND>_RUNTIME_ARGS``
for flag lists; backend-specific names for non-list options. The env vars are
read LAZILY at call time (no import-time snapshot) and applied ONLY for
option keys absent from the caller's dict. An empty/whitespace value is
treated as unset. A malformed value raises ``core.BackendError`` naming the
variable, the value, and the parse error — never a raw shlex/base64/json
exception, never silent. Option VALUES are never validated here — the
compiler validates them.

The staging-level env vars ``ETL_BACKEND`` / ``ETL_DEVICE`` /
``ETL_TARGET_BACKENDS`` are resolved separately by
``etl.pipeline._resolve_backend_device`` (build/evaluate only, unchanged).
This module never imports ``etl.backends`` (env resolution must not trigger
adapter imports).
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


#: (backend name, stage) -> tuple of (env var, option key, parser) entries
#: (a stage may have several env-var options, e.g. tvm compile).
ENV_OPTION_TABLE: dict[tuple[str, str], tuple[tuple[str, str, object], ...]] = {
    ("iree", "compile"): (
        ("ETL_IREE_COMPILE_ARGS", "iree_compile_args", _parse_flag_list),
    ),
    ("iree", "load"): (
        ("ETL_IREE_RUNTIME_ARGS", "iree_runtime_args", _parse_flag_list),
    ),
    ("xla", "compile"): (
        ("ETL_XLA_COMPILE_OPTIONS", "xla_compile_options", _parse_base64_bytes),
    ),
    ("tvm", "compile"): (
        ("ETL_TVM_TARGET", "tvm_target", lambda var, value: value.strip()),
        ("ETL_TVM_PASS_CONFIGS", "tvm_pass_configs", _parse_json_object),
    ),
}


def apply_env_options(
    backend_name: str | None, options: dict, stage: str
) -> dict:
    """Return a COPY of ``options`` with env-supplied keys added when absent.

    Precedence: an explicit option key always wins over the env var. A
    backend/stage with no table entry (e.g. the numpy backend, the lower and
    run stages in v1) is a no-op. The env vars are read lazily at call time;
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
