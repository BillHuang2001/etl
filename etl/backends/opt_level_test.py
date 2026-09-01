"""In-package unit tests for the first-class ``opt_level`` option.

Covers the etl-defined optimization-level option (``"O0"``..``"O3"`` /
``"0"``..``"3"`` / int 0..3 — the FIRST etl-validated common option,
``etl.backends.options.normalize_opt_level``) implemented across the
compiler adapters: the iree derived flag ``--iree-opt-level=N`` (with the
name-collision conflict rule), the xla protobuf wire-format editor
(``xla_util.set_opt_level`` — field 24 of ``executable_build_options``), the
tvm version-adaptive kwarg (loud ``BackendError`` on TVM builds without the
parameter), the ``KNOWN_OPTIONS`` union declarations, the ``ETL_OPT_LEVEL``
env half of ``etl.pipeline_options`` (coordinated feature — skipped until
the sibling wiring lands), and end-to-end pipeline smokes.

Why this file lives in-package: ``pyproject.toml`` sets
``testpaths = ["tests"]``, so files under ``etl/`` are NOT collected by the
default test run. Run it explicitly:

    python3 -m pytest etl/backends/opt_level_test.py -q

No edits to other files; the ``tests/`` suite is untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

import etl
from etl.core import BackendError
from etl.backends.options import OPT_LEVEL, normalize_opt_level, validate_options
from etl.backends.adapters import iree as iree_adapt
from etl.backends.adapters import tvm as tvm_adapt
from etl.backends.adapters import xla as xla_adapt
from etl.backends.adapters import xla_util
from etl.pipeline_options import ENV_OPTION_TABLE, apply_env_options

# ---------------------------------------------------------------------------
# Shared tiny graph (used by the end-to-end smokes): y = x*x + 1
# ---------------------------------------------------------------------------


def _tiny_fn(x):
    return etl.add(etl.multiply(x, x), 1.0)


_TINY_SPEC = etl.TensorSpec((2, 2), etl.float32)
_TINY_INPUT = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
_TINY_EXPECTED = _TINY_INPUT * _TINY_INPUT + 1.0


def _run_parity(backend, **build_options):
    """Build the tiny graph on ``backend`` and assert numpy parity."""
    exe = etl.build(_tiny_fn, _TINY_SPEC, backend=backend, **build_options)
    got = etl.run(exe, _TINY_INPUT).numpy()
    np.testing.assert_allclose(got, _TINY_EXPECTED, rtol=0, atol=1e-6)
    return got


# ---------------------------------------------------------------------------
# 1. normalize_opt_level — the shared value validator (no guards)
# ---------------------------------------------------------------------------


def test_opt_level_constant():
    assert OPT_LEVEL == "opt_level"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # "O0".."O3" strings
        ("O0", 0), ("O1", 1), ("O2", 2), ("O3", 3),
        # case-insensitive
        ("o0", 0), ("o3", 3),
        # whitespace-stripped
        (" O2 ", 2), ("\tO1\n", 1),
        # digit strings
        ("0", 0), ("1", 1), ("2", 2), ("3", 3),
        # ints 0..3
        (0, 0), (1, 1), (2, 2), (3, 3),
    ],
)
def test_normalize_opt_level_accepted(value, expected):
    assert normalize_opt_level(value) == expected


@pytest.mark.parametrize(
    "bad",
    ["O4", "4", -1, 3.5, "banana", "O", True, False, None, [], 3.0, "00"],
)
def test_normalize_opt_level_rejected(bad):
    with pytest.raises(BackendError) as excinfo:
        normalize_opt_level(bad)
    msg = str(excinfo.value)
    assert "opt_level" in msg
    assert "accepted forms" in msg


# ---------------------------------------------------------------------------
# 2. iree — the derived --iree-opt-level=N flag + the conflict rule
# ---------------------------------------------------------------------------


def _iree_opt_flags(flags):
    """All flags whose NAME is ``--iree-opt-level`` (name-based counting,
    mirroring the adapter's own override machinery — values never matter)."""
    return [f for f in flags if f.partition("=")[0] == "--iree-opt-level"]


def test_iree_resolve_compile_args_derived_flag():
    pytest.importorskip("iree.compiler")
    out = iree_adapt._resolve_iree_compile_args(
        ("llvm-cpu",), None, opt_level_args=("--iree-opt-level=O3",)
    )
    assert "--iree-opt-level=O3" in out
    assert len(_iree_opt_flags(out)) == 1


def test_iree_resolve_compile_args_conflict_rule():
    pytest.importorskip("iree.compiler")
    # A user flag whose NAME collides with the derived flag replaces it —
    # never both (the same machinery as every other etl default).
    out = iree_adapt._resolve_iree_compile_args(
        ("llvm-cpu",),
        ["--iree-opt-level=O1"],
        opt_level_args=("--iree-opt-level=O3",),
    )
    assert _iree_opt_flags(out) == ["--iree-opt-level=O1"]


def test_iree_resolve_compile_options_derived_flag():
    pytest.importorskip("iree.compiler")
    # The derived flag is the "O" form: iree-compile 3.11 rejects numeric
    # levels ("'3' value not a valid optimization level, use O0/O1/O2/O3" —
    # verified empirically), so the adapter maps the normalized int to
    # --iree-opt-level=O{N}.
    _, extra = iree_adapt.iree_backend._resolve_compile_options(
        {"opt_level": "O3"}
    )
    assert _iree_opt_flags(extra) == ["--iree-opt-level=O3"]


def test_iree_resolve_compile_options_user_wins():
    pytest.importorskip("iree.compiler")
    _, extra = iree_adapt.iree_backend._resolve_compile_options(
        {"opt_level": "O3", "iree_compile_args": ["--iree-opt-level=O0"]}
    )
    assert _iree_opt_flags(extra) == ["--iree-opt-level=O0"]


def test_iree_resolve_compile_options_bad_value():
    pytest.importorskip("iree.compiler")
    with pytest.raises(BackendError) as excinfo:
        iree_adapt.iree_backend._resolve_compile_options({"opt_level": "banana"})
    assert "opt_level" in str(excinfo.value)


def test_iree_resolve_compile_options_unset_default():
    pytest.importorskip("iree.compiler")
    _, extra = iree_adapt.iree_backend._resolve_compile_options({})
    assert _iree_opt_flags(extra) == []  # default path unchanged


# ---------------------------------------------------------------------------
# 3. xla wire editor — set_opt_level (pure python, no plugin needed)
# ---------------------------------------------------------------------------


def test_xla_set_opt_level_inject():
    # 1a 04 2001 2801 (EBO: num_replicas=1, num_partitions=1) + field-24
    # varint c0 01 03 => new length 07.
    assert (
        xla_util.set_opt_level(bytes.fromhex("1a0420012801"), 3)
        == bytes.fromhex("1a0720012801c00103")
    )


def test_xla_set_opt_level_zero_injected_explicitly():
    # Level 0 is PRESENT (c0 01 00) — never silently skipped (XLA reads it
    # as the default; wire-valid).
    assert (
        xla_util.set_opt_level(bytes.fromhex("1a0420012801"), 0)
        == bytes.fromhex("1a0720012801c00100")
    )


def test_xla_set_opt_level_conflict_rule():
    # An explicit optimization level in the user's bytes wins — unchanged.
    assert (
        xla_util.set_opt_level(bytes.fromhex("1a0720012801c00102"), 3)
        == bytes.fromhex("1a0720012801c00102")
    )


def test_xla_set_opt_level_no_ebo_appends_segment():
    # An outer message with NO field-3 segment gets a fresh field-3 segment
    # containing only the level field (protobuf merge semantics).
    assert (
        xla_util.set_opt_level(b"", 2) == bytes.fromhex("1a03c00102")
    )


def _parse_outer_segments(data):
    """Minimal outer-message segment parser (key, wiretype, wt2 payload)."""
    segments = []
    pos = 0
    while pos < len(data):
        tag, pos = xla_util._read_varint(data, pos)
        key, wt = tag >> 3, tag & 0x07
        if wt == 0:
            _, pos = xla_util._read_varint(data, pos)
            segments.append((key, wt, None))
        elif wt == 2:
            length, pos = xla_util._read_varint(data, pos)
            payload = data[pos : pos + length]
            pos += length
            segments.append((key, wt, payload))
        else:
            raise AssertionError(f"unexpected wiretype {wt}")
    return segments


def test_xla_set_opt_level_multibyte_length_roundtrip():
    # Build an outer message whose EBO payload length >= 128, so the rebuilt
    # field-3 length is a multi-byte varint: EBO = num_replicas +
    # num_partitions + a filler length-delimited field (key 200, wt 2) with
    # a 200-byte payload.
    filler_payload = bytes(range(200))
    filler_field = (
        xla_util._encode_varint((200 << 3) | 2)
        + xla_util._encode_varint(len(filler_payload))
        + filler_payload
    )
    ebo_payload = bytes.fromhex("20012801") + filler_field
    assert len(ebo_payload) >= 128
    outer = (
        xla_util._encode_varint((3 << 3) | 2)
        + xla_util._encode_varint(len(ebo_payload))
        + ebo_payload
    )

    out = xla_util.set_opt_level(outer, 3)

    # Still exactly one field-3 wt2 segment; original EBO fields intact and
    # the level appended at the end.
    segments = _parse_outer_segments(out)
    assert [(key, wt) for key, wt, _ in segments] == [(3, 2)]
    payload = segments[0][2]
    assert payload == ebo_payload + b"\xc0\x01\x03"

    # Idempotence: an existing field 24 wins — set_opt_level returns
    # unchanged on a second call.
    assert xla_util.set_opt_level(out, 0) == out


# ---------------------------------------------------------------------------
# 4. KNOWN_OPTIONS union declarations (all three adapters)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["compile", "lower"])
def test_iree_known_options_union(stage):
    pytest.importorskip("iree.compiler")
    # opt_level is valid at compile AND passes the union check at lower
    # (accepted-and-ignored there — build/evaluate forward one options dict).
    validate_options({"opt_level": 3}, iree_adapt.IreeBackend.KNOWN_OPTIONS, "iree", stage)


@pytest.mark.parametrize("stage", ["compile", "lower"])
def test_tvm_known_options_union(stage):
    pytest.importorskip("tvm")
    validate_options({"opt_level": 3}, tvm_adapt.TvmBackend.KNOWN_OPTIONS, "tvm", stage)


@pytest.mark.parametrize("stage", ["compile", "lower"])
def test_xla_known_options_union(stage):
    # The xla adapter module imports fine without a PJRT plugin (ctypes +
    # stdlib only) — no guard needed here.
    validate_options({"opt_level": 3}, xla_adapt.XlaBackend.KNOWN_OPTIONS, "xla", stage)


# ---------------------------------------------------------------------------
# 5. Env half — ETL_OPT_LEVEL in etl.pipeline_options (coordinated feature)
# ---------------------------------------------------------------------------


def _env_has_opt_level(backend):
    entries = ENV_OPTION_TABLE.get((backend, "compile"), ())
    return any(env_var == "ETL_OPT_LEVEL" for env_var, _, _ in entries)


def _require_env_opt_level(backend):
    if not _env_has_opt_level(backend):
        pytest.skip(
            "ETL_OPT_LEVEL not yet wired into etl.pipeline_options "
            "(coordinated feature)"
        )


def test_env_opt_level_applied(monkeypatch):
    _require_env_opt_level("iree")
    monkeypatch.setenv("ETL_OPT_LEVEL", "O2")
    assert apply_env_options("iree", {}, "compile") == {"opt_level": 2}


def test_env_opt_level_explicit_wins(monkeypatch):
    _require_env_opt_level("iree")
    monkeypatch.setenv("ETL_OPT_LEVEL", "O2")
    assert apply_env_options("iree", {"opt_level": "O1"}, "compile") == {
        "opt_level": "O1"
    }


def test_env_opt_level_malformed(monkeypatch):
    _require_env_opt_level("iree")
    monkeypatch.setenv("ETL_OPT_LEVEL", "banana")
    with pytest.raises(BackendError) as excinfo:
        apply_env_options("iree", {}, "compile")
    msg = str(excinfo.value)
    # Accept either substring — the sibling's wrapping choice (raw
    # normalize_opt_level error names 'opt_level'; a wrapped one names the
    # variable ETL_OPT_LEVEL).
    assert "ETL_OPT_LEVEL" in msg or "opt_level" in msg


def test_env_opt_level_numpy_unaffected(monkeypatch):
    # No guard needed: numpy has NO table entry by design (and never will),
    # so the env var is always a no-op for it — the assert below is true
    # with or without the sibling wiring.
    monkeypatch.setenv("ETL_OPT_LEVEL", "O2")
    assert apply_env_options("numpy", {}, "compile") == {}


# ---------------------------------------------------------------------------
# 6. End-to-end smokes
# ---------------------------------------------------------------------------


def test_smoke_numpy_opt_level_accepted_and_ignored():
    # The numpy reference backend documents-ignores unknown options:
    # opt_level="O3" flows through lower/compile without validation.
    exe = etl.build(_tiny_fn, _TINY_SPEC, opt_level="O3")
    got = etl.run(exe, _TINY_INPUT).numpy()
    np.testing.assert_array_equal(got, _TINY_EXPECTED)


def test_smoke_iree_opt_level():
    pytest.importorskip("iree.compiler")
    # O3 and 0 both compile (--iree-opt-level=O3 / =O0) and run with numpy
    # parity — x*x+1 is exact in fp32, so the comparison is tight.
    _run_parity("iree", opt_level="O3")
    _run_parity("iree", opt_level=0)


def test_smoke_tvm_opt_level():
    pytest.importorskip("tvm")
    # Default path unchanged: WITHOUT opt_level the tvm build runs with
    # numpy parity (slow — a real Relax VM build, seconds).
    _run_parity("tvm")
    # The opt_level mechanism on the installed TVM 0.26: relax.vm_build.build
    # has NO opt_level kwarg, and an empirical probe proved
    # tvm.transform.PassContext(opt_level=N) does NOT reach the LLVM codegen
    # (byte-identical host-module source at N=0 vs N=3 — only the target's
    # own opt-level attr changes the output). So parity at a given level is
    # NOT assertable on this TVM build; the documented behavior is a loud
    # BackendError naming 'opt_level' and the installed version — never a
    # silently dropped option.
    with pytest.raises(BackendError) as excinfo:
        etl.build(_tiny_fn, _TINY_SPEC, backend="tvm", opt_level=3)
    assert "opt_level" in str(excinfo.value)
