#!/usr/bin/env python3
"""``hf_bench_env`` — provision and verify a THROWAWAY ``etl_hf`` vs PyTorch benchmark env.

This is deliberately a *scratch* tool that lives in ``bench_logs/``: it builds an
isolated virtual environment under ``$TMPDIR`` (never the repo ``.venv``), installs
the wheel set a whole-model comparison needs, and then **verifies every leg** with
measured numbers. Its JSON output (``bench_logs/hf_bench_env.json``) is the
environment/version table a benchmark harness reads.

What is provisioned
-------------------
======================================  =====================================
exact command                            why
======================================  =====================================
``uv venv <venv> --python 3.14``          the repo's ``.python-version`` floor
``uv pip install … torch==2.13.0+rocm7.1`` the PyTorch **ROCm** baseline. PyPI's
                                         linux ``torch`` is a CUDA build, so the
                                         ROCm wheel comes from the pytorch
                                         ``rocm7.1`` index — and ``uv``'s
                                         default index strategy only considers
                                         versions published on the FIRST index
                                         that carries the package, so
                                         ``--index-strategy unsafe-best-match``
                                         is required for the ``+rocm7.1`` local
                                         version to be resolvable at all
                                         (without it: "there is no version of
                                         torch==2.13.0+rocm7.1")
``uv pip install … numpy transformers safetensors``
                                         the numpy-only ``etl``/``etl_hf``
                                         runtime + the HF loader / torch oracle
``uv pip install … iree-base-compiler==3.11.0 iree-base-runtime==3.11.0``
                                         the ``iree`` compiler adapter (matched
                                         compiler/runtime pair)
``uv pip install … jax-rocm7-pjrt==0.11.2``
                                         the ROCm **PJRT plugin** the ``xla``
                                         adapter dlopens (pure-python-tagged, so
                                         it installs on cp314); ships
                                         ``jax_plugins/xla_rocm7/xla_rocm_plugin.so``
======================================  =====================================

The exact ``uv`` invocations live in the module constants
(:data:`UV_PIP_TORCH`, :data:`UV_PIP_REST`), are echoed into the JSON
(``install_commands``) and their real execution records into ``install_runs``.

``etl`` / ``etl_hf`` are NOT installed: they are imported from the checkout via
``PYTHONPATH=<repo root>`` (the absolute path, and the ``etl.__file__`` each
child actually resolved, are recorded).

Provisioning is **idempotent**: the installed set is read back with
``uv pip list --format json``; a second run with everything present executes no
install command at all (``--force`` recreates the venv and reinstalls).

The checks (each a REPORTED RESULT — never an exception, never a silent skip)
----------------------------------------------------------------------------
1. ``python`` / ``rocm-smi`` / ``numpy`` / ``torch`` / ``transformers`` /
   ``safetensors`` / ``iree`` / ``pjrt-plugin`` / ``etl`` / ``etl_hf`` — imports
   and versions: the venv interpreter, the GPU name + driver from ``rocm-smi``,
   the ROCm ``torch`` build + ``torch.version.hip``, the IREE
   compiler/runtime pair plus the drivers it can reach, the PJRT plugin file,
   and that ``etl``/``etl_hf`` really came from this checkout.
2. ``torch-cpu`` — fp32 eager forward of ``tiny_gpt2`` through
   ``transformers.AutoModelForCausalLM.from_pretrained(<dir>)`` on a ``(1, 8)``
   id tensor: finite logits, shape + absmax recorded.
3. ``torch-rocm`` — the same model/ids on ``cuda:0`` (``.to("cuda")``,
   ``.eval()``, ``torch.no_grad()``; torch maps ROCm onto its *cuda* API, hence
   ``cuda:0`` + ``torch.version.hip``) — finite logits and the absmax difference
   vs the CPU result: the torch-GPU-baseline smoke test.
4. ``xla-rocm`` + ``xla-rocm-highest-precision`` — the ``xla``/ROCm leg through
   ``etl``: ``etl_hf`` ``tiny_llama`` at ``seq_len=32`` built and run
   device-resident on ``Device("rocm", 0)`` with the default compile options and
   then with ``dot_precision="HIGHEST"``, each compared against the
   numpy-interpreter leg.
5. ``iree-rocm`` — the same measurement on the IREE compiler adapter, device
   resident (``place_params`` + ``place_ids`` + one explicit ``.to(cpu)``).
6. ``xla-plugin-exit`` — the plugin's interpreter-teardown behaviour, observed
   live on a tiny xla/rocm graph.

Why every check runs in a SUBPROCESS
------------------------------------
The ROCm PJRT plugin **corrupts the heap at interpreter exit**: a process that
built a PJRT client aborts with ``malloc_consolidate(): invalid chunk size``
(SIGABRT / exit 134, sometimes SIGSEGV) *after* all work succeeded. That is the
documented, expected behaviour on this host, so:

* the xla-touching children write their result JSON and leave via
  ``os._exit(0)`` — the abort then never happens;
* every child runs in its own process group with a hard timeout
  (``--check-timeout``; the group is SIGKILLed on expiry) so a hung HIP call
  cannot wedge the run, and a crashing child is reported with its exit
  code/signal instead of taking the provisioner down. A SIGABRT/SIGSEGV exit is
  only acceptable when the child's JSON was written and says pass — which is
  exactly what the merge logic checks;
* :func:`child_xla_exit` deliberately exits NORMALLY to *observe* the teardown
  abort (``xla-plugin-exit``), and the parent records exit code + last stderr.

GPU work is strictly sequential — never two GPU workloads at once — and no
process is left behind (children are reaped; their group is killed on timeout).

Usage
-----
::

    python3 bench_logs/hf_bench_env.py                  # provision (idempotent) + verify
    python3 bench_logs/hf_bench_env.py --verify-only    # checks only, no installs
    python3 bench_logs/hf_bench_env.py --force          # recreate the venv, reinstall
    python3 bench_logs/hf_bench_env.py --only xla,iree  # re-run selected child groups

The parent may run under ANY python >= 3.10: every check executes in the
provisioned interpreter (children), so nothing needs to be installed in the
parent's interpreter.
"""

from __future__ import annotations

import argparse
import datetime
import filecmp
import glob as globmod
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
import platform
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# constants — the provisioning contract (these strings ARE the recipe)
# ---------------------------------------------------------------------------
#: Repository root (``bench_logs/`` parent) — what ``PYTHONPATH`` must point at so
#: the children import ``etl`` / ``etl_hf`` from THIS checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Default venv location (throwaway, under ``$TMPDIR``, never the repo).
DEFAULT_VENV = Path(os.environ.get("TMPDIR", "/tmp")) / "hfbench-venv"

#: Default JSON output (committed next to this script).
DEFAULT_OUT = Path(__file__).resolve().parent / "hf_bench_env.json"

#: Python requested for the venv (the repo's ``.python-version`` is 3.14).
VENV_PYTHON = "3.14"

#: The PyTorch ROCm wheel index. PyPI's linux ``torch`` is a CUDA build, so this
#: is the only source of a ROCm wheel.
TORCH_INDEX_URL = "https://download.pytorch.org/whl/rocm7.1"

#: PyPI — needed for torch's pure-python dependencies, which the pytorch index
#: does not host.
PYPI_INDEX_URL = "https://pypi.org/simple"

#: torch pin: the local version tag is what makes the ROCm wheel win against the
#: (newer) CUDA build published on PyPI.
TORCH_SPEC = "torch==2.13.0+rocm7.1"
TORCH_VERSION_EXPECTED = "2.13.0+rocm7.1"

#: The remaining wheels: etl/etl_hf runtime (numpy), the HF loader + torch
#: oracle, the IREE compiler adapter (matched pair) and the ROCm PJRT plugin.
REST_SPECS = (
    "numpy",
    "transformers",
    "safetensors",
    "iree-base-compiler==3.11.0",
    "iree-base-runtime==3.11.0",
    "jax-rocm7-pjrt==0.11.2",
)

#: Packages that must be installed + their pinned version (``None`` = any).
REQUIRED = {
    "torch": TORCH_VERSION_EXPECTED,
    "numpy": None,
    "transformers": None,
    "safetensors": None,
    "iree-base-compiler": "3.11.0",
    "iree-base-runtime": "3.11.0",
    "jax-rocm7-pjrt": "0.11.2",
}

# ---------------------------------------------------------------------------
# THE EXACT INSTALL COMMANDS. ``{python}`` is the venv interpreter, always
# passed explicitly so the commands are reproducible from outside the venv.
# ---------------------------------------------------------------------------
UV_VENV = ("uv", "venv", "{venv}", "--python", VENV_PYTHON)
UV_VENV_CLEAR = ("uv", "venv", "{venv}", "--python", VENV_PYTHON, "--clear")
UV_PIP_LIST = ("uv", "pip", "list", "--format", "json", "--python", "{python}")
UV_PIP_TORCH = (
    "uv",
    "pip",
    "install",
    "--python",
    "{python}",
    "--index-url",
    TORCH_INDEX_URL,
    "--extra-index-url",
    PYPI_INDEX_URL,
    # uv only considers versions from the first index that carries a package, so
    # without this the "+rocm7.1" local version is invisible behind PyPI's torch.
    "--index-strategy",
    "unsafe-best-match",
    TORCH_SPEC,
)
UV_PIP_REST = ("uv", "pip", "install", "--python", "{python}", *REST_SPECS)

#: XLA flags a benchmark run must set: HIP-graph command buffers SIGSEGV inside
#: ``PJRT_Executable_Execute`` with this plugin, so the command buffer is off.
XLA_FLAGS_RECOMMENDED = "--xla_gpu_enable_command_buffer="

#: Staged plugin (provisioned outside the venv) — preferred when present.
STAGED_PLUGIN = Path(os.environ.get("TMPDIR", "/tmp")) / "pjrt" / "xla_rocm_plugin.so"

#: Models used by the etl backend checks and the torch baseline check.
ETL_MODEL = "tiny_llama"
TORCH_MODEL = "tiny_gpt2"
ETL_SEQ_LEN = 32
TORCH_SEQ_LEN = 8

#: Parity budgets (documented; the measured values land in the JSON).
#: ``xla``/ROCm runs fp32 ``dot`` at TF32-class precision by default, so that leg
#: gets a relative-to-absmax 1e-2 budget; with ``dot_precision="HIGHEST"`` it is
#: true fp32 and gets the 1e-5 budget the device-resident parity suites use.
XLA_REL_TOL_TF32 = 1e-2
XLA_REL_TOL_FP32 = 1e-5
IREE_REL_TOL = 1e-5
TORCH_CPU_GPU_REL_TOL = 1e-5

#: Child groups → the checks each one must produce. The parent uses this to
#: report a crashed/timed-out child as FAILED checks (never a gap).
CHILD_CHECKS = {
    "versions": [
        "numpy",
        "torch",
        "transformers",
        "safetensors",
        "iree",
        "pjrt-plugin",
        "etl",
        "etl_hf",
    ],
    "torch": ["torch-cpu", "torch-rocm"],
    "xla": ["xla-rocm", "xla-rocm-highest-precision"],
    "xla-exit": ["xla-plugin-exit"],
    "iree": ["iree-device-kinds", "iree-rocm"],
}
CHILD_ORDER = tuple(CHILD_CHECKS)

#: Child groups that need ``ETL_PJRT_PLUGIN`` + ``XLA_FLAGS`` in their env.
XLA_CHILDREN = ("xla", "xla-exit")


# ---------------------------------------------------------------------------
# small shared helpers (parent + children)
# ---------------------------------------------------------------------------
def _iso_now() -> str:
    """ISO-8601 UTC timestamp (seconds resolution)."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _tail(text: str | None, lines: int = 6) -> list[str]:
    """Last ``lines`` non-empty lines of ``text`` (JSON-safe: a list of str)."""
    if not text:
        return []
    return [line for line in text.strip().splitlines() if line.strip()][-lines:]


def _exc_text(exc: BaseException) -> str:
    """``Type: message`` for an exception — verbatim, single line."""
    message = str(exc).strip().splitlines()
    detail = message[-1] if message else ""
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _mem_total_gb() -> float | None:
    """Host ``MemTotal`` in GiB (``None`` when unreadable)."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024**2, 1)
    except OSError:
        return None
    return None


def _render(template: tuple[str, ...], **values: object) -> list[str]:
    """Substitute ``{python}``/``{venv}`` placeholders; return the argv list."""
    return [part.format(**values) for part in template]


def run_process(argv: list[str], *, timeout: float, env: dict | None = None) -> dict:
    """Run ``argv`` in its OWN process group; never raise on failure.

    A crashing/aborting/timing-out child is normal here (the ROCm PJRT plugin
    aborts at interpreter exit), so every outcome is returned as a record. On
    timeout the whole process group is SIGKILLed — a hung HIP call must not wedge
    the run and must not leave a background process behind.

    Args:
        argv: command + arguments.
        timeout: wall-clock seconds before the group is killed.
        env: environment for the child (``None`` inherits).

    Returns:
        dict: ``command``, ``exit_code`` (``None`` when signal-killed),
        ``signal`` (name), ``timed_out``, ``duration_s``, ``stdout``,
        ``stderr`` (full text), ``stdout_tail``, ``stderr_tail``, ``error``
        (spawn failure only).
    """
    started = time.time()
    record: dict = {
        "command": shlex.join(argv),
        "exit_code": None,
        "signal": None,
        "timed_out": False,
        "duration_s": None,
        "stdout": "",
        "stderr": "",
        "stdout_tail": [],
        "stderr_tail": [],
        "error": None,
    }
    try:
        proc = subprocess.Popen(
            argv,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except OSError as exc:
        record["error"] = _exc_text(exc)
        return record
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        record["timed_out"] = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:  # pragma: no cover - the group may already be gone
            proc.kill()
        try:
            out, err = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL is final
            out, err = "", ""
    record["duration_s"] = round(time.time() - started, 3)
    returncode = proc.returncode
    if returncode is not None and returncode < 0:
        try:
            record["signal"] = signal.Signals(-returncode).name
        except ValueError:  # pragma: no cover - unknown signal number
            record["signal"] = f"signal {-returncode}"
    else:
        record["exit_code"] = returncode
    record["stdout"] = out or ""
    record["stderr"] = err or ""
    record["stdout_tail"] = _tail(out)
    record["stderr_tail"] = _tail(err)
    return record


def public_record(record: dict, *, keep_stdout: bool = False) -> dict:
    """Copy of a run record without the full captured output (JSON size)."""
    trimmed = {key: value for key, value in record.items() if key not in ("stdout", "stderr")}
    if keep_stdout:
        trimmed["stdout_tail"] = record.get("stdout_tail", [])
    return trimmed


def _describe_exit(record: dict) -> str:
    """One-line human description of a child's exit status (for check details)."""
    if record.get("error"):
        return f"could not start: {record['error']}"
    if record.get("timed_out"):
        return f"TIMED OUT after {record['duration_s']}s (process group SIGKILLed)"
    if record.get("signal"):
        return f"killed by {record['signal']} after {record['duration_s']}s"
    return f"exit {record['exit_code']} after {record['duration_s']}s"


def child_env(plugin: str | None, *, extra: dict | None = None) -> dict:
    """Environment shared by every child.

    ``PYTHONPATH`` is pinned to the checkout so ``etl``/``etl_hf`` resolve to THIS
    worktree (each child records the ``etl.__file__``/``etl_hf.__file__`` it got).
    ``PYTHONDONTWRITEBYTECODE`` keeps the worktree clean (``git status``).
    ``ETL_PJRT_PLUGIN``/``XLA_FLAGS`` are REMOVED unless a plugin is given, so the
    non-xla legs provably do not depend on them, and hub access is disabled
    (checkpoints are on disk; a missing one must fail, not download).

    Args:
        plugin: PJRT plugin path for the xla children, else ``None``.
        extra: additional variables merged last.

    Returns:
        dict: the child environment.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("ETL_PJRT_PLUGIN", None)
    env.pop("XLA_FLAGS", None)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    if plugin:
        env["ETL_PJRT_PLUGIN"] = plugin
        env["XLA_FLAGS"] = XLA_FLAGS_RECOMMENDED
    if extra:
        env.update(extra)
    return env


# ---------------------------------------------------------------------------
# parent: provisioning
# ---------------------------------------------------------------------------
def installed_packages(python: Path, *, timeout: float) -> dict:
    """Read the venv's distributions via ``uv pip list --format json``.

    Args:
        python: the venv interpreter.
        timeout: seconds allowed for the listing.

    Returns:
        dict: ``{name: version}`` — empty when the listing failed.
    """
    record = run_process(_render(UV_PIP_LIST, python=str(python)), timeout=timeout)
    if record["exit_code"] != 0:
        return {}
    try:
        listed = json.loads(record["stdout"])
    except json.JSONDecodeError:
        return {}
    return {item["name"]: item["version"] for item in listed}


def _satisfied(installed: dict, name: str, expected: str | None) -> bool:
    """Whether ``name`` is installed at the expected version.

    ``torch`` gets an extra gate: a CUDA build with the same version number is
    NOT acceptable for the GPU baseline, so the local version must say ``+rocm``.
    """
    version = installed.get(name)
    if version is None:
        return False
    if name == "torch" and "+rocm" not in version:
        return False
    if expected is not None and version != expected:
        return False
    return True


def provision(venv: Path, *, force: bool, install_timeout: float) -> dict:
    """Create/refresh the venv and install the pinned wheel set (idempotent).

    Args:
        venv: venv directory.
        force: recreate the venv and reinstall everything.
        install_timeout: seconds allowed per venv/install command.

    Returns:
        dict: ``created``, ``python``, ``python_version``, ``uv_version``,
        ``install_runs`` (executed commands with their real outcome),
        ``install_commands_executed``, ``packages`` (final ``uv pip list``),
        ``packages_before``, ``notes``.
    """
    python = venv / "bin" / "python"
    created = False
    runs: list[dict] = []
    notes: list[str] = []
    executed: list[str] = []

    if force or not python.exists():
        template = UV_VENV_CLEAR if force else UV_VENV
        argv = _render(template, venv=str(venv))
        record = run_process(argv, timeout=install_timeout)
        runs.append({"purpose": "create venv", **public_record(record)})
        created = record["exit_code"] == 0
        executed.append(shlex.join(argv))
        if not created:
            notes.append(
                f"venv creation FAILED: {_describe_exit(record)}; "
                f"stderr: {' | '.join(record['stderr_tail']) or '<empty>'}"
            )
    else:
        notes.append(f"reused the existing venv at {venv} (no uv venv command run)")

    if not python.exists():
        return {
            "created": created,
            "python": str(python),
            "python_version": None,
            "uv_version": None,
            "install_commands_executed": executed,
            "install_runs": runs,
            "packages": {},
            "packages_before": {},
            "notes": notes,
        }

    installed = installed_packages(python, timeout=install_timeout)
    need_torch = force or not _satisfied(installed, "torch", TORCH_VERSION_EXPECTED)
    need_rest = force or any(
        not _satisfied(installed, name, expected) for name, expected in REQUIRED.items()
    )

    plan: list[tuple[str, tuple[str, ...]]] = []
    if need_torch:
        plan.append(("install torch (pytorch rocm7.1 index)", UV_PIP_TORCH))
    if need_rest:
        plan.append(
            ("install numpy/transformers/safetensors/iree compiler+runtime/PJRT", UV_PIP_REST)
        )
    for purpose, template in plan:
        argv = _render(template, python=str(python))
        record = run_process(argv, timeout=install_timeout)
        runs.append({"purpose": purpose, **public_record(record)})
        executed.append(shlex.join(argv))
        if record["exit_code"] != 0:
            notes.append(
                f"{purpose} FAILED: {_describe_exit(record)}; "
                f"stderr: {' | '.join(record['stderr_tail']) or '<empty>'}"
            )
    if not plan:
        notes.append(
            "all required distributions are already present at the pinned versions — "
            "no install command was executed (idempotent re-run)"
        )

    final = installed_packages(python, timeout=install_timeout)
    version_record = run_process([str(python), "-V"], timeout=120)
    python_version = (
        " ".join(version_record["stdout_tail"]).strip()
        if version_record["exit_code"] == 0
        else None
    )
    uv_record = run_process(["uv", "--version"], timeout=60)
    uv_version = (
        " ".join(uv_record["stdout_tail"]).strip() if uv_record["exit_code"] == 0 else None
    )
    return {
        "created": created,
        "python": str(python),
        "python_version": python_version,
        "uv_version": uv_version,
        "install_commands_executed": executed,
        "install_runs": runs,
        "packages": final,
        "packages_before": installed,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# children: each writes its own JSON, then leaves (os._exit(0) where required)
# ---------------------------------------------------------------------------
def _write_child(path: Path, checks: list[dict], meta: dict) -> None:
    """Atomically (re)write a child's result file after every check."""
    payload = {"generated": _iso_now(), "checks": checks, "meta": meta}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


class ChildRecorder:
    """Collects a child's check entries and flushes the file after each one."""

    def __init__(self, path: Path, meta: dict | None = None) -> None:
        """Initialize the recorder.

        Args:
            path: result JSON path (written incrementally).
            meta: child-level metadata merged into every write.
        """
        self.path = path
        self.meta = dict(meta or {})
        self.checks: list[dict] = []

    def meta_set(self, **values: object) -> None:
        """Merge child-level metadata (e.g. the resolved device list) and flush."""
        self.meta.update(values)
        _write_child(self.path, self.checks, self.meta)

    def emit(
        self,
        name: str,
        status: str,
        detail: str,
        *,
        versions: dict | None = None,
        measured: dict | None = None,
    ) -> None:
        """Record one check and flush the file immediately."""
        self.checks.append(
            {
                "name": name,
                "status": status,
                "detail": detail,
                "versions": dict(versions or {}),
                "measured": dict(measured or {}),
            }
        )
        _write_child(self.path, self.checks, self.meta)


def _finish_child(recorder: ChildRecorder, *, hard: bool = True) -> None:
    """Flush the result file and leave the process.

    ``hard=True`` uses ``os._exit`` — mandatory for anything that touched a PJRT
    client, because the plugin corrupts the heap during interpreter teardown.

    Args:
        recorder: the child's recorder.
        hard: whether to bypass interpreter teardown.
    """
    _write_child(recorder.path, recorder.checks, recorder.meta)
    sys.stdout.flush()
    sys.stderr.flush()
    if hard:
        os._exit(0)
    sys.exit(0)


def _ids(seq_len: int, vocab: int):
    """Deterministic ``(1, seq_len)`` int32 token ids inside the vocabulary."""
    import numpy as np

    return (np.arange(seq_len, dtype=np.int32) % max(int(vocab), 1)).reshape(1, seq_len)


def wheel_plugin_path(prefix: Path) -> str | None:
    """The PJRT plugin shipped by the ``jax-rocm7-pjrt`` wheel inside ``prefix``.

    Args:
        prefix: a venv root (``.../hfbench-venv``) or a ``site-packages`` parent.

    Returns:
        str | None: the plugin path, or ``None`` when the wheel is not installed.
    """
    matches = sorted(
        globmod.glob(
            os.path.join(
                str(prefix),
                "lib",
                "python*",
                "site-packages",
                "jax_plugins",
                "xla_rocm7",
                "xla_rocm_plugin.so",
            )
        )
    )
    return matches[0] if matches else None


def child_versions(args: argparse.Namespace, path: Path) -> None:
    """Import/version checks for every package plus the PJRT plugin file."""
    recorder = ChildRecorder(path, meta={"kind": "versions"})
    versions: dict = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
    }

    def emit_import(
        name: str, package: str, module_name: str | None = None
    ) -> object | None:
        """Import ``module_name``, emit a check with its distribution version."""
        try:
            module = __import__(module_name or package)
            version = importlib_metadata.version(package)
            versions[name] = version
            recorder.emit(
                name,
                "pass",
                f"{package} {version} ({getattr(module, '__file__', '<namespace>')})",
                versions={name: version},
            )
            return module
        except BaseException as exc:  # noqa: BLE001 - a broken import is a result
            recorder.emit(name, "fail", f"import failed: {_exc_text(exc)}")
            return None

    emit_import("numpy", "numpy")

    torch_module = emit_import("torch", "torch")
    if torch_module is not None:
        hip = getattr(torch_module.version, "hip", None)
        try:
            device_count = torch_module.cuda.device_count()
            probe_error = None
        except BaseException as exc:  # noqa: BLE001 - the ROCm runtime probe
            device_count, probe_error = None, _exc_text(exc)
        versions["torch_hip"] = hip
        versions["torch_cuda_device_count"] = device_count
        entry = recorder.checks[-1]
        entry["status"] = "pass" if hip else "fail"
        entry["detail"] = (
            f"torch {torch_module.__version__}; torch.version.hip={hip!r}; "
            f"torch.cuda.device_count()={device_count}"
            f"{'' if probe_error is None else f' (probe error: {probe_error})'}"
        )
        if not hip:
            entry["detail"] += (
                " — NOT a ROCm build: the GPU baseline needs the rocm7.1 wheel "
                f"({TORCH_SPEC} from {TORCH_INDEX_URL})"
            )
        entry["versions"] = {
            "torch": torch_module.__version__,
            "torch_hip": hip,
            "torch_cuda_device_count": device_count,
        }
        entry["measured"] = {"cuda_device_count": device_count}
        _write_child(recorder.path, recorder.checks, recorder.meta)

    emit_import("transformers", "transformers")
    emit_import("safetensors", "safetensors")

    # --- iree: compiler + runtime versions and the drivers they can reach ---
    try:
        import iree.compiler  # noqa: F401
        import iree.runtime

        compiler_version = importlib_metadata.version("iree-base-compiler")
        runtime_version = importlib_metadata.version("iree-base-runtime")
        versions["iree-base-compiler"] = compiler_version
        versions["iree-base-runtime"] = runtime_version
        try:
            drivers = list(iree.runtime.query_available_drivers())
            driver_error = None
        except BaseException as exc:  # noqa: BLE001 - a driver probe failure
            drivers, driver_error = None, _exc_text(exc)
        if drivers is not None:
            versions["iree_drivers"] = drivers
        recorder.meta_set(iree_drivers=drivers)
        matched = compiler_version == runtime_version
        recorder.emit(
            "iree",
            "pass" if matched and driver_error is None else "fail",
            f"iree-base-compiler {compiler_version} / iree-base-runtime "
            f"{runtime_version} ({'matched' if matched else 'MISMATCH'}); "
            f"drivers={drivers}"
            + ("" if driver_error is None else f"; driver probe error: {driver_error}"),
            versions={
                "iree-base-compiler": compiler_version,
                "iree-base-runtime": runtime_version,
            },
            measured={"drivers": drivers, "compiler_runtime_matched": matched},
        )
    except BaseException as exc:  # noqa: BLE001
        recorder.emit("iree", "fail", f"import failed: {_exc_text(exc)}")

    # --- the ROCm PJRT plugin the xla adapter dlopens ---
    used = Path(args.plugin) if args.plugin else None
    wheel_plugin = wheel_plugin_path(Path(sys.prefix))
    try:
        pjrt_version = importlib_metadata.version("jax-rocm7-pjrt")
    except importlib_metadata.PackageNotFoundError:
        pjrt_version = None
    versions["jax-rocm7-pjrt"] = pjrt_version
    pieces: list[str] = []
    used_ok = False
    plugin_bytes = None
    for label, candidate in (("used", used), ("wheel", wheel_plugin), ("staged", STAGED_PLUGIN)):
        if not candidate:
            pieces.append(f"{label}=<none>")
            continue
        candidate_path = Path(candidate)
        if candidate_path.is_file():
            size = candidate_path.stat().st_size
            pieces.append(f"{label}={candidate_path} ({size / 1024**2:.1f} MB)")
            if label == "used":
                used_ok, plugin_bytes = True, size
        else:
            pieces.append(f"{label}={candidate_path} (MISSING)")
    identical = None
    if used is not None and used.is_file() and wheel_plugin:
        try:
            identical = filecmp.cmp(str(used), wheel_plugin, shallow=False)
        except OSError:
            identical = None
    try:
        jax_plugins_spec = importlib.util.find_spec("jax_plugins")
        if jax_plugins_spec is None:
            jax_plugins_origin = None
        elif jax_plugins_spec.origin:
            jax_plugins_origin = jax_plugins_spec.origin
        else:  # a PEP-420 namespace package: report its search path
            jax_plugins_origin = (
                "namespace package: "
                + ", ".join(jax_plugins_spec.submodule_search_locations or [])
            )
    except BaseException as exc:  # noqa: BLE001
        jax_plugins_origin = f"probe failed: {_exc_text(exc)}"
    versions["jax_plugins_package"] = jax_plugins_origin
    if plugin_bytes is not None:
        versions["pjrt_plugin_bytes"] = plugin_bytes
    recorder.meta_set(pjrt_plugin=str(used) if used else None, pjrt_plugin_matches_wheel=identical)
    recorder.emit(
        "pjrt-plugin",
        "pass" if used_ok and pjrt_version else "fail",
        "; ".join(pieces)
        + f"; jax-rocm7-pjrt={pjrt_version}; jax_plugins={jax_plugins_origin}"
        + ("" if identical is None else f"; used == wheel copy: {identical}"),
        versions={
            "jax-rocm7-pjrt": pjrt_version,
            "pjrt_plugin_path": str(used) if used else None,
        },
        measured={"plugin_bytes": plugin_bytes, "used_matches_wheel": identical},
    )

    # --- etl / etl_hf must resolve to THIS checkout (PYTHONPATH) ---
    for package in ("etl", "etl_hf"):
        try:
            module = __import__(package)
            module_file = getattr(module, "__file__", None)
            version = getattr(module, "__version__", None)
            inside = None
            if module_file:
                try:
                    inside = REPO_ROOT.resolve() in Path(module_file).resolve().parents
                except OSError:
                    inside = None
            versions[package] = version
            detail = f"{package} {version} from {module_file}"
            if inside is False:
                detail += (
                    f" — NOT imported from the checkout ({REPO_ROOT}); check PYTHONPATH"
                )
            recorder.emit(
                package,
                "pass" if inside else "fail",
                detail,
                versions={package: version},
                measured={"path": module_file, "from_checkout": inside},
            )
        except BaseException as exc:  # noqa: BLE001
            recorder.emit(package, "fail", f"import failed: {_exc_text(exc)}")

    # --- the iree HAL device list etl sees (rocm must be in it) ---
    try:
        import etl

        devices = [{"kind": device.kind, "index": device.index} for device in etl.devices()]
        numbers = [f"{device['kind']}:{device['index']}" for device in devices]
        versions["etl_devices"] = numbers
        recorder.meta_set(etl_devices=devices)
    except BaseException as exc:  # noqa: BLE001 - a device probe failure
        recorder.meta_set(etl_devices=None, etl_devices_error=_exc_text(exc))

    recorder.meta_set(versions=versions)
    _finish_child(recorder, hard=True)


def child_torch(args: argparse.Namespace, path: Path) -> None:
    """torch CPU + ROCm fp32 eager forward of one checkpoint (the GPU baseline)."""
    recorder = ChildRecorder(path, meta={"kind": "torch", "model": TORCH_MODEL})
    model_dir = Path(args.models_dir) / TORCH_MODEL
    ids_recipe = f"(1, {TORCH_SEQ_LEN}) int64 ids = np.arange(1, {TORCH_SEQ_LEN + 1})"

    def fail_both(detail: str, versions: dict | None = None) -> None:
        """Emit both torch checks as failed (a prerequisite failed)."""
        for name in ("torch-cpu", "torch-rocm"):
            recorder.emit(name, "fail", detail, versions=versions or {})

    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM
    except BaseException as exc:  # noqa: BLE001
        fail_both(f"import failed: {_exc_text(exc)}")
        _finish_child(recorder)
        return

    versions = {
        "torch": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "transformers": importlib_metadata.version("transformers"),
    }
    recorder.meta_set(versions=versions, model_dir=str(model_dir), ids_recipe=ids_recipe)
    if not model_dir.is_dir():
        fail_both(f"checkpoint not found at {model_dir} (no download is attempted)", versions)
        _finish_child(recorder)
        return

    ids_np = np.arange(1, TORCH_SEQ_LEN + 1, dtype=np.int64).reshape(1, TORCH_SEQ_LEN)
    cpu_logits = None
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_dir)).eval()
        with torch.no_grad():
            cpu_logits = model(input_ids=torch.from_numpy(ids_np)).logits.float().numpy()
        finite = bool(np.isfinite(cpu_logits).all())
        absmax = float(np.abs(cpu_logits).max())
        recorder.emit(
            "torch-cpu",
            "pass" if finite else "fail",
            f"fp32 eager forward of {TORCH_MODEL} on cpu: logits {cpu_logits.shape}, "
            f"finite={finite}, absmax={absmax:.6g} [{ids_recipe}]",
            versions=versions,
            measured={
                "shape": list(cpu_logits.shape),
                "finite": finite,
                "absmax": absmax,
                "device": "cpu",
            },
        )
    except BaseException as exc:  # noqa: BLE001
        recorder.emit(
            "torch-cpu",
            "fail",
            f"forward failed: {_exc_text(exc)}\n"
            + "\n".join(traceback.format_exc().splitlines()[-6:]),
            versions=versions,
        )

    try:
        if cpu_logits is None:
            raise RuntimeError("torch-cpu failed, so the CPU/GPU comparison has no reference")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "torch.cuda.is_available() is False — no ROCm device is usable by torch "
                f"(torch.version.hip={getattr(torch.version, 'hip', None)!r})"
            )
        gpu_model = model.to("cuda").eval()
        with torch.no_grad():
            gpu_logits = (
                gpu_model(input_ids=torch.from_numpy(ids_np).to("cuda"))
                .logits.float()
                .cpu()
                .numpy()
            )
        finite = bool(np.isfinite(gpu_logits).all())
        absmax = float(np.abs(gpu_logits).max())
        ref_absmax = float(np.abs(cpu_logits).max())
        diff = float(np.abs(gpu_logits - cpu_logits).max())
        rel = diff / ref_absmax if ref_absmax else diff
        ok = finite and rel <= TORCH_CPU_GPU_REL_TOL
        device_name = torch.cuda.get_device_name(0)
        recorder.emit(
            "torch-rocm",
            "pass" if ok else "fail",
            f"fp32 eager forward of {TORCH_MODEL} on cuda:0 ({device_name}): logits "
            f"{gpu_logits.shape}, finite={finite}, absmax={absmax:.6g}; max-abs difference "
            f"vs the torch cpu result = {diff:.3g} (rel-to-absmax {rel:.3g}, budget "
            f"{TORCH_CPU_GPU_REL_TOL:g})",
            versions={**versions, "torch_device_name": device_name},
            measured={
                "shape": list(gpu_logits.shape),
                "finite": finite,
                "absmax": absmax,
                "maxabs_diff_vs_cpu": diff,
                "rel_vs_cpu": rel,
                "device": "cuda:0 (ROCm)",
                "torch_device_name": device_name,
            },
        )
    except BaseException as exc:  # noqa: BLE001
        recorder.emit(
            "torch-rocm",
            "fail",
            f"ROCm forward failed: {_exc_text(exc)}\n"
            + "\n".join(traceback.format_exc().splitlines()[-6:]),
            versions=versions,
        )
    _finish_child(recorder)


def _etl_backend_leg(
    recorder: ChildRecorder,
    *,
    check_name: str,
    backend: str,
    options: dict,
    rel_tol: float,
    note: str,
) -> None:
    """Build + run ``tiny_llama`` device-resident on one compiler backend.

    Numpy-interpreter reference leg, then ``backend`` on ``Device("rocm", 0)``,
    then one explicit ``.to(cpu)`` — the documented device-resident recipe.

    Args:
        recorder: the child's recorder.
        check_name: check name to emit.
        backend: ``"xla"`` or ``"iree"``.
        options: compile options for this variant.
        rel_tol: relative-to-absmax parity budget.
        note: variant description appended to the detail string.
    """
    import numpy as np

    import etl
    import etl_hf

    lib_versions: dict = {
        "numpy": np.__version__,
        "etl": getattr(etl, "__version__", None),
        "etl_hf": getattr(etl_hf, "__version__", None),
    }
    if backend == "iree":
        for package in ("iree-base-compiler", "iree-base-runtime"):
            try:
                lib_versions[package] = importlib_metadata.version(package)
            except importlib_metadata.PackageNotFoundError:
                lib_versions[package] = None
    else:
        try:
            lib_versions["jax-rocm7-pjrt"] = importlib_metadata.version("jax-rocm7-pjrt")
        except importlib_metadata.PackageNotFoundError:
            lib_versions["jax-rocm7-pjrt"] = None
        lib_versions["pjrt_plugin"] = os.environ.get("ETL_PJRT_PLUGIN")
    lib_versions = {key: value for key, value in lib_versions.items() if value is not None}

    cpu = etl.core.Device("cpu", 0)
    device = etl.core.Device("rocm", 0)
    model_dir = Path(recorder.meta["models_dir"]) / ETL_MODEL
    loaded = etl_hf.load_hf(model_dir)
    fwd = etl_hf.forward_fn(loaded.config, seq_len=ETL_SEQ_LEN)
    ids = _ids(ETL_SEQ_LEN, loaded.config.vocab_size)
    reference = etl_hf.run(
        etl_hf.build(fwd, loaded.config, backend="numpy"), loaded.params, ids
    ).numpy()
    ref_absmax = float(np.abs(reference).max())

    started = time.time()
    exe = etl_hf.build(fwd, loaded.config, backend=backend, device=device, **options)
    build_s = time.time() - started
    placed = etl_hf.place_params(loaded.params, device)
    out = etl_hf.run(exe, placed, etl_hf.place_ids(ids, device))
    output_device = f"{out.device.kind}:{out.device.index}"
    numpy_raises = False
    try:
        out.numpy()
    except etl.DeviceError:
        numpy_raises = True
    except BaseException as exc:  # noqa: BLE001 - record, never mask
        numpy_raises = f"unexpected {_exc_text(exc)} (expected etl.DeviceError)"
    logits = out.to(cpu).numpy()
    finite = bool(np.isfinite(logits).all())
    diff = float(np.abs(logits - reference).max())
    rel = diff / ref_absmax if ref_absmax else diff
    repeat = etl_hf.run(exe, placed, etl_hf.place_ids(ids, device)).to(cpu).numpy()
    identical = bool(np.array_equal(logits, repeat))
    ok = (
        finite
        and rel <= rel_tol
        and output_device.startswith("rocm")
        and numpy_raises is True
        and identical
    )
    recorder.emit(
        check_name,
        "pass" if ok else "fail",
        f"{backend}/rocm device-resident etl_hf {ETL_MODEL} (seq_len={ETL_SEQ_LEN}){note}: "
        f"logits {logits.shape}, finite={finite}, absmax={float(np.abs(logits).max()):.6g}; "
        f"max-abs parity vs the numpy interpreter = {diff:.3g} (rel-to-absmax {rel:.3g}, "
        f"budget {rel_tol:g}); output device={output_device}, .numpy() raised "
        f"DeviceError={numpy_raises}; repeat run bit-identical={identical}; "
        f"build={build_s:.2f}s",
        versions=lib_versions,
        measured={
            "backend": backend,
            "model": ETL_MODEL,
            "seq_len": ETL_SEQ_LEN,
            "options": options,
            "build_s": round(build_s, 3),
            "shape": list(logits.shape),
            "finite": finite,
            "absmax": float(np.abs(logits).max()),
            "maxabs_parity_vs_numpy": diff,
            "rel_parity_vs_numpy": rel,
            "rel_tol": rel_tol,
            "output_device": output_device,
            "device_numpy_raises": numpy_raises,
            "repeat_bit_identical": identical,
            "numpy_absmax": ref_absmax,
            "env": {
                "ETL_PJRT_PLUGIN": os.environ.get("ETL_PJRT_PLUGIN"),
                "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
            },
        },
    )


def child_xla(args: argparse.Namespace, path: Path) -> None:
    """The xla/ROCm leg through etl: default options, then true-fp32 dots."""
    recorder = ChildRecorder(
        path,
        meta={
            "kind": "xla",
            "models_dir": args.models_dir,
            "env": {
                "ETL_PJRT_PLUGIN": os.environ.get("ETL_PJRT_PLUGIN"),
                "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
            },
        },
    )
    variants = (
        ("xla-rocm", {}, XLA_REL_TOL_TF32, " [default options — fp32 dots run TF32-class]"),
        (
            "xla-rocm-highest-precision",
            {"dot_precision": "HIGHEST"},
            XLA_REL_TOL_FP32,
            ' [dot_precision="HIGHEST" — true fp32 dots]',
        ),
    )
    for check_name, options, rel_tol, note in variants:
        try:
            _etl_backend_leg(
                recorder,
                check_name=check_name,
                backend="xla",
                options=options,
                rel_tol=rel_tol,
                note=note,
            )
        except BaseException as exc:  # noqa: BLE001 - a failed capability is a result
            recorder.emit(
                check_name,
                "fail",
                f"xla/rocm leg failed: {_exc_text(exc)}\n"
                + "\n".join(traceback.format_exc().splitlines()[-8:]),
                versions={"pjrt_plugin": os.environ.get("ETL_PJRT_PLUGIN")},
                measured={
                    "backend": "xla",
                    "options": options,
                    "env": {
                        "ETL_PJRT_PLUGIN": os.environ.get("ETL_PJRT_PLUGIN"),
                        "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
                    },
                },
            )
    _finish_child(recorder)


def child_xla_exit(args: argparse.Namespace, path: Path) -> None:
    """Observe the plugin's interpreter-exit behaviour on a tiny xla graph.

    This child returns NORMALLY (no ``os._exit``) so the parent can record the
    teardown abort: the documented outcome is SIGABRT (exit 134) with
    ``malloc_consolidate(): invalid chunk size`` *after* all work succeeded.
    """
    recorder = ChildRecorder(path, meta={"kind": "xla-exit"})
    try:
        import numpy as np

        import etl

        @etl.defn
        def add(a, b):
            return etl.add(a, b)

        device = etl.core.Device("rocm", 0)
        started = time.time()
        exe = etl.build(
            add,
            etl.TensorSpec((2, 2), etl.float32),
            etl.TensorSpec((2, 2), etl.float32),
            backend="xla",
            device=device,
        )
        build_s = time.time() - started
        lhs = etl.tensor(np.ones((2, 2), np.float32)).to(device)
        rhs = etl.tensor(np.full((2, 2), 2.0, np.float32)).to(device)
        out = etl.run(exe, lhs, rhs).to(etl.core.Device("cpu", 0)).numpy()
        ok = bool(np.allclose(out, 3.0))
        recorder.emit(
            "xla-plugin-exit",
            "pass" if ok else "fail",
            f"tiny xla/rocm graph built ({build_s:.2f}s) and ran "
            f"(sum={float(out.sum()):.6g}, expected 12); the process now exits NORMALLY "
            f"so the parent can record the plugin's interpreter-exit behaviour",
            measured={"build_s": round(build_s, 3), "sum": float(out.sum()), "work_completed": ok},
        )
    except BaseException as exc:  # noqa: BLE001
        recorder.emit(
            "xla-plugin-exit",
            "fail",
            f"tiny xla graph failed: {_exc_text(exc)}\n"
            + "\n".join(traceback.format_exc().splitlines()[-6:]),
        )
    _finish_child(recorder, hard=False)


def child_iree(args: argparse.Namespace, path: Path) -> None:
    """The iree/ROCm leg through etl (device resident), plus a device-kind probe.

    The device-kind probe exists because ``etl.devices()`` lists BOTH ``cuda:0``
    (iree's CUDA driver enumerating a device it cannot actually use) and
    ``rocm:0`` on this host: a tiny graph is built on each kind so the benchmark
    harness learns which kind is real, with the losing kind's error verbatim.
    """
    recorder = ChildRecorder(path, meta={"kind": "iree", "models_dir": args.models_dir})
    try:
        import numpy as np

        import etl

        @etl.defn
        def add(a, b):
            return etl.add(a, b)

        kinds: dict = {}
        for kind in ("cuda", "rocm"):
            device = etl.core.Device(kind, 0)
            try:
                exe = etl.build(
                    add,
                    etl.TensorSpec((2, 2), etl.float32),
                    etl.TensorSpec((2, 2), etl.float32),
                    backend="iree",
                    device=device,
                )
                out = etl.run(
                    exe,
                    etl.tensor(np.ones((2, 2), np.float32)).to(device),
                    etl.tensor(np.full((2, 2), 2.0, np.float32)).to(device),
                ).to(etl.core.Device("cpu", 0)).numpy()
                kinds[kind] = {"usable": True, "sum": float(out.sum())}
            except BaseException as exc:  # noqa: BLE001 - an unusable kind is a result
                kinds[kind] = {"usable": False, "error": _exc_text(exc)}
        rocm_ok = bool(kinds.get("rocm", {}).get("usable"))
        recorder.emit(
            "iree-device-kinds",
            "pass" if rocm_ok else "fail",
            "tiny iree graph per accelerator kind: "
            + "; ".join(
                f"{kind}={'usable (sum %s)' % entry['sum'] if entry.get('usable') else 'NOT usable: ' + entry.get('error', '')}"
                for kind, entry in kinds.items()
            ),
            measured={"kinds": kinds, "etl_devices": [
                f"{device.kind}:{device.index}" for device in etl.devices()
            ]},
        )
    except BaseException as exc:  # noqa: BLE001
        recorder.emit(
            "iree-device-kinds",
            "fail",
            f"device-kind probe failed: {_exc_text(exc)}\n"
            + "\n".join(traceback.format_exc().splitlines()[-6:]),
        )
    try:
        _etl_backend_leg(
            recorder,
            check_name="iree-rocm",
            backend="iree",
            options={},
            rel_tol=IREE_REL_TOL,
            note=" [library default options]",
        )
    except BaseException as exc:  # noqa: BLE001
        recorder.emit(
            "iree-rocm",
            "fail",
            f"iree/rocm leg failed: {_exc_text(exc)}\n"
            + "\n".join(traceback.format_exc().splitlines()[-8:]),
            measured={"backend": "iree"},
        )
    _finish_child(recorder)


CHILDREN = {
    "versions": child_versions,
    "torch": child_torch,
    "xla": child_xla,
    "xla-exit": child_xla_exit,
    "iree": child_iree,
}


# ---------------------------------------------------------------------------
# parent: host / device probing
# ---------------------------------------------------------------------------
def rocm_smi() -> dict:
    """``rocm-smi --json`` product/driver/utilisation snapshot (never raises)."""
    raw: dict = {}
    for key, argv in (
        ("identity", ["rocm-smi", "--showproductname", "--showdriverversion", "--json"]),
        ("use", ["rocm-smi", "--showuse", "--json"]),
        ("version", ["rocm-smi", "--version"]),
    ):
        record = run_process(argv, timeout=60)
        raw[key] = {
            "exit_code": record["exit_code"],
            "stdout_tail": record["stdout_tail"],
            "stderr_tail": record["stderr_tail"],
            "error": record["error"],
        }
    parsed: dict = {}
    try:
        identity_text = "\n".join(raw["identity"]["stdout_tail"]) or "{}"
        cards = json.loads(identity_text)
        card = cards.get("card0", {})
        parsed = {
            "product": card.get("Card Series") or card.get("Card Model"),
            "gfx": card.get("GFX Version"),
            "vendor": card.get("Card Vendor"),
            "driver_version": (cards.get("system") or {}).get("Driver version"),
        }
    except (json.JSONDecodeError, AttributeError) as exc:
        parsed = {"parse_error": f"{_exc_text(exc)}: {raw['identity']['stdout_tail']}"}
    try:
        use = json.loads("\n".join(raw["use"]["stdout_tail"]))
        parsed["gpu_use_percent"] = use.get("card0", {}).get("GPU use (%)")
    except (json.JSONDecodeError, AttributeError):
        parsed["gpu_use_percent"] = None
    parsed["rocm_smi_version"] = " ".join(raw["version"]["stdout_tail"]) or None
    # rocm-smi's "Driver version" is the amdgpu KERNEL driver; the ROCm toolkit
    # version comes from /opt/rocm/.info/version (and torch.version.hip).
    try:
        parsed["rocm_version"] = (
            Path("/opt/rocm/.info/version").read_text().strip() or None
        )
    except OSError:
        parsed["rocm_version"] = None
    parsed["raw"] = raw
    return parsed


def host_block() -> dict:
    """Static host description (CPU/mem/OS/repo + the rocm-smi snapshot)."""
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "mem_total_gb": _mem_total_gb(),
        "python_system": sys.version.split()[0],
        "python_system_executable": sys.executable,
        "repo_root": str(REPO_ROOT),
        "tmpdir": os.environ.get("TMPDIR"),
        "rocm_smi": rocm_smi(),
    }


def resolve_plugin(explicit: str | None, venv: Path) -> str | None:
    """Resolve the PJRT plugin: explicit > staged ``$TMPDIR/pjrt`` > venv wheel.

    Args:
        explicit: ``--plugin`` value (wins when given).
        venv: venv root (locates the installed wheel's copy).

    Returns:
        str | None: the plugin path, or ``None`` when none is available.
    """
    if explicit:
        return explicit
    if STAGED_PLUGIN.is_file():
        return str(STAGED_PLUGIN)
    return wheel_plugin_path(venv)


# ---------------------------------------------------------------------------
# parent: orchestration
# ---------------------------------------------------------------------------
def run_children(
    *,
    python: Path,
    plugin: str | None,
    models_dir: Path,
    groups: list[str],
    check_timeout: float,
    work_dir: Path,
) -> tuple[list[dict], dict]:
    """Run each child group in its own process and merge its check entries.

    Args:
        python: the venv interpreter the children run in.
        plugin: PJRT plugin path (only the xla children get it in their env).
        models_dir: checkpoint root.
        groups: child groups to run, in order.
        check_timeout: per-child wall-clock budget (process group SIGKILLed).
        work_dir: where children write their partial JSON.

    Returns:
        tuple: ``(checks, child_records)`` — merged check entries and the run
        records of every child.
    """
    script = str(Path(__file__).resolve())
    checks: list[dict] = []
    child_records: dict = {}
    for group in groups:
        expected = CHILD_CHECKS[group]
        child_out = work_dir / f"child_{group}.json"
        if child_out.exists():
            child_out.unlink()
        argv = [
            str(python),
            script,
            "--child",
            group,
            "--child-out",
            str(child_out),
            "--models-dir",
            str(models_dir),
        ]
        wants_plugin = group in XLA_CHILDREN
        # The versions child only INSPECTS the plugin file (its env carries no
        # ETL_PJRT_PLUGIN / XLA_FLAGS, so the import checks stay xla-free).
        if plugin and (wants_plugin or group == "versions"):
            argv += ["--plugin", plugin]
        env = child_env(plugin if wants_plugin else None)
        print(f"[hf_bench_env] child {group}: {shlex.join(argv)}", flush=True)
        record = run_process(argv, timeout=check_timeout, env=env)
        child_records[group] = public_record(record)
        produced: dict = {}
        if child_out.exists():
            try:
                payload = json.loads(child_out.read_text())
                produced = {entry["name"]: entry for entry in payload.get("checks", [])}
                child_records[group]["child_meta"] = payload.get("meta")
            except json.JSONDecodeError as exc:
                child_records[group]["child_parse_error"] = _exc_text(exc)
        for name in expected:
            entry = produced.get(name)
            if entry is None:
                detail = (
                    f"check did not complete — child {group} {_describe_exit(record)}; "
                    f"last stderr: {' | '.join(record['stderr_tail'][-3:]) or '<empty>'}"
                )
                if record.get("error"):
                    detail = f"check did not run — {record['error']}"
                checks.append(
                    {
                        "name": name,
                        "status": "fail",
                        "detail": detail,
                        "versions": {},
                        "measured": {},
                        "child": group,
                        "child_exit": {
                            "exit_code": record["exit_code"],
                            "signal": record["signal"],
                            "timed_out": record["timed_out"],
                            "duration_s": record["duration_s"],
                        },
                    }
                )
                continue
            entry["child"] = group
            entry["child_exit"] = {
                "exit_code": record["exit_code"],
                "signal": record["signal"],
                "timed_out": record["timed_out"],
                "duration_s": record["duration_s"],
            }
            if group == "xla-exit" and name == "xla-plugin-exit" and entry.get("status") == "pass":
                # This child exits NORMALLY on purpose: the documented teardown
                # abort is EXPECTED and does not change the status (the work
                # completed before the exit — the whole point of the probe).
                aborted = record["signal"] in ("SIGABRT", "SIGSEGV") or record[
                    "exit_code"
                ] in (134, 139)
                last = record["stderr_tail"][-1] if record["stderr_tail"] else "<no stderr>"
                entry["measured"].update(
                    {
                        "teardown_aborted": bool(aborted),
                        "teardown_exit_code": record["exit_code"],
                        "teardown_signal": record["signal"],
                        "teardown_last_stderr": last,
                    }
                )
                entry["detail"] += (
                    f"; process teardown (expected, AFTER all work): "
                    f"{_describe_exit(record)}; last stderr: {last!r}"
                )
            checks.append(entry)
        print(
            f"[hf_bench_env] child {group}: {_describe_exit(record)} "
            f"({len(produced)}/{len(expected)} checks reported)",
            flush=True,
        )
    return checks, child_records


def missing_python_checks(groups: list[str], python: Path, reason: str) -> list[dict]:
    """Synthesize FAILED checks for every expected check when there is no python."""
    checks: list[dict] = []
    for group in groups:
        for name in CHILD_CHECKS[group]:
            checks.append(
                {
                    "name": name,
                    "status": "fail",
                    "detail": f"check did not run — {reason} (interpreter: {python})",
                    "versions": {},
                    "measured": {},
                    "child": group,
                }
            )
    return checks


def parent_checks(python: Path, host: dict, venv_info: dict) -> list[dict]:
    """Checks the parent produces itself (no venv imports needed)."""
    checks: list[dict] = []
    py_version = venv_info.get("python_version")
    ok = bool(py_version) and python.exists()
    checks.append(
        {
            "name": "python",
            "status": "pass" if ok else "fail",
            "detail": (
                f"venv interpreter {python}: {py_version}"
                if ok
                else f"venv interpreter {python} is missing or not runnable"
            ),
            "versions": {"python": py_version},
            "measured": {"python": str(python)},
            "child": "parent",
        }
    )
    smi = host.get("rocm_smi", {})
    product = smi.get("product")
    ok = bool(product)
    checks.append(
        {
            "name": "rocm-smi",
            "status": "pass" if ok else "fail",
            "detail": (
                f"rocm-smi reports {product} (gfx {smi.get('gfx')}), ROCm toolkit "
                f"{smi.get('rocm_version')}, amdgpu kernel driver "
                f"{smi.get('driver_version')}, GPU use {smi.get('gpu_use_percent')}%; "
                f"{smi.get('rocm_smi_version')}"
                if ok
                else f"rocm-smi did not report a GPU: {smi.get('parse_error') or smi.get('raw')}"
            ),
            "versions": {
                "rocm_version": smi.get("rocm_version"),
                "amdgpu_kernel_driver": smi.get("driver_version"),
            },
            "measured": {
                "gfx": smi.get("gfx"),
                "product": product,
                "gpu_use_percent": smi.get("gpu_use_percent"),
                "rocm_version": smi.get("rocm_version"),
                "amdgpu_kernel_driver": smi.get("driver_version"),
            },
            "child": "parent",
        }
    )
    return checks


def _child_meta(child_records: dict, key: str) -> dict:
    """The ``meta`` dict a child wrote (empty when it never got that far)."""
    return ((child_records.get(key) or {}).get("child_meta") or {})


def build_report(
    *,
    venv: Path,
    venv_info: dict,
    host: dict,
    checks: list[dict],
    child_records: dict,
    plugin: str | None,
) -> dict:
    """Assemble the self-describing JSON report (a superset of the required keys)."""
    versions: dict = {}
    for entry in checks:
        for key, value in (entry.get("versions") or {}).items():
            if key not in versions and value is not None:
                versions[key] = value
    for entry in checks:
        # child-reported versions are authoritative (they come from the venv)
        versions.update({k: v for k, v in (entry.get("versions") or {}).items() if v is not None})
    versions.update(_child_meta(child_records, "versions").get("versions") or {})
    versions["python"] = venv_info.get("python_version") or versions.get("python")
    versions["uv"] = venv_info.get("uv_version")
    versions.update(venv_info.get("packages") or {})

    etl_devices = _child_meta(child_records, "versions").get("etl_devices")
    iree_drivers = _child_meta(child_records, "versions").get("iree_drivers")
    plugin_matches = _child_meta(child_records, "versions").get("pjrt_plugin_matches_wheel")
    kind_probe = next(
        (entry.get("measured") or {} for entry in checks if entry["name"] == "iree-device-kinds"),
        {},
    )
    smi = host.get("rocm_smi", {})
    device = {
        "kind": "rocm",
        "index": 0,
        "name": smi.get("product"),
        "gfx": smi.get("gfx"),
        "rocm_version": smi.get("rocm_version"),
        "amdgpu_kernel_driver": smi.get("driver_version"),
        "hal_device_id": 1,
        "torch_device_string": "cuda:0 (torch exposes ROCm through its cuda API)",
        "etl_devices": etl_devices,
        "iree_drivers": iree_drivers,
        "accelerator_kind_probe": kind_probe.get("kinds"),
        "rocm_smi_gpu_use_percent": smi.get("gpu_use_percent"),
        "note": (
            "Use etl.core.Device('rocm', 0). etl.devices() ALSO lists cuda:0, but that "
            "kind is not runnable here — see accelerator_kind_probe for the verbatim "
            "error (iree's CUDA driver finds no libcuda.so on this AMD-only host)."
        ),
    }

    failed = [entry["name"] for entry in checks if entry.get("status") != "pass"]
    env_vars = {
        "PYTHONPATH": str(REPO_ROOT),
        "ETL_PJRT_PLUGIN": plugin,
        "XLA_FLAGS": XLA_FLAGS_RECOMMENDED,
    }
    torch_versions = next(
        (entry.get("versions") or {} for entry in checks if entry["name"] == "torch"), {}
    )
    torch_block = {
        "index_url": TORCH_INDEX_URL,
        "spec": TORCH_SPEC,
        "version": (venv_info.get("packages") or {}).get("torch")
        or torch_versions.get("torch"),
        "hip": torch_versions.get("torch_hip"),
        "device_string": "cuda:0 (torch maps ROCm onto its cuda API)",
    }
    return {
        "generated": _iso_now(),
        "host": host,
        "venv": {
            "path": str(venv),
            "python": venv_info.get("python"),
            "python_version": venv_info.get("python_version"),
            "created_this_run": venv_info.get("created"),
            "uv_version": venv_info.get("uv_version"),
            "packages": venv_info.get("packages") or {},
            "notes": venv_info.get("notes") or [],
        },
        "install_commands": [
            shlex.join(_render(UV_PIP_TORCH, python=str(venv_info.get("python") or ""))),
            shlex.join(_render(UV_PIP_REST, python=str(venv_info.get("python") or ""))),
        ],
        "install_commands_executed": venv_info.get("install_commands_executed") or [],
        "install_runs": venv_info.get("install_runs") or [],
        "versions": versions,
        "device": device,
        "checks": checks,
        "checks_summary": {
            "total": len(checks),
            "passed": len(checks) - len(failed),
            "failed": len(failed),
            "failed_names": failed,
        },
        "overall_status": "pass" if not failed else "fail",
        "xla": {
            "plugin_path": plugin,
            "plugin_path_candidates": {
                "used": plugin,
                "staged": str(STAGED_PLUGIN),
                "wheel": wheel_plugin_path(venv),
            },
            "plugin_matches_wheel": plugin_matches,
            "xla_flags": XLA_FLAGS_RECOMMENDED,
            "env_vars": env_vars,
            "plugin_exit_behaviour": (
                "A process that created a PJRT client ABORTS during interpreter teardown "
                "(SIGABRT / exit 134 with 'malloc_consolidate(): invalid chunk size', or "
                "SIGSEGV) AFTER all work succeeded; os._exit(0) is always clean. A benchmark "
                "must treat a 134/139 exit that follows completed work as normal and read "
                "the work's own output instead of the exit code."
            ),
        },
        "torch": torch_block,
        "env_vars": env_vars,
        "bench_recipe": (
            "Use <venv>/bin/python with PYTHONPATH=<repo root> so etl/etl_hf come from the "
            "checkout. For backend='xla' also set ETL_PJRT_PLUGIN and XLA_FLAGS as above and "
            "expect the documented teardown abort; run each xla leg in a subprocess and "
            "leave via os._exit(0)."
        ),
        "argv": sys.argv[1:],
        "children": child_records,
    }


def print_summary(report: dict) -> None:
    """Print a compact operator summary of the checks."""
    print("\n=== hf_bench_env summary ===", flush=True)
    interesting = (
        "maxabs_parity_vs_numpy",
        "rel_parity_vs_numpy",
        "maxabs_diff_vs_cpu",
        "rel_vs_cpu",
        "absmax",
        "build_s",
        "teardown_exit_code",
        "teardown_signal",
    )
    for entry in report["checks"]:
        measured = entry.get("measured") or {}
        numbers = ", ".join(
            f"{key}={measured[key]}" for key in interesting if key in measured
        )
        print(f"  [{entry['status'].upper():4}] {entry['name']:30} {numbers}", flush=True)
    summary = report["checks_summary"]
    tail = (
        f"; FAILED: {', '.join(summary['failed_names'])}"
        if summary["failed_names"]
        else ""
    )
    print(f"  -> {summary['passed']}/{summary['total']} pass{tail}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI (benchmark-facing flags plus the internal child mode)."""
    parser = argparse.ArgumentParser(
        prog="hf_bench_env",
        description=(
            "Provision + verify a throwaway etl_hf vs PyTorch benchmark environment "
            "(venv under $TMPDIR, never the repo .venv)."
        ),
    )
    parser.add_argument(
        "--venv", default=str(DEFAULT_VENV), help=f"venv path (default: {DEFAULT_VENV})"
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT), help=f"result JSON (default: {DEFAULT_OUT})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recreate the venv (uv venv --clear) and reinstall every wheel",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="skip venv creation/installs; only run the checks",
    )
    parser.add_argument(
        "--only",
        default=",".join(CHILD_ORDER),
        help=f"comma-separated child groups to run (default: {','.join(CHILD_ORDER)})",
    )
    parser.add_argument(
        "--check-timeout",
        type=float,
        default=900.0,
        help="per-child wall-clock budget in seconds (default: 900)",
    )
    parser.add_argument(
        "--install-timeout",
        type=float,
        default=1800.0,
        help="per-command budget for venv/install commands (default: 1800)",
    )
    parser.add_argument(
        "--models-dir",
        default=str(Path(os.environ.get("TMPDIR", "/tmp")) / "etl_hf_models"),
        help="checkpoint root (default: $TMPDIR/etl_hf_models; nothing is downloaded)",
    )
    parser.add_argument(
        "--plugin",
        default=None,
        help=(
            "PJRT plugin .so (default: $TMPDIR/pjrt/xla_rocm_plugin.so when present, "
            "else the jax-rocm7-pjrt wheel's copy)"
        ),
    )
    parser.add_argument(
        "--child", choices=sorted(CHILDREN), help="internal: run one check group here"
    )
    parser.add_argument("--child-out", help="internal: where the child writes its JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one child (``--child``) or the full provision + verify flow."""
    args = parse_args(argv)

    if args.child:
        if not args.child_out:
            print("--child requires --child-out", file=sys.stderr)
            return 2
        CHILDREN[args.child](args, Path(args.child_out))
        return 0  # unreachable: children always leave through _finish_child

    started = time.time()
    venv = Path(args.venv).expanduser().resolve()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Child result files are scratch: keep them out of the repository.
    work_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "hf_bench_env_children"
    work_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir).expanduser()

    groups = [group.strip() for group in args.only.split(",") if group.strip()]
    unknown = [group for group in groups if group not in CHILD_CHECKS]
    if unknown:
        print(
            f"unknown child group(s): {unknown}; known: {list(CHILD_CHECKS)}",
            file=sys.stderr,
        )
        return 2

    print(f"[hf_bench_env] repo root: {REPO_ROOT}", flush=True)
    print(f"[hf_bench_env] venv:      {venv}", flush=True)

    if args.verify_only:
        python_path = venv / "bin" / "python"
        record = run_process([str(python_path), "-V"], timeout=120)
        uv_record = run_process(["uv", "--version"], timeout=60)
        venv_info = {
            "created": False,
            "python": str(python_path),
            "python_version": (
                " ".join(record["stdout_tail"]).strip() if record["exit_code"] == 0 else None
            ),
            "uv_version": (
                " ".join(uv_record["stdout_tail"]).strip()
                if uv_record["exit_code"] == 0
                else None
            ),
            "install_commands_executed": [],
            "install_runs": [],
            "packages": installed_packages(python_path, timeout=300),
            "packages_before": {},
            "notes": ["--verify-only: no venv creation, no installs"],
        }
    else:
        venv_info = provision(venv, force=args.force, install_timeout=args.install_timeout)
    print(
        f"[hf_bench_env] venv ready: {venv_info.get('python_version')} "
        f"({venv_info.get('python')}); install commands run this invocation: "
        f"{len(venv_info.get('install_commands_executed') or [])}",
        flush=True,
    )

    python = Path(venv_info.get("python") or (venv / "bin" / "python"))
    plugin = resolve_plugin(args.plugin, venv)
    print(f"[hf_bench_env] pjrt plugin: {plugin}", flush=True)

    host = host_block()
    checks = parent_checks(python, host, venv_info)
    if python.exists():
        child_checks, child_records = run_children(
            python=python,
            plugin=plugin,
            models_dir=models_dir,
            groups=groups,
            check_timeout=args.check_timeout,
            work_dir=work_dir,
        )
    else:
        child_checks, child_records = (
            missing_python_checks(
                groups,
                python,
                "the venv interpreter is missing"
                + (
                    f": {' | '.join(venv_info.get('notes') or [])}"
                    if venv_info.get("notes")
                    else ""
                ),
            ),
            {},
        )
    checks.extend(child_checks)

    report = build_report(
        venv=venv,
        venv_info=venv_info,
        host=host,
        checks=checks,
        child_records=child_records,
        plugin=plugin,
    )
    report["duration_s"] = round(time.time() - started, 1)
    out_path.write_text(json.dumps(report, indent=2))
    print_summary(report)
    print(f"[hf_bench_env] wrote {out_path} ({report['overall_status']})", flush=True)
    return 0 if report["overall_status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
