"""Private lazy-torch helpers for ``etl.bench``.

torch-optionality contract (binding): ``import etl`` and ``import etl.bench``
must ALWAYS succeed without torch installed. torch is imported ONLY inside
function bodies in this subpackage — never at module top level, never at
import time. When torch is missing, callers get a clear ``ImportError``
mentioning ``pip install etl[bench]`` — never a raw ``ModuleNotFoundError``
traceback escaping to the user.
"""
from __future__ import annotations

__all__ = ["torch_available", "require_torch"]

_HINT = (
    "Install it with `pip install etl[bench]` (or `pip install torch`) and "
    "retry."
)

_probed = False
_available = False


def torch_available() -> bool:
    """Return True iff torch can be imported (probed once, then cached)."""
    global _probed, _available
    if not _probed:
        _probed = True
        try:
            import torch  # noqa: F401  (lazy: optional dependency)
        except Exception:
            _available = False
        else:
            _available = True
    return _available


def require_torch():
    """Import and return torch, or raise a clear ``ImportError`` with the
    ``pip install etl[bench]`` hint when torch is unavailable."""
    global _probed, _available
    if not _probed:
        _probed = True
        try:
            import torch  # noqa: F401  (lazy: optional dependency)
        except Exception as exc:
            _available = False
            raise ImportError(
                "torch is required for this etl.bench operation but could "
                f"not be imported ({type(exc).__name__}: {exc}). {_HINT}"
            ) from exc
        else:
            _available = True
    elif not _available:
        raise ImportError(
            "torch is required for this etl.bench operation but is not "
            f"installed. {_HINT}"
        )
    return torch
