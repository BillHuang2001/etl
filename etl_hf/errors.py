"""Exception hierarchy for the ``etl_hf`` package.

Every error raised by ``etl_hf`` derives from :class:`HFError`, which itself
derives from :class:`etl.core.ETLError` so callers can catch the whole etl
error surface with a single ``except ETLError``.

Import-time constraint: this module imports **only** ``etl.core``. It must never
pull in numpy or the optional dependencies (torch / transformers / iree /
safetensors), so that ``import etl_hf.errors`` stays dependency-free.
"""

from etl.core import ETLError

__all__ = ["HFError", "HFConfigError", "HFWeightError", "HFUnsupportedError"]


class HFError(ETLError):
    """Base class for every error raised by ``etl_hf``.

    Catch this to handle any HuggingFace-integration failure without also
    catching unrelated errors from the underlying ``etl`` package.
    """


class HFConfigError(HFError):
    """Malformed or unsupported ``config.json``.

    Raised when a required field is missing/invalid or a config value names a
    feature outside the supported matrix. The message names the offending field
    and the file it came from.
    """


class HFWeightError(HFError):
    """Checkpoint reading, key-mapping, or tensor-shape problem.

    Raised when a checkpoint file cannot be parsed, a required weight key is
    missing, or a loaded tensor has a shape incompatible with the config. The
    message names the file/field/tensor involved.
    """


class HFUnsupportedError(HFError):
    """A well-formed request outside the v1 support matrix.

    Raised when the config and checkpoint are valid but the requested model
    family, activation, or backend path is not implemented in this version.
    """
