"""Small private validation helpers shared by :mod:`etl_hf.program` / :mod:`etl_hf.pipeline`.

Every helper validates **one argument** and puts the argument's name in the
error message, so a failure points at the call site instead of at some internal
attribute access three frames down. They are private (no ``__all__`` export from
the package): the public API surface is :mod:`etl_hf.program` /
:mod:`etl_hf.pipeline`, and the messages name the public functions.

Import-time constraint: only ``etl.core`` (for :class:`~etl.core.Device`) and
sibling ``etl_hf`` modules — never torch / transformers / safetensors / iree.
"""

from __future__ import annotations

from etl import core as _core
from etl_hf.config import ModelConfig
from etl_hf.errors import HFConfigError

__all__ = ["positive_int", "require_bool", "require_config", "require_device"]


def positive_int(value: object, what: str) -> int:
    """Validate a positive integer argument.

    Args:
        value: the candidate value.
        what: argument name used in the error message.

    Returns:
        int: ``value`` as an ``int``.

    Raises:
        HFConfigError: ``value`` is not a positive integer (booleans are
            rejected even though ``bool`` is an ``int`` subclass — a ``True``
            sequence length is always a bug).
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HFConfigError(
            f"{what} must be a positive integer, got {value!r} "
            f"(type {type(value).__name__})"
        )
    return int(value)


def require_config(cfg: object) -> ModelConfig:
    """Validate that ``cfg`` is a :class:`ModelConfig`.

    Args:
        cfg: the candidate config.

    Returns:
        ModelConfig: ``cfg`` unchanged.

    Raises:
        TypeError: ``cfg`` is not a :class:`ModelConfig` (fail before any
            attribute access, so the message is actionable).
    """
    if not isinstance(cfg, ModelConfig):
        raise TypeError(
            "expected an etl_hf.ModelConfig, got "
            f"{type(cfg).__name__}; load one with ModelConfig.from_dir(model_dir) "
            "or ModelConfig.from_json(config_path)"
        )
    return cfg


def require_bool(value: object, what: str) -> bool:
    """Validate a boolean flag argument.

    Args:
        value: the candidate value.
        what: argument name used in the error message.

    Returns:
        bool: ``value`` as a ``bool``.

    Raises:
        TypeError: ``value`` is not a ``bool`` (an int/str would silently flip
            the graph's params tree shape, so it is rejected).
    """
    if not isinstance(value, bool):
        raise TypeError(
            f"{what} must be a bool (True/False), got {value!r} of type "
            f"{type(value).__name__}"
        )
    return value


def require_device(device: object, what: str) -> _core.Device:
    """Validate that ``device`` is an explicit :class:`etl.core.Device`.

    Args:
        device: the candidate device.
        what: argument name used in the error message.

    Returns:
        etl.core.Device: ``device`` unchanged.

    Raises:
        TypeError: ``device`` is not a :class:`etl.core.Device`; etl_hf never
            guesses a device from a string (that would be hidden placement).
    """
    if not isinstance(device, _core.Device):
        raise TypeError(
            f"{what} must be an etl.core.Device (e.g. Device('cpu', 0) or "
            f"Device('cuda', 0)), got {device!r} of type {type(device).__name__}"
        )
    return device
