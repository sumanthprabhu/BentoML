"""
Zero-Copy Configuration Module for BentoML.

This module provides configuration management for zero-copy tensor serialization,
allowing users to enable/disable zero-copy optimizations at the service level.
"""

from __future__ import annotations

import contextvars
import typing as t
from dataclasses import dataclass

# Context variable for zero-copy settings
_zero_copy_enabled: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "zero_copy_enabled", default=False
)


@dataclass(frozen=True)
class ZeroCopyConfig:
    """
    Configuration for zero-copy tensor serialization.

    Attributes:
        enabled: Whether zero-copy optimizations are enabled
        preallocate_batch_threshold: Number of batches above which to use
            pre-allocation strategy (default: 4)
        use_arrow_ipc: Whether to use Arrow IPC format for tensor serialization
        preserve_tensor_shape: Whether to preserve tensor shape in Arrow serialization
    """

    enabled: bool = False
    preallocate_batch_threshold: int = 4
    use_arrow_ipc: bool = False
    preserve_tensor_shape: bool = True

    def __post_init__(self) -> None:
        if self.preallocate_batch_threshold < 1:
            raise ValueError("preallocate_batch_threshold must be >= 1")


# Global default configuration
_default_config = ZeroCopyConfig()


def get_zero_copy_enabled() -> bool:
    """Get whether zero-copy is enabled in current context."""
    return _zero_copy_enabled.get()


def set_zero_copy_enabled(enabled: bool) -> contextvars.Token[bool]:
    """Set zero-copy enabled state in current context."""
    return _zero_copy_enabled.set(enabled)


def get_default_config() -> ZeroCopyConfig:
    """Get the default zero-copy configuration."""
    return _default_config


def set_default_config(config: ZeroCopyConfig) -> None:
    """Set the default zero-copy configuration."""
    global _default_config
    _default_config = config


class ZeroCopyContext:
    """
    Context manager for temporarily enabling/disabling zero-copy.

    Example:
        with ZeroCopyContext(enabled=True):
            # Zero-copy is enabled here
            result = serialize_tensor(data)
    """

    def __init__(self, enabled: bool = True, config: ZeroCopyConfig | None = None):
        self.enabled = enabled
        self.config = config
        self._token: contextvars.Token[bool] | None = None
        self._old_config: ZeroCopyConfig | None = None

    def __enter__(self) -> "ZeroCopyContext":
        self._token = set_zero_copy_enabled(self.enabled)
        if self.config is not None:
            self._old_config = get_default_config()
            set_default_config(self.config)
        return self

    def __exit__(self, exc_type: t.Any, exc_val: t.Any, exc_tb: t.Any) -> None:
        if self._token is not None:
            _zero_copy_enabled.reset(self._token)
        if self._old_config is not None:
            set_default_config(self._old_config)


def zero_copy_context(
    enabled: bool = True,
    config: ZeroCopyConfig | None = None,
) -> ZeroCopyContext:
    """
    Create a context manager for zero-copy settings.

    Args:
        enabled: Whether to enable zero-copy in this context
        config: Optional custom configuration for this context

    Returns:
        A context manager that sets zero-copy settings

    Example:
        with zero_copy_context(enabled=True):
            # Zero-copy enabled
            pass
    """
    return ZeroCopyContext(enabled=enabled, config=config)


__all__ = [
    "ZeroCopyConfig",
    "get_zero_copy_enabled",
    "set_zero_copy_enabled",
    "get_default_config",
    "set_default_config",
    "ZeroCopyContext",
    "zero_copy_context",
]
