"""
Tests for the zero_copy_config module.

This module tests the configuration management for zero-copy tensor serialization,
including context variables, configuration dataclasses, and context managers.
"""

from __future__ import annotations

import pytest


class TestZeroCopyConfig:
    """Tests for the ZeroCopyConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from _bentoml_impl.zero_copy_config import ZeroCopyConfig

        config = ZeroCopyConfig()
        assert config.enabled is False
        assert config.preallocate_batch_threshold == 4
        assert config.use_arrow_ipc is False
        assert config.preserve_tensor_shape is True

    def test_custom_values(self):
        """Test custom configuration values."""
        from _bentoml_impl.zero_copy_config import ZeroCopyConfig

        config = ZeroCopyConfig(
            enabled=True,
            preallocate_batch_threshold=8,
            use_arrow_ipc=True,
            preserve_tensor_shape=False,
        )
        assert config.enabled is True
        assert config.preallocate_batch_threshold == 8
        assert config.use_arrow_ipc is True
        assert config.preserve_tensor_shape is False

    def test_invalid_preallocate_threshold(self):
        """Test that invalid preallocate_batch_threshold raises ValueError."""
        from _bentoml_impl.zero_copy_config import ZeroCopyConfig

        with pytest.raises(
            ValueError, match="preallocate_batch_threshold must be >= 1"
        ):
            ZeroCopyConfig(preallocate_batch_threshold=0)

        with pytest.raises(
            ValueError, match="preallocate_batch_threshold must be >= 1"
        ):
            ZeroCopyConfig(preallocate_batch_threshold=-1)

    def test_frozen_dataclass(self):
        """Test that ZeroCopyConfig is immutable."""
        from _bentoml_impl.zero_copy_config import ZeroCopyConfig

        config = ZeroCopyConfig()
        with pytest.raises(AttributeError):
            config.enabled = True  # type: ignore


class TestContextVariables:
    """Tests for context variable management."""

    def test_get_zero_copy_enabled_default(self):
        """Test default value of zero_copy_enabled."""
        from _bentoml_impl.zero_copy_config import get_zero_copy_enabled

        # Default should be False
        assert get_zero_copy_enabled() is False

    def test_set_zero_copy_enabled(self):
        """Test setting zero_copy_enabled."""
        from _bentoml_impl.zero_copy_config import _zero_copy_enabled
        from _bentoml_impl.zero_copy_config import get_zero_copy_enabled
        from _bentoml_impl.zero_copy_config import set_zero_copy_enabled

        # Enable zero_copy
        token = set_zero_copy_enabled(True)
        assert get_zero_copy_enabled() is True

        # Reset to original value
        _zero_copy_enabled.reset(token)
        assert get_zero_copy_enabled() is False

    def test_get_default_config(self):
        """Test getting the default configuration."""
        from _bentoml_impl.zero_copy_config import ZeroCopyConfig
        from _bentoml_impl.zero_copy_config import get_default_config

        config = get_default_config()
        assert isinstance(config, ZeroCopyConfig)

    def test_set_default_config(self):
        """Test setting the default configuration."""
        from _bentoml_impl.zero_copy_config import ZeroCopyConfig
        from _bentoml_impl.zero_copy_config import get_default_config
        from _bentoml_impl.zero_copy_config import set_default_config

        original = get_default_config()

        try:
            new_config = ZeroCopyConfig(enabled=True, preallocate_batch_threshold=16)
            set_default_config(new_config)

            assert get_default_config() is new_config
            assert get_default_config().enabled is True
            assert get_default_config().preallocate_batch_threshold == 16
        finally:
            # Restore original config
            set_default_config(original)


class TestZeroCopyContext:
    """Tests for ZeroCopyContext context manager."""

    def test_context_enables_zero_copy(self):
        """Test that ZeroCopyContext enables zero_copy in context."""
        from _bentoml_impl.zero_copy_config import ZeroCopyContext
        from _bentoml_impl.zero_copy_config import get_zero_copy_enabled

        # Before context
        assert get_zero_copy_enabled() is False

        with ZeroCopyContext(enabled=True):
            # Inside context
            assert get_zero_copy_enabled() is True

        # After context
        assert get_zero_copy_enabled() is False

    def test_context_disables_zero_copy(self):
        """Test that ZeroCopyContext can disable zero_copy."""
        from _bentoml_impl.zero_copy_config import ZeroCopyContext
        from _bentoml_impl.zero_copy_config import _zero_copy_enabled
        from _bentoml_impl.zero_copy_config import get_zero_copy_enabled
        from _bentoml_impl.zero_copy_config import set_zero_copy_enabled

        # First enable it
        token = set_zero_copy_enabled(True)

        try:
            assert get_zero_copy_enabled() is True

            with ZeroCopyContext(enabled=False):
                # Inside context
                assert get_zero_copy_enabled() is False

            # After context - should be restored
            assert get_zero_copy_enabled() is True
        finally:
            _zero_copy_enabled.reset(token)

    def test_context_with_custom_config(self):
        """Test ZeroCopyContext with custom configuration."""
        from _bentoml_impl.zero_copy_config import ZeroCopyConfig
        from _bentoml_impl.zero_copy_config import ZeroCopyContext
        from _bentoml_impl.zero_copy_config import get_default_config

        original_config = get_default_config()
        custom_config = ZeroCopyConfig(enabled=True, preallocate_batch_threshold=32)

        with ZeroCopyContext(enabled=True, config=custom_config):
            current_config = get_default_config()
            assert current_config.preallocate_batch_threshold == 32
            assert current_config.enabled is True

        # Config should be restored
        assert get_default_config() is original_config

    def test_nested_contexts(self):
        """Test nested ZeroCopyContext managers."""
        from _bentoml_impl.zero_copy_config import ZeroCopyContext
        from _bentoml_impl.zero_copy_config import get_zero_copy_enabled

        assert get_zero_copy_enabled() is False

        with ZeroCopyContext(enabled=True):
            assert get_zero_copy_enabled() is True

            with ZeroCopyContext(enabled=False):
                assert get_zero_copy_enabled() is False

            # After inner context
            assert get_zero_copy_enabled() is True

        # After outer context
        assert get_zero_copy_enabled() is False

    def test_context_exception_handling(self):
        """Test that context restores state even on exception."""
        from _bentoml_impl.zero_copy_config import ZeroCopyContext
        from _bentoml_impl.zero_copy_config import get_zero_copy_enabled

        assert get_zero_copy_enabled() is False

        with pytest.raises(ValueError):
            with ZeroCopyContext(enabled=True):
                assert get_zero_copy_enabled() is True
                raise ValueError("Test exception")

        # State should still be restored
        assert get_zero_copy_enabled() is False


class TestZeroCopyContextFunction:
    """Tests for the zero_copy_context function."""

    def test_zero_copy_context_function(self):
        """Test the zero_copy_context helper function."""
        from _bentoml_impl.zero_copy_config import get_zero_copy_enabled
        from _bentoml_impl.zero_copy_config import zero_copy_context

        with zero_copy_context(enabled=True):
            assert get_zero_copy_enabled() is True

        assert get_zero_copy_enabled() is False

    def test_zero_copy_context_with_config(self):
        """Test zero_copy_context with custom config."""
        from _bentoml_impl.zero_copy_config import ZeroCopyConfig
        from _bentoml_impl.zero_copy_config import get_default_config
        from _bentoml_impl.zero_copy_config import zero_copy_context

        custom_config = ZeroCopyConfig(enabled=True, use_arrow_ipc=True)

        with zero_copy_context(enabled=True, config=custom_config):
            assert get_default_config().use_arrow_ipc is True

        assert get_default_config().use_arrow_ipc is False
