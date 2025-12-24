"""
Tests for zero-copy functionality in the container module.

This module tests the NdarrayContainer batch handling with zero-copy optimizations.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestNdarrayContainerBatchesToBatch:
    """Tests for NdarrayContainer.batches_to_batch method."""

    def test_simple_batch(self):
        """Test combining simple arrays into a batch."""
        from bentoml._internal.runner.container import NdarrayContainer

        batches = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
        ]

        batch, indices = NdarrayContainer.batches_to_batch(batches, batch_dim=0)

        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        np.testing.assert_array_equal(batch, expected)
        assert indices == [0, 2, 4]

    def test_different_batch_sizes(self):
        """Test combining arrays with different batch sizes."""
        from bentoml._internal.runner.container import NdarrayContainer

        batches = [
            np.array([[1, 2]]),
            np.array([[3, 4], [5, 6], [7, 8]]),
        ]

        batch, indices = NdarrayContainer.batches_to_batch(batches, batch_dim=0)

        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        np.testing.assert_array_equal(batch, expected)
        assert indices == [0, 1, 4]

    def test_empty_batches_raises_error(self):
        """Test that empty batches list raises ValueError."""
        from bentoml._internal.runner.container import NdarrayContainer

        with pytest.raises(ValueError, match="Cannot create batch from empty sequence"):
            NdarrayContainer.batches_to_batch([], batch_dim=0)

    def test_batch_dim_1(self):
        """Test batching along dimension 1."""
        from bentoml._internal.runner.container import NdarrayContainer

        batches = [
            np.array([[1], [2]]),  # shape (2, 1)
            np.array([[3, 4], [5, 6]]),  # shape (2, 2)
        ]

        batch, indices = NdarrayContainer.batches_to_batch(batches, batch_dim=1)

        expected = np.array([[1, 3, 4], [2, 5, 6]])  # shape (2, 3)
        np.testing.assert_array_equal(batch, expected)
        assert indices == [0, 1, 3]

    def test_preserves_dtype(self):
        """Test that dtype is preserved during batching."""
        from bentoml._internal.runner.container import NdarrayContainer

        batches = [
            np.array([[1, 2]], dtype=np.float32),
            np.array([[3, 4]], dtype=np.float32),
        ]

        batch, indices = NdarrayContainer.batches_to_batch(batches, batch_dim=0)

        assert batch.dtype == np.float32


class TestNdarrayContainerBatchToBatches:
    """Tests for NdarrayContainer.batch_to_batches method."""

    def test_split_batch(self):
        """Test splitting a batch into individual arrays."""
        from bentoml._internal.runner.container import NdarrayContainer

        batch = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        indices = [0, 2, 4]

        batches = NdarrayContainer.batch_to_batches(batch, indices, batch_dim=0)

        assert len(batches) == 2
        np.testing.assert_array_equal(batches[0], [[1, 2], [3, 4]])
        np.testing.assert_array_equal(batches[1], [[5, 6], [7, 8]])

    def test_split_uneven_batch(self):
        """Test splitting a batch with uneven sizes."""
        from bentoml._internal.runner.container import NdarrayContainer

        batch = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        indices = [0, 1, 4]

        batches = NdarrayContainer.batch_to_batches(batch, indices, batch_dim=0)

        assert len(batches) == 2
        np.testing.assert_array_equal(batches[0], [[1, 2]])
        np.testing.assert_array_equal(batches[1], [[3, 4], [5, 6], [7, 8]])

    def test_split_returns_views(self):
        """Test that split returns views when possible (zero-copy behavior)."""
        from bentoml._internal.runner.container import NdarrayContainer

        batch = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        indices = [0, 2, 4]

        batches = NdarrayContainer.batch_to_batches(batch, indices, batch_dim=0)

        # np.split returns views, so modifying batch should affect batches
        # (unless a copy was made)
        # This is the zero-copy behavior we want
        original_batch = batch.copy()
        batch[0, 0] = 999

        # The split arrays should see the change (they're views)
        assert batches[0][0, 0] == 999

        # Restore
        batch[0, 0] = original_batch[0, 0]


class TestNdarrayContainerPayload:
    """Tests for NdarrayContainer payload methods."""

    def test_to_payload(self):
        """Test converting array to payload."""
        from bentoml._internal.runner.container import NdarrayContainer

        arr = np.array([[1, 2], [3, 4]], dtype=np.float64)

        payload = NdarrayContainer.to_payload(arr, batch_dim=0)

        assert payload.batch_size == 2
        assert payload.container == "NdarrayContainer"
        assert len(payload.data) > 0

    def test_from_payload(self):
        """Test converting payload back to array."""
        from bentoml._internal.runner.container import NdarrayContainer

        arr = np.array([[1, 2], [3, 4]], dtype=np.float64)

        payload = NdarrayContainer.to_payload(arr, batch_dim=0)
        result = NdarrayContainer.from_payload(payload)

        np.testing.assert_array_equal(result, arr)

    def test_roundtrip_large_array(self):
        """Test roundtrip of a large array."""
        from bentoml._internal.runner.container import NdarrayContainer

        arr = np.random.randn(100, 100)

        payload = NdarrayContainer.to_payload(arr, batch_dim=0)
        result = NdarrayContainer.from_payload(payload)

        np.testing.assert_array_almost_equal(result, arr)

    def test_roundtrip_different_dtypes(self):
        """Test roundtrip with different dtypes."""
        from bentoml._internal.runner.container import NdarrayContainer

        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            arr = np.array([[1, 2], [3, 4]], dtype=dtype)

            payload = NdarrayContainer.to_payload(arr, batch_dim=0)
            result = NdarrayContainer.from_payload(payload)

            np.testing.assert_array_equal(result, arr)
            assert result.dtype == dtype

    def test_get_batch_size(self):
        """Test getting batch size from array."""
        from bentoml._internal.runner.container import NdarrayContainer

        arr = np.array([[1, 2], [3, 4], [5, 6]])

        size = NdarrayContainer.get_batch_size(arr, batch_dim=0)

        assert size == 3

    def test_non_contiguous_array(self):
        """Test handling non-contiguous arrays."""
        from bentoml._internal.runner.container import NdarrayContainer

        # Create non-contiguous array by transposing
        arr = np.array([[1, 2], [3, 4], [5, 6]]).T
        assert not arr.flags["C_CONTIGUOUS"]

        payload = NdarrayContainer.to_payload(arr, batch_dim=0)
        result = NdarrayContainer.from_payload(payload)

        np.testing.assert_array_equal(result, arr)


class TestNdarrayContainerZeroCopyIntegration:
    """Integration tests for NdarrayContainer with zero-copy settings."""

    def test_batching_with_zero_copy_disabled(self):
        """Test batching behavior when zero_copy is disabled."""
        from bentoml._internal.runner.container import NdarrayContainer

        batches = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
            np.array([[9, 10], [11, 12]]),
        ]

        # With zero_copy disabled (default), should use np.concatenate
        batch, indices = NdarrayContainer.batches_to_batch(batches, batch_dim=0)

        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        np.testing.assert_array_equal(batch, expected)

    def test_batching_preserves_memory_layout(self):
        """Test that batching produces C-contiguous array."""
        from bentoml._internal.runner.container import NdarrayContainer

        batches = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
        ]

        batch, indices = NdarrayContainer.batches_to_batch(batches, batch_dim=0)

        assert batch.flags["C_CONTIGUOUS"]


class TestPayloadClass:
    """Tests for the Payload class."""

    def test_payload_creation(self):
        """Test creating a Payload."""
        from bentoml._internal.runner.container import Payload

        data = b"test data"
        meta = {"key": "value"}

        payload = Payload(data, meta, "TestContainer", batch_size=5)

        assert payload.data == data
        assert payload.meta == {"key": "value"}
        assert payload.container == "TestContainer"
        assert payload.batch_size == 5

    def test_payload_default_batch_size(self):
        """Test Payload default batch_size."""
        from bentoml._internal.runner.container import Payload

        payload = Payload(b"data", {}, "TestContainer")

        assert payload.batch_size == -1


class TestDataContainerRegistry:
    """Tests for DataContainerRegistry."""

    def test_find_by_single_type_numpy(self):
        """Test finding container by numpy array type."""
        from bentoml._internal.runner.container import DataContainerRegistry
        from bentoml._internal.runner.container import NdarrayContainer

        container = DataContainerRegistry.find_by_single_type(np.ndarray)

        assert container is NdarrayContainer

    def test_find_by_batch_type_numpy(self):
        """Test finding container by numpy array batch type."""
        from bentoml._internal.runner.container import DataContainerRegistry
        from bentoml._internal.runner.container import NdarrayContainer

        container = DataContainerRegistry.find_by_batch_type(np.ndarray)

        assert container is NdarrayContainer

    def test_find_by_name(self):
        """Test finding container by name."""
        from bentoml._internal.runner.container import DataContainerRegistry
        from bentoml._internal.runner.container import NdarrayContainer

        container = DataContainerRegistry.find_by_name("NdarrayContainer")

        assert container is NdarrayContainer

    def test_find_by_name_unknown(self):
        """Test finding unknown container by name raises error."""
        from bentoml._internal.runner.container import DataContainerRegistry

        with pytest.raises(ValueError, match="can not find specified container"):
            DataContainerRegistry.find_by_name("UnknownContainer")


class TestAutoContainer:
    """Tests for AutoContainer."""

    def test_auto_container_numpy(self):
        """Test AutoContainer with numpy array."""
        from bentoml._internal.runner.container import AutoContainer

        arr = np.array([[1, 2], [3, 4]])

        payload = AutoContainer.to_payload(arr, batch_dim=0)

        assert payload.container == "NdarrayContainer"

    def test_auto_container_batch_size(self):
        """Test AutoContainer.get_batch_size."""
        from bentoml._internal.runner.container import AutoContainer

        arr = np.array([[1, 2], [3, 4], [5, 6]])

        size = AutoContainer.get_batch_size(arr, batch_dim=0)

        assert size == 3

    def test_auto_container_from_payload(self):
        """Test AutoContainer.from_payload."""
        from bentoml._internal.runner.container import AutoContainer

        arr = np.array([[1, 2], [3, 4]])

        payload = AutoContainer.to_payload(arr, batch_dim=0)
        result = AutoContainer.from_payload(payload)

        np.testing.assert_array_equal(result, arr)

    def test_auto_container_batches_to_batch(self):
        """Test AutoContainer.batches_to_batch."""
        from bentoml._internal.runner.container import AutoContainer

        batches = [
            np.array([[1, 2]]),
            np.array([[3, 4]]),
        ]

        batch, indices = AutoContainer.batches_to_batch(batches, batch_dim=0)

        expected = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(batch, expected)

    def test_auto_container_batch_to_batches(self):
        """Test AutoContainer.batch_to_batches."""
        from bentoml._internal.runner.container import AutoContainer

        batch = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        indices = [0, 2, 4]

        batches = AutoContainer.batch_to_batches(batch, indices, batch_dim=0)

        assert len(batches) == 2
