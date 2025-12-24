"""
Tests for the zero_copy module.

This module tests the zero-copy tensor serialization/deserialization utilities
including NumPy, PyTorch, and TensorFlow tensor handling.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestZeroCopyBuffer:
    """Tests for ZeroCopyBuffer class."""

    def test_buffer_from_bytes(self):
        """Test creating ZeroCopyBuffer from bytes."""
        from _bentoml_impl.zero_copy import ZeroCopyBuffer

        data = b"hello world"
        buffer = ZeroCopyBuffer(data)

        assert len(buffer) == len(data)
        assert buffer.data == data

    def test_buffer_from_memoryview(self):
        """Test creating ZeroCopyBuffer from memoryview."""
        from _bentoml_impl.zero_copy import ZeroCopyBuffer

        data = b"hello world"
        mv = memoryview(data)
        buffer = ZeroCopyBuffer(mv)

        assert len(buffer) == len(data)

    def test_buffer_from_bytearray(self):
        """Test creating ZeroCopyBuffer from bytearray."""
        from _bentoml_impl.zero_copy import ZeroCopyBuffer

        data = bytearray(b"hello world")
        buffer = ZeroCopyBuffer(data)

        assert len(buffer) == len(data)

    def test_buffer_protocol(self):
        """Test that ZeroCopyBuffer supports buffer protocol."""
        from _bentoml_impl.zero_copy import ZeroCopyBuffer

        data = b"hello world"
        buffer = ZeroCopyBuffer(data)

        # Should be able to create memoryview from buffer
        mv = memoryview(buffer)
        assert bytes(mv) == data


class TestTensorTypeDetection:
    """Tests for tensor type detection functions."""

    def test_is_torch_tensor_with_numpy(self):
        """Test that numpy arrays are not detected as torch tensors."""
        from _bentoml_impl.zero_copy import _is_torch_tensor

        arr = np.array([1, 2, 3])
        assert _is_torch_tensor(arr) is False

    def test_is_tf_tensor_with_numpy(self):
        """Test that numpy arrays are not detected as tf tensors."""
        from _bentoml_impl.zero_copy import _is_tf_tensor

        arr = np.array([1, 2, 3])
        assert _is_tf_tensor(arr) is False

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"),
        reason="torch not installed",
    )
    def test_is_torch_tensor_with_torch(self):
        """Test that torch tensors are correctly detected."""
        import torch

        from _bentoml_impl.zero_copy import _is_torch_tensor

        tensor = torch.tensor([1, 2, 3])
        assert _is_torch_tensor(tensor) is True

    @pytest.mark.skipif(
        not pytest.importorskip("tensorflow", reason="tensorflow not installed"),
        reason="tensorflow not installed",
    )
    def test_is_tf_tensor_with_tf(self):
        """Test that tf tensors are correctly detected."""
        import tensorflow as tf

        from _bentoml_impl.zero_copy import _is_tf_tensor

        tensor = tf.constant([1, 2, 3])
        assert _is_tf_tensor(tensor) is True


class TestTensorToNumpyZeroCopy:
    """Tests for tensor_to_numpy_zero_copy function."""

    def test_numpy_array_passthrough(self):
        """Test that numpy arrays pass through without copying when zero_copy enabled."""
        from _bentoml_impl.zero_copy import tensor_to_numpy_zero_copy

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = tensor_to_numpy_zero_copy(arr, zero_copy=True)

        # Should be the same object
        assert result is arr

    def test_numpy_array_force_copy(self):
        """Test that numpy arrays are copied when force_copy is True."""
        from _bentoml_impl.zero_copy import tensor_to_numpy_zero_copy

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = tensor_to_numpy_zero_copy(arr, force_copy=True, zero_copy=True)

        # Should be a different object
        assert result is not arr
        np.testing.assert_array_equal(result, arr)

    def test_numpy_array_copy_when_zero_copy_disabled(self):
        """Test that numpy arrays are copied when zero_copy is disabled."""
        from _bentoml_impl.zero_copy import tensor_to_numpy_zero_copy

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = tensor_to_numpy_zero_copy(arr, zero_copy=False)

        # Should be a different object
        assert result is not arr
        np.testing.assert_array_equal(result, arr)

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"),
        reason="torch not installed",
    )
    def test_torch_cpu_tensor(self):
        """Test conversion of PyTorch CPU tensor."""
        import torch

        from _bentoml_impl.zero_copy import tensor_to_numpy_zero_copy

        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = tensor_to_numpy_zero_copy(tensor, zero_copy=True)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"),
        reason="torch not installed",
    )
    def test_torch_tensor_with_grad(self):
        """Test conversion of PyTorch tensor with gradients."""
        import torch

        from _bentoml_impl.zero_copy import tensor_to_numpy_zero_copy

        tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = tensor_to_numpy_zero_copy(tensor, zero_copy=True)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"),
        reason="torch not installed",
    )
    def test_torch_non_contiguous_tensor(self):
        """Test conversion of non-contiguous PyTorch tensor."""
        import torch

        from _bentoml_impl.zero_copy import tensor_to_numpy_zero_copy

        tensor = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]]
        ).T  # Transpose makes it non-contiguous
        assert not tensor.is_contiguous()

        result = tensor_to_numpy_zero_copy(tensor, zero_copy=True)

        assert isinstance(result, np.ndarray)
        expected = np.array([[1.0, 3.0], [2.0, 4.0]])
        np.testing.assert_array_almost_equal(result, expected)


class TestNumpyToTensorZeroCopy:
    """Tests for numpy_to_tensor_zero_copy function."""

    def test_numpy_to_numpy(self):
        """Test converting numpy to numpy format."""
        from _bentoml_impl.zero_copy import numpy_to_tensor_zero_copy

        arr = np.array([1.0, 2.0, 3.0])
        result = numpy_to_tensor_zero_copy(arr, "numpy-array", zero_copy=True)

        assert result is arr

    def test_numpy_to_numpy_copy(self):
        """Test converting numpy to numpy format with copy."""
        from _bentoml_impl.zero_copy import numpy_to_tensor_zero_copy

        arr = np.array([1.0, 2.0, 3.0])
        result = numpy_to_tensor_zero_copy(arr, "numpy-array", zero_copy=False)

        assert result is not arr
        np.testing.assert_array_equal(result, arr)

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"),
        reason="torch not installed",
    )
    def test_numpy_to_torch(self):
        """Test converting numpy to torch tensor."""
        import torch

        from _bentoml_impl.zero_copy import numpy_to_tensor_zero_copy

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = numpy_to_tensor_zero_copy(arr, "torch-tensor", zero_copy=True)

        assert isinstance(result, torch.Tensor)
        np.testing.assert_array_almost_equal(result.numpy(), arr)

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"),
        reason="torch not installed",
    )
    def test_numpy_to_torch_non_contiguous(self):
        """Test converting non-contiguous numpy to torch tensor."""
        import torch

        from _bentoml_impl.zero_copy import numpy_to_tensor_zero_copy

        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order="F")
        assert not arr.flags.c_contiguous

        result = numpy_to_tensor_zero_copy(arr, "torch-tensor", zero_copy=True)

        assert isinstance(result, torch.Tensor)
        np.testing.assert_array_almost_equal(result.numpy(), arr)

    def test_invalid_format(self):
        """Test that invalid format raises ValueError."""
        from _bentoml_impl.zero_copy import numpy_to_tensor_zero_copy

        arr = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Unknown target format"):
            numpy_to_tensor_zero_copy(arr, "invalid-format", zero_copy=True)


class TestZeroCopyPickleBuffer:
    """Tests for ZeroCopyPickleBuffer class."""

    def test_serialize_simple_object(self):
        """Test serializing a simple object."""
        from _bentoml_impl.zero_copy import ZeroCopyPickleBuffer

        buffer = ZeroCopyPickleBuffer()
        obj = {"key": "value", "number": 42}
        buffer.serialize(obj, zero_copy=True)

        assert buffer.total_size() > 0
        assert len(buffer.buffer_lengths) >= 1

    def test_serialize_numpy_array(self):
        """Test serializing a numpy array with out-of-band buffers."""
        from _bentoml_impl.zero_copy import ZeroCopyPickleBuffer

        buffer = ZeroCopyPickleBuffer()
        arr = np.random.randn(100, 100)  # Large array for out-of-band
        buffer.serialize(arr, zero_copy=True)

        assert buffer.total_size() > 0

    def test_iter_chunks(self):
        """Test iterating over serialized chunks."""
        from _bentoml_impl.zero_copy import ZeroCopyPickleBuffer

        buffer = ZeroCopyPickleBuffer()
        obj = np.array([1, 2, 3])
        buffer.serialize(obj, zero_copy=True)

        chunks = list(buffer.iter_chunks())
        assert len(chunks) >= 1
        assert sum(len(c) for c in chunks) == buffer.total_size()

    def test_get_metadata(self):
        """Test getting metadata for buffer lengths."""
        from _bentoml_impl.zero_copy import ZeroCopyPickleBuffer

        buffer = ZeroCopyPickleBuffer()
        obj = np.array([1, 2, 3])
        buffer.serialize(obj, zero_copy=True)

        metadata = buffer.get_metadata()
        assert "buffer-lengths" in metadata
        assert len(metadata["buffer-lengths"]) > 0

    def test_release(self):
        """Test releasing buffer references."""
        from _bentoml_impl.zero_copy import ZeroCopyPickleBuffer

        buffer = ZeroCopyPickleBuffer()
        arr = np.random.randn(100, 100)
        buffer.serialize(arr, zero_copy=True)

        buffer.release()
        assert len(buffer._pickle_buffers) == 0
        assert len(buffer.out_of_band_buffers) == 0

    def test_deserialize(self):
        """Test deserializing from buffers."""
        from _bentoml_impl.zero_copy import ZeroCopyPickleBuffer

        # Serialize
        original = np.array([[1, 2, 3], [4, 5, 6]])
        buffer = ZeroCopyPickleBuffer()
        buffer.serialize(original, zero_copy=True)

        # Get chunks and metadata
        chunks = list(buffer.iter_chunks())
        buffer_lengths = list(
            map(int, buffer.get_metadata()["buffer-lengths"].split(","))
        )

        # Deserialize
        result = ZeroCopyPickleBuffer.deserialize(
            chunks, buffer_lengths, zero_copy=True
        )

        np.testing.assert_array_equal(result, original)


class TestPreallocatedBatchBuffer:
    """Tests for batch buffer pre-allocation functions."""

    def test_create_preallocated_batch_buffer(self):
        """Test creating a pre-allocated batch buffer."""
        from _bentoml_impl.zero_copy import create_preallocated_batch_buffer

        buffer = create_preallocated_batch_buffer(
            shape=(10, 10),
            dtype=np.float32,
            batch_size=5,
            batch_dim=0,
        )

        assert buffer.shape == (5, 10, 10)
        assert buffer.dtype == np.float32

    def test_create_preallocated_batch_buffer_different_dim(self):
        """Test creating a pre-allocated batch buffer with different batch dim."""
        from _bentoml_impl.zero_copy import create_preallocated_batch_buffer

        buffer = create_preallocated_batch_buffer(
            shape=(3, 4),
            dtype=np.float64,
            batch_size=2,
            batch_dim=1,
        )

        assert buffer.shape == (3, 2, 4)
        assert buffer.dtype == np.float64

    def test_fill_batch_buffer_inplace(self):
        """Test filling a batch buffer in-place."""
        from _bentoml_impl.zero_copy import create_preallocated_batch_buffer
        from _bentoml_impl.zero_copy import fill_batch_buffer_inplace

        # Create items
        items = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
            np.array([[9, 10], [11, 12]]),
        ]

        # Create buffer
        buffer = create_preallocated_batch_buffer(
            shape=(2, 2),
            dtype=np.int64,
            batch_size=6,  # 2 + 2 + 2
            batch_dim=0,
        )

        # Fill buffer
        fill_batch_buffer_inplace(buffer, items, batch_dim=0)

        expected = np.array(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
                [9, 10],
                [11, 12],
            ]
        )
        np.testing.assert_array_equal(buffer, expected)


class TestArrowFunctions:
    """Tests for Arrow-related functions."""

    @pytest.mark.skipif(
        not pytest.importorskip("pyarrow", reason="pyarrow not installed"),
        reason="pyarrow not installed",
    )
    def test_get_arrow_dtype(self):
        """Test getting Arrow dtype from numpy dtype."""
        import pyarrow as pa

        from _bentoml_impl.zero_copy import get_arrow_dtype

        arrow_type = get_arrow_dtype("float32")
        assert arrow_type == pa.float32()

        arrow_type = get_arrow_dtype("int64")
        assert arrow_type == pa.int64()

    @pytest.mark.skipif(
        not pytest.importorskip("pyarrow", reason="pyarrow not installed"),
        reason="pyarrow not installed",
    )
    def test_tensor_to_arrow_buffer(self):
        """Test converting tensor to Arrow buffer."""
        import pyarrow as pa

        from _bentoml_impl.zero_copy import tensor_to_arrow_buffer

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        buffer = tensor_to_arrow_buffer(arr, zero_copy=True)

        assert isinstance(buffer, pa.Buffer)
        assert len(buffer) == arr.nbytes

    @pytest.mark.skipif(
        not pytest.importorskip("pyarrow", reason="pyarrow not installed"),
        reason="pyarrow not installed",
    )
    def test_arrow_buffer_to_numpy(self):
        """Test converting Arrow buffer back to numpy."""
        from _bentoml_impl.zero_copy import arrow_buffer_to_numpy
        from _bentoml_impl.zero_copy import tensor_to_arrow_buffer

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        buffer = tensor_to_arrow_buffer(arr, zero_copy=True)

        result = arrow_buffer_to_numpy(buffer, np.float64, (3,), zero_copy=True)

        np.testing.assert_array_equal(result, arr)


class TestArrowTensorSerializer:
    """Tests for ArrowTensorSerializer class."""

    @pytest.mark.skipif(
        not pytest.importorskip("pyarrow", reason="pyarrow not installed"),
        reason="pyarrow not installed",
    )
    def test_serialize_with_shape(self):
        """Test serializing a tensor with shape preservation."""
        from _bentoml_impl.zero_copy import ArrowTensorSerializer

        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        serialized = ArrowTensorSerializer.serialize(
            arr, preserve_shape=True, zero_copy=True
        )

        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

    @pytest.mark.skipif(
        not pytest.importorskip("pyarrow", reason="pyarrow not installed"),
        reason="pyarrow not installed",
    )
    def test_serialize_without_shape(self):
        """Test serializing a tensor without shape preservation."""
        from _bentoml_impl.zero_copy import ArrowTensorSerializer

        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        serialized = ArrowTensorSerializer.serialize(
            arr, preserve_shape=False, zero_copy=True
        )

        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

    @pytest.mark.skipif(
        not pytest.importorskip("pyarrow", reason="pyarrow not installed"),
        reason="pyarrow not installed",
    )
    def test_roundtrip_with_shape(self):
        """Test roundtrip serialization with shape preservation."""
        from _bentoml_impl.zero_copy import ArrowTensorSerializer

        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        serialized = ArrowTensorSerializer.serialize(
            arr, preserve_shape=True, zero_copy=True
        )
        result = ArrowTensorSerializer.deserialize(
            serialized, has_shape=True, zero_copy=True
        )

        np.testing.assert_array_equal(result, arr)

    @pytest.mark.skipif(
        not pytest.importorskip("pyarrow", reason="pyarrow not installed"),
        reason="pyarrow not installed",
    )
    def test_roundtrip_without_shape(self):
        """Test roundtrip serialization without shape preservation."""
        from _bentoml_impl.zero_copy import ArrowTensorSerializer

        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        serialized = ArrowTensorSerializer.serialize(
            arr, preserve_shape=False, zero_copy=True
        )
        result = ArrowTensorSerializer.deserialize(
            serialized, has_shape=False, shape=(2, 2), zero_copy=True
        )

        np.testing.assert_array_equal(result, arr)


class TestFrombufferZeroCopy:
    """Tests for frombuffer_zero_copy function."""

    def test_frombuffer_zero_copy_enabled(self):
        """Test creating array from buffer with zero_copy enabled."""
        from _bentoml_impl.zero_copy import frombuffer_zero_copy

        data = np.array([1.0, 2.0, 3.0], dtype=np.float64).tobytes()
        result = frombuffer_zero_copy(data, np.float64, zero_copy=True)

        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_frombuffer_zero_copy_disabled(self):
        """Test creating array from buffer with zero_copy disabled."""
        from _bentoml_impl.zero_copy import frombuffer_zero_copy

        data = np.array([1.0, 2.0, 3.0], dtype=np.float64).tobytes()
        result = frombuffer_zero_copy(data, np.float64, zero_copy=False)

        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
        # Result should own its data (copy was made)
        assert result.flags.owndata

    def test_frombuffer_with_shape(self):
        """Test creating shaped array from buffer."""
        from _bentoml_impl.zero_copy import frombuffer_zero_copy

        arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
        data = arr.tobytes()
        result = frombuffer_zero_copy(data, np.int32, shape=(3, 2), zero_copy=True)

        np.testing.assert_array_equal(result, arr)

    def test_frombuffer_from_memoryview(self):
        """Test creating array from memoryview."""
        from _bentoml_impl.zero_copy import frombuffer_zero_copy

        data = np.array([1.0, 2.0, 3.0], dtype=np.float64).tobytes()
        mv = memoryview(data)
        result = frombuffer_zero_copy(mv, np.float64, zero_copy=True)

        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
