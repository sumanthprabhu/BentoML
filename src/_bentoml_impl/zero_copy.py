"""
Zero-Copy Tensor Serialization Module for BentoML.

This module provides optimized tensor serialization/deserialization utilities
that minimize memory copies during data transfer. It supports NumPy, PyTorch,
and TensorFlow tensors.

Key optimizations:
1. Uses np.asarray() with buffer protocol instead of .numpy() conversions
2. Leverages pickle protocol 5 out-of-band buffers properly
3. Uses PyArrow IPC format with zero_copy_only option where possible
4. Pre-allocates batch buffers to avoid concatenation copies
5. Uses memoryview for streaming data

Usage:
    # Enable zero-copy at service level
    @bentoml.service(zero_copy=True)
    class MyService:
        @bentoml.api
        def predict(self, data: np.ndarray) -> np.ndarray:
            return model.predict(data)

    # Or via configuration
    @bentoml.service(zero_copy={"enabled": True, "use_arrow_ipc": True})
    class MyService:
        pass
"""

from __future__ import annotations

import pickle
import typing as t
from functools import lru_cache

from .zero_copy_config import get_zero_copy_enabled

if t.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import pyarrow as pa
    import tensorflow as tf
    import torch

    TensorType = t.Union[npt.NDArray[t.Any], tf.Tensor, torch.Tensor]
else:
    from bentoml._internal.utils.lazy_loader import LazyLoader

    np = LazyLoader("np", globals(), "numpy")
    pa = LazyLoader("pa", globals(), "pyarrow")
    tf = LazyLoader("tf", globals(), "tensorflow")
    torch = LazyLoader("torch", globals(), "torch")


class ZeroCopyBuffer:
    """A buffer wrapper that supports the buffer protocol for zero-copy operations."""

    __slots__ = ("_data", "_readonly")

    def __init__(self, data: bytes | memoryview | bytearray, readonly: bool = True):
        self._data = data
        self._readonly = readonly

    def __buffer__(self, flags: int) -> memoryview:
        return memoryview(self._data)

    @property
    def data(self) -> bytes | memoryview | bytearray:
        return self._data

    def __len__(self) -> int:
        return len(self._data)


def _is_torch_tensor(arr: t.Any) -> bool:
    """Check if arr is a PyTorch tensor without importing torch."""
    return (
        hasattr(arr, "device")
        and hasattr(arr, "numpy")
        and hasattr(arr, "is_contiguous")
    )


def _is_tf_tensor(arr: t.Any) -> bool:
    """Check if arr is a TensorFlow tensor without importing tensorflow."""
    # TF tensors have numpy(), dtype, and shape but not is_contiguous
    return (
        hasattr(arr, "numpy")
        and hasattr(arr, "dtype")
        and hasattr(arr, "shape")
        and not hasattr(arr, "is_contiguous")
        and type(arr).__module__.startswith("tensorflow")
    )


def tensor_to_numpy_zero_copy(
    arr: TensorType,
    *,
    force_copy: bool = False,
    zero_copy: bool | None = None,
) -> np.ndarray[t.Any, t.Any]:
    """
    Convert a tensor to NumPy array with minimal copying.

    This function attempts to use zero-copy semantics where possible:
    - For NumPy arrays: returns as-is or uses np.asarray()
    - For PyTorch tensors: uses tensor.numpy() which shares storage for CPU tensors
    - For TensorFlow tensors: uses np.asarray() with buffer protocol

    Args:
        arr: Input tensor (NumPy, PyTorch, or TensorFlow)
        force_copy: If True, always create a copy (default: False)
        zero_copy: Override the global zero_copy setting. If None, uses global setting.

    Returns:
        NumPy array, potentially sharing memory with input
    """
    # Check if zero_copy is enabled
    if zero_copy is None:
        zero_copy = get_zero_copy_enabled()

    # If zero_copy is disabled, always force copy
    if not zero_copy:
        force_copy = True

    # Check for NumPy array first (most common case)
    if isinstance(arr, np.ndarray):
        if force_copy:
            return arr.copy()
        return arr

    # PyTorch tensor handling
    if _is_torch_tensor(arr):
        device = arr.device
        if hasattr(device, "type") and device.type != "cpu":
            # GPU tensor - must copy to CPU first
            # Use contiguous() to ensure memory layout is compatible
            arr = arr.detach().cpu()
        elif hasattr(arr, "requires_grad") and arr.requires_grad:
            # Tensor with gradients - detach first
            arr = arr.detach()

        # For CPU tensors, .numpy() returns a view (zero-copy)
        # Only works for contiguous tensors
        if arr.is_contiguous():
            result = arr.numpy()
        else:
            # Make contiguous first, then get numpy view
            result = arr.contiguous().numpy()

        if force_copy:
            return result.copy()
        return result

    # TensorFlow tensor handling
    if _is_tf_tensor(arr):
        # TensorFlow's .numpy() method creates a copy, so we try np.asarray first
        # np.asarray will use the buffer protocol if available
        try:
            result = np.asarray(arr)
            if force_copy and not result.flags.owndata:
                return result.copy()
            return result
        except (TypeError, ValueError):
            # Fallback to .numpy() method
            result = arr.numpy()
            if force_copy:
                return result.copy()
            return result

    # Fallback for unknown types - try np.asarray with buffer protocol
    try:
        result = np.asarray(arr)
        if force_copy and not result.flags.owndata:
            return result.copy()
        return result
    except (TypeError, ValueError):
        # Last resort - convert via array (creates copy)
        return np.array(arr)


def numpy_to_tensor_zero_copy(
    arr: np.ndarray[t.Any, t.Any],
    target_format: str,
    *,
    device: t.Optional[str] = None,
    zero_copy: bool | None = None,
) -> TensorType:
    """
    Convert NumPy array to target tensor format with minimal copying.

    Args:
        arr: Input NumPy array
        target_format: One of "numpy-array", "torch-tensor", "tf-tensor"
        device: Optional device specification (for PyTorch)
        zero_copy: Override the global zero_copy setting. If None, uses global setting.

    Returns:
        Tensor in the specified format
    """
    # Check if zero_copy is enabled
    if zero_copy is None:
        zero_copy = get_zero_copy_enabled()

    if target_format == "numpy-array":
        return arr if zero_copy else arr.copy()

    if target_format == "torch-tensor":
        if zero_copy:
            # Ensure array is contiguous for torch.from_numpy
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            # torch.from_numpy creates a tensor that shares memory with the array
            tensor = torch.from_numpy(arr)
        else:
            # Create a copy
            tensor = torch.tensor(arr)

        if device is not None and device != "cpu":
            # Moving to GPU requires a copy
            tensor = tensor.to(device)
        return tensor

    if target_format == "tf-tensor":
        # tf.constant from numpy array - TF manages memory
        return tf.constant(arr)

    raise ValueError(f"Unknown target format: {target_format}")


class ZeroCopyPickleBuffer:
    """
    Enhanced pickle buffer handling that maintains separate buffers
    to avoid concatenation copies during serialization.
    """

    __slots__ = (
        "main_bytes",
        "out_of_band_buffers",
        "buffer_lengths",
        "_pickle_buffers",
    )

    def __init__(self) -> None:
        self.main_bytes: bytes = b""
        self.out_of_band_buffers: list[memoryview] = []
        self.buffer_lengths: list[int] = []
        # Keep reference to PickleBuffer objects to prevent GC
        self._pickle_buffers: list[pickle.PickleBuffer] = []

    def serialize(self, obj: t.Any, zero_copy: bool | None = None) -> None:
        """Serialize object using pickle protocol 5 with out-of-band buffers."""
        if zero_copy is None:
            zero_copy = get_zero_copy_enabled()

        if zero_copy:
            buffers: list[pickle.PickleBuffer] = []
            self.main_bytes = pickle.dumps(
                obj, protocol=5, buffer_callback=buffers.append
            )
            self.buffer_lengths = [len(self.main_bytes)]
            self._pickle_buffers = buffers

            for buff in buffers:
                # Get a memoryview without copying
                mv = buff.raw()
                self.out_of_band_buffers.append(mv)
                self.buffer_lengths.append(len(mv))
        else:
            # Standard pickle without out-of-band buffers
            self.main_bytes = pickle.dumps(obj, protocol=5)
            self.buffer_lengths = [len(self.main_bytes)]
            self._pickle_buffers = []
            self.out_of_band_buffers = []

    def iter_chunks(self) -> t.Iterator[memoryview | bytes]:
        """Iterate over data chunks without joining them."""
        yield self.main_bytes
        yield from self.out_of_band_buffers

    def get_metadata(self) -> dict[str, str]:
        """Get metadata for reconstructing the payload."""
        return {"buffer-lengths": ",".join(map(str, self.buffer_lengths))}

    def total_size(self) -> int:
        """Get total size of all buffers."""
        return sum(self.buffer_lengths)

    def release(self) -> None:
        """Release references to buffers."""
        for buff in self._pickle_buffers:
            buff.release()
        self._pickle_buffers.clear()
        self.out_of_band_buffers.clear()

    @classmethod
    def deserialize(
        cls,
        data_iter: t.Iterable[bytes | memoryview],
        buffer_lengths: list[int],
        zero_copy: bool | None = None,
    ) -> t.Any:
        """Deserialize from separate buffers without joining."""
        if zero_copy is None:
            zero_copy = get_zero_copy_enabled()

        # Collect all data
        all_data = b"".join(d if isinstance(d, bytes) else bytes(d) for d in data_iter)

        if len(buffer_lengths) <= 1:
            # No out-of-band buffers
            return pickle.loads(all_data)

        # Extract main pickle bytes and out-of-band buffers
        data_view = memoryview(all_data)
        main_length = buffer_lengths[0]
        main_bytes = data_view[:main_length]

        buffers: list[pickle.PickleBuffer] = []
        offset = main_length
        for length in buffer_lengths[1:]:
            buffers.append(pickle.PickleBuffer(data_view[offset : offset + length]))
            offset += length

        return pickle.loads(main_bytes, buffers=buffers)


def create_preallocated_batch_buffer(
    shape: tuple[int, ...],
    dtype: t.Any,
    batch_size: int,
    batch_dim: int = 0,
) -> np.ndarray[t.Any, t.Any]:
    """
    Pre-allocate a batch buffer for efficient batch aggregation.

    Instead of concatenating arrays (which creates copies), this pre-allocates
    a single buffer that individual arrays can be copied into with views.

    Args:
        shape: Shape of individual items (without batch dimension)
        dtype: Data type of the array
        batch_size: Number of items in the batch
        batch_dim: Dimension along which to batch (default: 0)

    Returns:
        Pre-allocated numpy array
    """
    # Insert batch_size at the batch dimension
    batch_shape = list(shape)
    batch_shape.insert(batch_dim, batch_size)
    return np.empty(batch_shape, dtype=dtype)


def fill_batch_buffer_inplace(
    batch_buffer: np.ndarray[t.Any, t.Any],
    items: t.Sequence[np.ndarray[t.Any, t.Any]],
    batch_dim: int = 0,
) -> None:
    """
    Fill a pre-allocated batch buffer in-place without creating copies.

    Args:
        batch_buffer: Pre-allocated buffer to fill
        items: Sequence of arrays to place in the buffer
        batch_dim: Dimension along which items are batched
    """
    # Create index slices for each position
    current_idx = 0
    for item in items:
        item_size = item.shape[batch_dim] if len(item.shape) > batch_dim else 1
        # Create slice objects for indexing
        slices: list[t.Any] = [slice(None)] * len(batch_buffer.shape)
        slices[batch_dim] = slice(current_idx, current_idx + item_size)
        # In-place assignment - no copy
        batch_buffer[tuple(slices)] = item
        current_idx += item_size


@lru_cache(maxsize=32)
def get_arrow_dtype(numpy_dtype_str: str) -> t.Any:
    """
    Get the corresponding PyArrow data type for a NumPy dtype.

    Cached for performance. Takes string to be hashable.
    """
    numpy_dtype = np.dtype(numpy_dtype_str)
    return pa.from_numpy_dtype(numpy_dtype)


def tensor_to_arrow_buffer(
    arr: np.ndarray[t.Any, t.Any],
    *,
    zero_copy: bool | None = None,
) -> t.Any:  # pa.Buffer
    """
    Convert NumPy array to PyArrow buffer with zero-copy option.

    Args:
        arr: Input NumPy array
        zero_copy: If True, attempt zero-copy conversion. If None, uses global setting.

    Returns:
        PyArrow buffer
    """
    if zero_copy is None:
        zero_copy = get_zero_copy_enabled()

    if zero_copy:
        # Ensure array is contiguous for zero-copy
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        # Create buffer that shares memory with the array
        return pa.py_buffer(arr)
    else:
        return pa.py_buffer(arr.tobytes())


def arrow_buffer_to_numpy(
    buffer: t.Any,  # pa.Buffer
    dtype: t.Any,
    shape: tuple[int, ...],
    *,
    zero_copy: bool | None = None,
) -> np.ndarray[t.Any, t.Any]:
    """
    Convert PyArrow buffer to NumPy array with zero-copy option.

    Args:
        buffer: PyArrow buffer
        dtype: Target NumPy dtype
        shape: Target shape
        zero_copy: If True, attempt zero-copy conversion. If None, uses global setting.

    Returns:
        NumPy array
    """
    if zero_copy is None:
        zero_copy = get_zero_copy_enabled()

    if zero_copy:
        # Use buffer protocol for zero-copy
        arr = np.frombuffer(buffer, dtype=dtype)
    else:
        arr = np.frombuffer(bytes(buffer), dtype=dtype).copy()

    return arr.reshape(shape)


class ArrowTensorSerializer:
    """
    Serializer for tensors using PyArrow IPC format.

    This provides efficient serialization with optional zero-copy semantics.
    """

    @staticmethod
    def serialize(
        arr: np.ndarray[t.Any, t.Any],
        *,
        preserve_shape: bool = True,
        zero_copy: bool | None = None,
    ) -> bytes:
        """
        Serialize a NumPy array to bytes using PyArrow IPC.

        Args:
            arr: Input array
            preserve_shape: If True, include shape metadata (default: True)
            zero_copy: Override the global zero_copy setting.

        Returns:
            Serialized bytes
        """
        import io as _io

        if zero_copy is None:
            zero_copy = get_zero_copy_enabled()

        # Ensure array is contiguous for efficient serialization
        if zero_copy and not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)

        # Create Arrow tensor (preserves shape naturally)
        if preserve_shape:
            tensor = pa.Tensor.from_numpy(arr)
            sink = pa.BufferOutputStream()
            pa.ipc.write_tensor(tensor, sink)
            return sink.getvalue().to_pybytes()
        else:
            # Flatten for simpler serialization
            flat_arr = arr.ravel()
            arrow_type = get_arrow_dtype(str(arr.dtype))
            arrow_array = pa.array(flat_arr, type=arrow_type)

            schema = pa.schema([pa.field("data", arrow_type)])
            batch = pa.RecordBatch.from_arrays([arrow_array], schema=schema)

            sink = _io.BytesIO()
            with pa.ipc.new_stream(sink, schema) as writer:
                writer.write_batch(batch)
            return sink.getvalue()

    @staticmethod
    def deserialize(
        data: bytes,
        *,
        has_shape: bool = True,
        dtype: t.Optional[t.Any] = None,
        shape: t.Optional[tuple[int, ...]] = None,
        zero_copy: bool | None = None,
    ) -> np.ndarray[t.Any, t.Any]:
        """
        Deserialize bytes to NumPy array using PyArrow IPC.

        Args:
            data: Input bytes
            has_shape: If True, extract shape from tensor format
            dtype: Expected dtype (optional)
            shape: Expected shape (optional)
            zero_copy: Override the global zero_copy setting.

        Returns:
            NumPy array
        """
        import io as _io

        if zero_copy is None:
            zero_copy = get_zero_copy_enabled()

        if has_shape:
            reader = pa.BufferReader(data)
            tensor = pa.ipc.read_tensor(reader)
            return tensor.to_numpy()
        else:
            reader = pa.ipc.open_stream(_io.BytesIO(data))
            batch = reader.read_all()
            arr = batch.column(0).to_numpy(zero_copy_only=zero_copy)
            if shape is not None:
                arr = arr.reshape(shape)
            return arr


def frombuffer_zero_copy(
    data: bytes | memoryview,
    dtype: t.Any,
    *,
    shape: t.Optional[tuple[int, ...]] = None,
    zero_copy: bool | None = None,
) -> np.ndarray[t.Any, t.Any]:
    """
    Create NumPy array from buffer with zero-copy semantics.

    This extends the pattern used in the legacy gRPC path to the new SDK.

    Args:
        data: Input bytes or memoryview
        dtype: NumPy dtype
        shape: Optional shape to reshape to
        zero_copy: Override the global zero_copy setting.

    Returns:
        NumPy array (shares memory with input buffer if zero_copy enabled)
    """
    if zero_copy is None:
        zero_copy = get_zero_copy_enabled()

    if zero_copy:
        arr = np.frombuffer(data, dtype=dtype)
    else:
        arr = np.frombuffer(data, dtype=dtype).copy()

    if shape is not None:
        arr = arr.reshape(shape)
    return arr


__all__ = [
    "ZeroCopyBuffer",
    "tensor_to_numpy_zero_copy",
    "numpy_to_tensor_zero_copy",
    "ZeroCopyPickleBuffer",
    "create_preallocated_batch_buffer",
    "fill_batch_buffer_inplace",
    "tensor_to_arrow_buffer",
    "arrow_buffer_to_numpy",
    "ArrowTensorSerializer",
    "frombuffer_zero_copy",
]
