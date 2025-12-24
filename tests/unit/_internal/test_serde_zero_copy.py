"""
Tests for zero-copy functionality in the serde module.

This module tests the Payload and PickleSerde classes with zero-copy optimizations.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest


class TestPayload:
    """Tests for the Payload class."""

    def test_payload_with_bytes(self):
        """Test creating Payload with bytes data."""
        from _bentoml_impl.serde import Payload

        data = [b"hello", b" ", b"world"]
        payload = Payload(data)

        assert payload.total_bytes() == 11
        assert list(payload.iter_bytes()) == data

    def test_payload_with_memoryview(self):
        """Test creating Payload with memoryview data."""
        from _bentoml_impl.serde import Payload

        data = b"hello world"
        mv = memoryview(data)
        payload = Payload([mv])

        assert payload.total_bytes() == 11

    def test_payload_metadata(self):
        """Test Payload with metadata."""
        from _bentoml_impl.serde import Payload

        data = [b"test"]
        metadata = {"content-type": "application/json", "custom": "value"}
        payload = Payload(data, metadata)

        assert payload.metadata["content-type"] == "application/json"
        assert payload.metadata["custom"] == "value"

    def test_payload_headers(self):
        """Test Payload headers property."""
        from _bentoml_impl.serde import Payload

        data = [b"test data"]  # 9 bytes
        metadata = {"x-custom": "header"}
        payload = Payload(data, metadata)

        headers = payload.headers
        assert headers["content-length"] == "9"
        assert headers["x-custom"] == "header"

    @pytest.mark.asyncio
    async def test_payload_aiter_bytes(self):
        """Test async iteration over Payload bytes."""
        from _bentoml_impl.serde import Payload

        data = [b"chunk1", b"chunk2", b"chunk3"]
        payload = Payload(data)

        chunks = []
        async for chunk in payload.aiter_bytes():
            chunks.append(chunk)

        assert chunks == data


class TestPickleSerde:
    """Tests for PickleSerde with zero-copy support."""

    def test_serialize_simple_object(self):
        """Test serializing a simple object."""
        from _bentoml_impl.serde import Payload
        from _bentoml_impl.serde import PickleSerde

        serde = PickleSerde()
        obj = {"key": "value", "number": 42}

        payload = serde.serialize_value(obj)

        assert isinstance(payload, Payload)
        assert payload.total_bytes() > 0

    def test_deserialize_simple_object(self):
        """Test deserializing a simple object."""
        from _bentoml_impl.serde import PickleSerde

        serde = PickleSerde()
        obj = {"key": "value", "number": 42}

        payload = serde.serialize_value(obj)
        result = serde.deserialize_value(payload)

        assert result == obj

    def test_serialize_numpy_array(self):
        """Test serializing a numpy array with out-of-band buffers."""
        from _bentoml_impl.serde import PickleSerde

        serde = PickleSerde()
        arr = np.random.randn(100, 100)

        payload = serde.serialize_value(arr)

        assert payload.total_bytes() > 0
        assert "buffer-lengths" in payload.metadata

    def test_deserialize_numpy_array(self):
        """Test deserializing a numpy array."""
        from _bentoml_impl.serde import PickleSerde

        serde = PickleSerde()
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

        payload = serde.serialize_value(arr)
        result = serde.deserialize_value(payload)

        np.testing.assert_array_equal(result, arr)

    def test_roundtrip_complex_object(self):
        """Test roundtrip of complex object with nested arrays."""
        from _bentoml_impl.serde import PickleSerde

        serde = PickleSerde()
        obj = {
            "array1": np.array([1, 2, 3]),
            "array2": np.random.randn(50, 50),
            "nested": {
                "array3": np.array([[1.0, 2.0], [3.0, 4.0]]),
            },
        }

        payload = serde.serialize_value(obj)
        result = serde.deserialize_value(payload)

        np.testing.assert_array_equal(result["array1"], obj["array1"])
        np.testing.assert_array_almost_equal(result["array2"], obj["array2"])
        np.testing.assert_array_almost_equal(
            result["nested"]["array3"], obj["nested"]["array3"]
        )

    def test_buffer_lengths_metadata(self):
        """Test that buffer lengths are properly stored in metadata."""
        from _bentoml_impl.serde import PickleSerde

        serde = PickleSerde()
        arr = np.random.randn(100, 100)

        payload = serde.serialize_value(arr)

        # Parse buffer lengths from metadata
        lengths_str = payload.metadata.get("buffer-lengths", "")
        lengths = [int(x) for x in lengths_str.split(",") if x]

        # Should have at least one length (the main pickle bytes)
        assert len(lengths) >= 1
        # Total of lengths should equal total bytes
        assert sum(lengths) == payload.total_bytes()

    def test_deserialize_without_buffer_lengths(self):
        """Test deserializing payload without buffer-lengths metadata."""
        from _bentoml_impl.serde import Payload
        from _bentoml_impl.serde import PickleSerde

        serde = PickleSerde()
        obj = "simple string"

        # Create payload without buffer-lengths
        data = pickle.dumps(obj)
        payload = Payload([data], metadata={})

        result = serde.deserialize_value(payload)
        assert result == obj


class TestPickleSerdeModel:
    """Tests for PickleSerde with IODescriptor models."""

    def test_serialize_model(self):
        """Test serializing an IODescriptor model."""
        from pydantic import BaseModel

        from _bentoml_impl.serde import PickleSerde

        class TestModel(BaseModel):
            value: int
            name: str

        serde = PickleSerde()
        model = TestModel(value=42, name="test")

        payload = serde.serialize_model(model)

        assert payload.total_bytes() > 0

    def test_deserialize_model(self):
        """Test deserializing an IODescriptor model."""
        from pydantic import BaseModel

        from _bentoml_impl.serde import PickleSerde

        class TestModel(BaseModel):
            value: int
            name: str

        serde = PickleSerde()
        model = TestModel(value=42, name="test")

        payload = serde.serialize_model(model)
        result = serde.deserialize_model(payload, TestModel)

        assert result.value == 42
        assert result.name == "test"

    def test_deserialize_model_from_dict(self):
        """Test deserializing model when payload contains dict instead of model."""
        from pydantic import BaseModel

        from _bentoml_impl.serde import PickleSerde

        class TestModel(BaseModel):
            value: int
            name: str

        serde = PickleSerde()
        data = {"value": 42, "name": "test"}

        payload = serde.serialize_value(data)
        result = serde.deserialize_model(payload, TestModel)

        assert result.value == 42
        assert result.name == "test"


class TestSerializationInfo:
    """Tests for SerializationInfo class."""

    def test_json_mode(self):
        """Test SerializationInfo in JSON mode."""
        from _bentoml_impl.serde import SerializationInfo

        info = SerializationInfo(mode="json")

        assert info.mode_is_json() is True

    def test_python_mode(self):
        """Test SerializationInfo in Python mode."""
        from _bentoml_impl.serde import SerializationInfo

        info = SerializationInfo(mode="python")

        assert info.mode_is_json() is False

    def test_other_mode(self):
        """Test SerializationInfo with other mode."""
        from _bentoml_impl.serde import SerializationInfo

        info = SerializationInfo(mode="pickle")

        assert info.mode_is_json() is False


class TestJSONSerdeWithTensors:
    """Tests for JSONSerde with tensor data."""

    def test_serialize_tensor_schema(self):
        """Test encoding tensor data through JSONSerde."""
        from _bentoml_impl.serde import JSONSerde

        serde = JSONSerde()

        schema = {
            "type": "tensor",
            "format": "numpy-array",
            "dtype": "float32",
        }

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        payload = serde.serialize(arr, schema)

        # JSON serialization should work
        assert payload.total_bytes() > 0

    def test_deserialize_tensor_schema(self):
        """Test decoding tensor data through JSONSerde."""
        from _bentoml_impl.serde import JSONSerde

        serde = JSONSerde()

        schema = {
            "type": "tensor",
            "format": "numpy-array",
            "dtype": "float32",
        }

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        payload = serde.serialize(arr, schema)
        result = serde.deserialize(payload, schema)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, arr)


class TestAllSerde:
    """Tests for the ALL_SERDE registry."""

    def test_all_serde_contains_pickle(self):
        """Test that ALL_SERDE contains PickleSerde."""
        from _bentoml_impl.serde import ALL_SERDE
        from _bentoml_impl.serde import PickleSerde

        assert "application/vnd.bentoml+pickle" in ALL_SERDE
        assert ALL_SERDE["application/vnd.bentoml+pickle"] is PickleSerde

    def test_all_serde_contains_json(self):
        """Test that ALL_SERDE contains JSONSerde."""
        from _bentoml_impl.serde import ALL_SERDE
        from _bentoml_impl.serde import JSONSerde

        assert "application/json" in ALL_SERDE
        assert ALL_SERDE["application/json"] is JSONSerde

    def test_all_serde_contains_multipart(self):
        """Test that ALL_SERDE contains MultipartSerde."""
        from _bentoml_impl.serde import ALL_SERDE
        from _bentoml_impl.serde import MultipartSerde

        assert "multipart/form-data" in ALL_SERDE
        assert ALL_SERDE["multipart/form-data"] is MultipartSerde

        # Also check the special case
        assert "application/x-www-form-urlencoded" in ALL_SERDE
        assert ALL_SERDE["application/x-www-form-urlencoded"] is MultipartSerde
