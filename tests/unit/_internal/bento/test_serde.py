"""Unit tests for Serde classes in _bentoml_impl/serde.py."""

from __future__ import annotations

import json

import numpy as np
from pydantic import BaseModel

from _bentoml_impl.serde import ALL_SERDE
from _bentoml_impl.serde import JSONSerde
from _bentoml_impl.serde import Payload
from _bentoml_impl.serde import PickleSerde
from _bentoml_impl.serde import ZeroCopySerde


class TestPayload:
    """Tests for Payload class."""

    def test_payload_creation(self):
        """Test basic payload creation."""
        data = [b"hello", b"world"]
        payload = Payload(data)
        assert list(payload.data) == data

    def test_payload_with_metadata(self):
        """Test payload with metadata."""
        data = [b"test"]
        metadata = {"key": "value"}
        payload = Payload(data, metadata)
        assert payload.metadata == metadata

    def test_total_bytes(self):
        """Test total_bytes calculation."""
        data = [b"hello", b"world"]
        payload = Payload(data)
        assert payload.total_bytes() == 10

    def test_iter_bytes(self):
        """Test iter_bytes iterator."""
        data = [b"hello", b"world"]
        payload = Payload(data)
        result = list(payload.iter_bytes())
        assert result == [b"hello", b"world"]

    def test_headers(self):
        """Test headers property."""
        data = [b"hello"]
        metadata = {"custom": "header"}
        payload = Payload(data, metadata)
        headers = payload.headers
        assert headers["content-length"] == "5"
        assert headers["custom"] == "header"


class TestJSONSerde:
    """Tests for JSONSerde class."""

    def test_media_type(self):
        """Test media type."""
        serde = JSONSerde()
        assert serde.media_type == "application/json"

    def test_serialize_value(self):
        """Test value serialization."""
        serde = JSONSerde()
        data = {"key": "value", "number": 42}
        payload = serde.serialize_value(data)
        result = json.loads(b"".join(payload.data))
        assert result == data

    def test_deserialize_value(self):
        """Test value deserialization."""
        serde = JSONSerde()
        data = {"key": "value"}
        json_bytes = json.dumps(data).encode()
        payload = Payload((json_bytes,))
        result = serde.deserialize_value(payload)
        assert result == data

    def test_serialize_with_tensor_schema(self):
        """Test serialization with tensor schema."""
        serde = JSONSerde()
        arr = np.array([1.0, 2.0, 3.0])
        schema = {"type": "tensor", "format": "numpy-array"}
        payload = serde.serialize(arr, schema)
        result = json.loads(b"".join(payload.data))
        assert result == [1.0, 2.0, 3.0]

    def test_deserialize_with_tensor_schema(self):
        """Test deserialization with tensor schema."""
        serde = JSONSerde()
        json_bytes = json.dumps([1.0, 2.0, 3.0]).encode()
        payload = Payload((json_bytes,))
        schema = {"type": "tensor", "format": "numpy-array"}
        result = serde.deserialize(payload, schema)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))


class TestPickleSerde:
    """Tests for PickleSerde class."""

    def test_media_type(self):
        """Test media type."""
        serde = PickleSerde()
        assert serde.media_type == "application/vnd.bentoml+pickle"

    def test_serialize_value(self):
        """Test value serialization."""
        serde = PickleSerde()
        data = {"key": "value", "number": 42}
        payload = serde.serialize_value(data)
        result = serde.deserialize_value(payload)
        assert result == data

    def test_numpy_array_roundtrip(self):
        """Test numpy array serialization roundtrip."""
        serde = PickleSerde()
        arr = np.random.rand(10, 10)
        payload = serde.serialize_value(arr)
        result = serde.deserialize_value(payload)
        np.testing.assert_array_equal(arr, result)

    def test_buffer_lengths_metadata(self):
        """Test buffer-lengths metadata is set."""
        serde = PickleSerde()
        arr = np.random.rand(100, 100)
        payload = serde.serialize_value(arr)
        assert "buffer-lengths" in payload.metadata

    def test_complex_structure(self):
        """Test serialization of complex nested structures."""
        serde = PickleSerde()
        data = {
            "arrays": [np.array([1, 2, 3]), np.array([4, 5, 6])],
            "metadata": {"name": "test"},
            "nested": {"arr": np.random.rand(5, 5)},
        }
        payload = serde.serialize_value(data)
        result = serde.deserialize_value(payload)
        np.testing.assert_array_equal(data["arrays"][0], result["arrays"][0])
        np.testing.assert_array_equal(data["nested"]["arr"], result["nested"]["arr"])


class TestZeroCopySerde:
    """Tests for ZeroCopySerde class."""

    def test_media_type(self):
        """Test media type."""
        serde = ZeroCopySerde()
        assert serde.media_type == "application/vnd.bentoml+zerocopy"

    def test_serialize_value(self):
        """Test value serialization."""
        serde = ZeroCopySerde()
        data = {"key": "value", "number": 42}
        payload = serde.serialize_value(data)
        result = serde.deserialize_value(payload)
        assert result == data

    def test_zero_copy_metadata(self):
        """Test zero-copy metadata is set."""
        serde = ZeroCopySerde()
        arr = np.random.rand(100, 100)
        payload = serde.serialize_value(arr)
        assert payload.metadata.get("zero-copy") == "true"

    def test_numpy_array_roundtrip(self):
        """Test numpy array serialization roundtrip."""
        serde = ZeroCopySerde()
        arr = np.random.rand(100, 100)
        payload = serde.serialize_value(arr)
        result = serde.deserialize_value(payload)
        np.testing.assert_array_equal(arr, result)

    def test_preserves_dtype(self):
        """Test dtype preservation."""
        serde = ZeroCopySerde()
        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            arr = np.array([1, 2, 3], dtype=dtype)
            payload = serde.serialize_value(arr)
            result = serde.deserialize_value(payload)
            assert result.dtype == dtype

    def test_large_array(self):
        """Test large array serialization."""
        serde = ZeroCopySerde()
        arr = np.random.rand(1000, 1000)
        payload = serde.serialize_value(arr)
        result = serde.deserialize_value(payload)
        np.testing.assert_array_equal(arr, result)


class TestALLSERDE:
    """Tests for ALL_SERDE mapping."""

    def test_json_serde_registered(self):
        """Test JSONSerde is registered."""
        assert "application/json" in ALL_SERDE
        assert ALL_SERDE["application/json"] is JSONSerde

    def test_pickle_serde_registered(self):
        """Test PickleSerde is registered."""
        assert "application/vnd.bentoml+pickle" in ALL_SERDE
        assert ALL_SERDE["application/vnd.bentoml+pickle"] is PickleSerde

    def test_zerocopy_serde_registered(self):
        """Test ZeroCopySerde is registered."""
        assert "application/vnd.bentoml+zerocopy" in ALL_SERDE
        assert ALL_SERDE["application/vnd.bentoml+zerocopy"] is ZeroCopySerde

    def test_form_urlencoded_alias(self):
        """Test form-urlencoded alias."""
        from _bentoml_impl.serde import MultipartSerde

        assert "application/x-www-form-urlencoded" in ALL_SERDE
        assert ALL_SERDE["application/x-www-form-urlencoded"] is MultipartSerde


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    value: int


class TestSerdeModelSerialization:
    """Tests for IODescriptor model serialization."""

    def test_json_serde_model(self):
        """Test JSONSerde with Pydantic models."""
        from _bentoml_sdk.io_models import IODescriptor

        class TestInput(IODescriptor):
            name: str
            count: int

        serde = JSONSerde()
        model = TestInput(name="test", count=42)
        payload = serde.serialize_model(model)
        result = serde.deserialize_model(payload, TestInput)
        assert result.name == "test"
        assert result.count == 42

    def test_pickle_serde_model(self):
        """Test PickleSerde with Pydantic models."""
        from _bentoml_sdk.io_models import IODescriptor

        class TestInput(IODescriptor):
            name: str
            count: int

        serde = PickleSerde()
        model = TestInput(name="test", count=42)
        payload = serde.serialize_model(model)
        result = serde.deserialize_model(payload, TestInput)
        assert result.name == "test"
        assert result.count == 42

    def test_zerocopy_serde_model(self):
        """Test ZeroCopySerde with Pydantic models."""
        from _bentoml_sdk.io_models import IODescriptor

        class TestInput(IODescriptor):
            name: str
            count: int

        serde = ZeroCopySerde()
        model = TestInput(name="test", count=42)
        payload = serde.serialize_model(model)
        result = serde.deserialize_model(payload, TestInput)
        assert result.name == "test"
        assert result.count == 42
