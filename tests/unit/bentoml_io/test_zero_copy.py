"""End-to-end tests for zero-copy tensor serialization."""

from __future__ import annotations

import numpy as np
import pytest
from starlette.testclient import TestClient

import bentoml


@pytest.fixture
def zero_copy_service():
    """Create a service with zero_copy=True."""

    @bentoml.service(zero_copy=True, metrics={"enabled": False})
    class ZeroCopyService:
        @bentoml.api
        def process_array(self, data: np.ndarray) -> np.ndarray:
            return data * 2

        @bentoml.api
        def get_array(self) -> np.ndarray:
            return np.array([1.0, 2.0, 3.0])

    return ZeroCopyService


@pytest.fixture
def no_zero_copy_service():
    """Create a service with zero_copy=False."""

    @bentoml.service(zero_copy=False, metrics={"enabled": False})
    class NoZeroCopyService:
        @bentoml.api
        def process_array(self, data: np.ndarray) -> np.ndarray:
            return data * 2

        @bentoml.api
        def get_array(self) -> np.ndarray:
            return np.array([1.0, 2.0, 3.0])

    return NoZeroCopyService


class TestZeroCopyConfig:
    """Test zero_copy configuration."""

    def test_zero_copy_enabled(self, zero_copy_service):
        """Test that zero_copy is properly configured when enabled."""
        assert zero_copy_service.zero_copy is True
        assert zero_copy_service.config.get("zero_copy") is True

    def test_zero_copy_disabled(self, no_zero_copy_service):
        """Test that zero_copy is properly configured when disabled."""
        assert no_zero_copy_service.zero_copy is False
        assert no_zero_copy_service.config.get("zero_copy") is False


class TestZeroCopyLocalExecution:
    """Test local execution with zero_copy services."""

    def test_local_array_processing_enabled(self, zero_copy_service):
        """Test local array processing with zero_copy enabled."""
        svc = zero_copy_service()
        input_arr = np.array([1.0, 2.0, 3.0])
        result = svc.process_array(input_arr)
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_equal(result, expected)

    def test_local_array_processing_disabled(self, no_zero_copy_service):
        """Test local array processing with zero_copy disabled."""
        svc = no_zero_copy_service()
        input_arr = np.array([1.0, 2.0, 3.0])
        result = svc.process_array(input_arr)
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_equal(result, expected)

    def test_local_get_array_enabled(self, zero_copy_service):
        """Test local get_array with zero_copy enabled."""
        svc = zero_copy_service()
        result = svc.get_array()
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)


class TestZeroCopyHTTP:
    """Test HTTP endpoints with zero_copy services."""

    def test_http_array_json(self, zero_copy_service):
        """Test HTTP endpoint with JSON serialization."""
        with TestClient(app=zero_copy_service.to_asgi()) as client:
            response = client.post(
                "/process_array",
                json={"data": [1.0, 2.0, 3.0]},
            )
            assert response.status_code == 200
            result = np.array(response.json())
            expected = np.array([2.0, 4.0, 6.0])
            np.testing.assert_array_equal(result, expected)

    def test_http_get_array_json(self, zero_copy_service):
        """Test HTTP endpoint returning array with JSON."""
        with TestClient(app=zero_copy_service.to_asgi()) as client:
            response = client.post("/get_array", json={})
            assert response.status_code == 200
            result = np.array(response.json())
            expected = np.array([1.0, 2.0, 3.0])
            np.testing.assert_array_equal(result, expected)


class TestZeroCopySerdeSelection:
    """Test that correct serde is selected based on zero_copy setting."""

    def test_zerocopy_media_type(self):
        """Test ZeroCopySerde media type."""
        from _bentoml_impl.serde import ZeroCopySerde

        serde = ZeroCopySerde()
        assert serde.media_type == "application/vnd.bentoml+zerocopy"

    def test_pickle_media_type(self):
        """Test PickleSerde media type."""
        from _bentoml_impl.serde import PickleSerde

        serde = PickleSerde()
        assert serde.media_type == "application/vnd.bentoml+pickle"

    def test_serde_roundtrip_zerocopy(self):
        """Test ZeroCopySerde roundtrip."""
        from _bentoml_impl.serde import ZeroCopySerde

        serde = ZeroCopySerde()
        arr = np.random.rand(100, 100)
        payload = serde.serialize_value(arr)
        result = serde.deserialize_value(payload)
        np.testing.assert_array_equal(arr, result)

    def test_serde_roundtrip_pickle(self):
        """Test PickleSerde roundtrip."""
        from _bentoml_impl.serde import PickleSerde

        serde = PickleSerde()
        arr = np.random.rand(100, 100)
        payload = serde.serialize_value(arr)
        result = serde.deserialize_value(payload)
        np.testing.assert_array_equal(arr, result)


class TestZeroCopyDataTypes:
    """Test zero-copy with different data types."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    def test_dtype_preservation(self, dtype):
        """Test that dtype is preserved in zero-copy serialization."""
        from _bentoml_impl.serde import ZeroCopySerde

        serde = ZeroCopySerde()
        arr = np.array([1, 2, 3, 4], dtype=dtype)
        payload = serde.serialize_value(arr)
        result = serde.deserialize_value(payload)
        assert result.dtype == dtype
        np.testing.assert_array_equal(arr, result)

    @pytest.mark.parametrize("shape", [(10,), (5, 5), (2, 3, 4)])
    def test_shape_preservation(self, shape):
        """Test that shape is preserved in zero-copy serialization."""
        from _bentoml_impl.serde import ZeroCopySerde

        serde = ZeroCopySerde()
        arr = np.random.rand(*shape)
        payload = serde.serialize_value(arr)
        result = serde.deserialize_value(payload)
        assert result.shape == shape
        np.testing.assert_array_equal(arr, result)


class TestZeroCopyComparison:
    """Compare zero-copy vs regular pickle performance characteristics."""

    def test_both_produce_same_results(self):
        """Test that both serdes produce identical results."""
        from _bentoml_impl.serde import PickleSerde
        from _bentoml_impl.serde import ZeroCopySerde

        pickle_serde = PickleSerde()
        zerocopy_serde = ZeroCopySerde()

        arr = np.random.rand(100, 100)

        pickle_payload = pickle_serde.serialize_value(arr)
        pickle_result = pickle_serde.deserialize_value(pickle_payload)

        zerocopy_payload = zerocopy_serde.serialize_value(arr)
        zerocopy_result = zerocopy_serde.deserialize_value(zerocopy_payload)

        np.testing.assert_array_equal(pickle_result, zerocopy_result)

    def test_zerocopy_has_metadata(self):
        """Test that zero-copy adds metadata marker."""
        from _bentoml_impl.serde import ZeroCopySerde

        serde = ZeroCopySerde()
        arr = np.random.rand(10, 10)
        payload = serde.serialize_value(arr)
        assert payload.metadata.get("zero-copy") == "true"
