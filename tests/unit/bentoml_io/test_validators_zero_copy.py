"""
Tests for zero-copy functionality in the validators module.

This module tests the TensorSchema validation and encoding with zero-copy optimizations.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestTensorSchemaEncode:
    """Tests for TensorSchema.encode method with zero-copy."""

    def test_encode_numpy_json_mode(self):
        """Test encoding numpy array in JSON mode."""
        from _bentoml_sdk.validators import TensorSchema

        schema = TensorSchema(format="numpy-array", dtype="float32")
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Create mock serialization info for JSON mode
        class MockInfo:
            def mode_is_json(self):
                return True

        result = schema.encode(arr, MockInfo())

        # In JSON mode, should return a list
        assert result == [1.0, 2.0, 3.0]

    def test_encode_numpy_python_mode(self):
        """Test encoding numpy array in Python mode."""
        from _bentoml_sdk.validators import TensorSchema

        schema = TensorSchema(format="numpy-array", dtype="float32")
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        class MockInfo:
            def mode_is_json(self):
                return False

        result = schema.encode(arr, MockInfo())

        # In Python mode, should return numpy array
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"),
        reason="torch not installed",
    )
    def test_encode_torch_tensor(self):
        """Test encoding PyTorch tensor."""
        import torch

        from _bentoml_sdk.validators import TensorSchema

        schema = TensorSchema(format="torch-tensor", dtype="float32")
        tensor = torch.tensor([1.0, 2.0, 3.0])

        class MockInfo:
            def mode_is_json(self):
                return True

        result = schema.encode(tensor, MockInfo())

        # In JSON mode, should return a list
        assert result == pytest.approx([1.0, 2.0, 3.0])

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"),
        reason="torch not installed",
    )
    def test_encode_torch_gpu_tensor(self):
        """Test encoding PyTorch GPU tensor falls back to CPU."""
        import torch

        from _bentoml_sdk.validators import TensorSchema

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        schema = TensorSchema(format="torch-tensor", dtype="float32")
        tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()

        class MockInfo:
            def mode_is_json(self):
                return True

        result = schema.encode(tensor, MockInfo())

        assert result == pytest.approx([1.0, 2.0, 3.0])


class TestTensorSchemaValidate:
    """Tests for TensorSchema.validate method with zero-copy."""

    def test_validate_numpy_passthrough(self):
        """Test that numpy arrays pass through validation."""
        from _bentoml_sdk.validators import TensorSchema

        schema = TensorSchema(format="numpy-array")
        arr = np.array([1.0, 2.0, 3.0])

        result = schema.validate(arr)

        assert result is arr

    def test_validate_list_to_numpy(self):
        """Test validating list to numpy array."""
        from _bentoml_sdk.validators import TensorSchema

        schema = TensorSchema(format="numpy-array", dtype="float32")
        data = [1.0, 2.0, 3.0]

        result = schema.validate(data)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, data)

    def test_validate_with_shape(self):
        """Test validating with shape specification."""
        from _bentoml_sdk.validators import TensorSchema

        schema = TensorSchema(format="numpy-array", shape=(2, 3))
        data = [[1, 2, 3], [4, 5, 6]]

        result = schema.validate(data)

        assert result.shape == (2, 3)

    def test_validate_reshape(self):
        """Test validating with reshape."""
        from _bentoml_sdk.validators import TensorSchema

        schema = TensorSchema(format="numpy-array", shape=(3, 2))
        data = [1, 2, 3, 4, 5, 6]

        result = schema.validate(data)

        assert result.shape == (3, 2)

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"),
        reason="torch not installed",
    )
    def test_validate_torch_passthrough(self):
        """Test that torch tensors pass through validation."""
        import torch

        from _bentoml_sdk.validators import TensorSchema

        schema = TensorSchema(format="torch-tensor")
        tensor = torch.tensor([1.0, 2.0, 3.0])

        result = schema.validate(tensor)

        assert result is tensor

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"),
        reason="torch not installed",
    )
    def test_validate_numpy_to_torch(self):
        """Test validating numpy array to torch tensor."""
        import torch

        from _bentoml_sdk.validators import TensorSchema

        schema = TensorSchema(format="torch-tensor")
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        result = schema.validate(arr)

        assert isinstance(result, torch.Tensor)

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"),
        reason="torch not installed",
    )
    def test_validate_list_to_torch(self):
        """Test validating list to torch tensor."""
        import torch

        from _bentoml_sdk.validators import TensorSchema

        schema = TensorSchema(format="torch-tensor", dtype="float32")
        data = [1.0, 2.0, 3.0]

        result = schema.validate(data)

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32


class TestTensorSchemaProperties:
    """Tests for TensorSchema properties."""

    def test_dim_property(self):
        """Test dim property calculation."""
        from _bentoml_sdk.validators import TensorSchema

        schema = TensorSchema(format="numpy-array", shape=(2, 3, 4))

        assert schema.dim == 24  # 2 * 3 * 4

    def test_dim_property_none(self):
        """Test dim property when shape is None."""
        from _bentoml_sdk.validators import TensorSchema

        schema = TensorSchema(format="numpy-array")

        assert schema.dim is None

    def test_framework_dtype_numpy(self):
        """Test framework_dtype for numpy."""
        from _bentoml_sdk.validators import TensorSchema

        schema = TensorSchema(format="numpy-array", dtype="float32")

        assert schema.framework_dtype == np.float32

    def test_framework_dtype_none(self):
        """Test framework_dtype when dtype is None."""
        from _bentoml_sdk.validators import TensorSchema

        schema = TensorSchema(format="numpy-array")

        assert schema.framework_dtype is None

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"),
        reason="torch not installed",
    )
    def test_framework_dtype_torch(self):
        """Test framework_dtype for torch."""
        import torch

        from _bentoml_sdk.validators import TensorSchema

        schema = TensorSchema(format="torch-tensor", dtype="float32")

        assert schema.framework_dtype == torch.float32


class TestTensorSchemaJsonSchema:
    """Tests for TensorSchema JSON schema generation."""

    def test_json_schema_validation_mode(self):
        """Test JSON schema in validation mode."""
        from _bentoml_sdk.validators import TensorSchema

        schema = TensorSchema(format="numpy-array", dtype="float32", shape=(2, 3))

        class MockHandler:
            mode = "validation"

            def __call__(self, schema):
                return {}

        result = schema.__get_pydantic_json_schema__(None, MockHandler())

        assert result["type"] == "tensor"
        assert result["format"] == "numpy-array"
        assert result["dtype"] == "float32"
        assert result["shape"] == (2, 3)
        assert result["dim"] == 6

    def test_json_schema_serialization_mode(self):
        """Test JSON schema in serialization mode."""
        from _bentoml_sdk.validators import TensorSchema

        schema = TensorSchema(format="numpy-array", shape=(2, 3))

        class MockHandler:
            mode = "serialization"

            def __call__(self, schema):
                return {}

        result = schema.__get_pydantic_json_schema__(None, MockHandler())

        # Serialization mode returns array structure
        assert result["type"] == "array"


class TestArrowSerialization:
    """Tests for arrow_serialization context manager."""

    def test_arrow_serialization_context(self):
        """Test arrow_serialization context manager."""
        from _bentoml_sdk.validators import __in_arrow_serialization__
        from _bentoml_sdk.validators import arrow_serialization

        # Before context
        assert __in_arrow_serialization__ is False

        with arrow_serialization():
            # Import again to get current value
            from _bentoml_sdk.validators import __in_arrow_serialization__ as in_arrow

            assert in_arrow is True

        # After context
        from _bentoml_sdk.validators import __in_arrow_serialization__ as after_arrow

        assert after_arrow is False

    def test_encode_in_arrow_mode(self):
        """Test that encode flattens array in arrow mode."""
        from _bentoml_sdk.validators import TensorSchema
        from _bentoml_sdk.validators import arrow_serialization

        schema = TensorSchema(format="numpy-array")
        arr = np.array([[1, 2], [3, 4]])

        class MockInfo:
            def mode_is_json(self):
                return False

        with arrow_serialization():
            result = schema.encode(arr, MockInfo())

        # Should be flattened
        assert result.shape == (4,)
        np.testing.assert_array_equal(result, [1, 2, 3, 4])


class TestDataframeSchema:
    """Tests for DataframeSchema."""

    @pytest.mark.skipif(
        not pytest.importorskip("pandas", reason="pandas not installed"),
        reason="pandas not installed",
    )
    def test_validate_dict_to_dataframe(self):
        """Test validating dict to DataFrame."""
        import pandas as pd

        from _bentoml_sdk.validators import DataframeSchema

        schema = DataframeSchema(orient="records")
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

        result = schema.validate(data)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]

    @pytest.mark.skipif(
        not pytest.importorskip("pandas", reason="pandas not installed"),
        reason="pandas not installed",
    )
    def test_validate_dataframe_passthrough(self):
        """Test that DataFrame passes through validation."""
        import pandas as pd

        from _bentoml_sdk.validators import DataframeSchema

        schema = DataframeSchema()
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        result = schema.validate(df)

        assert result is df

    @pytest.mark.skipif(
        not pytest.importorskip("pandas", reason="pandas not installed"),
        reason="pandas not installed",
    )
    def test_encode_records_orient(self):
        """Test encoding DataFrame with records orient."""
        import pandas as pd

        from _bentoml_sdk.validators import DataframeSchema

        schema = DataframeSchema(orient="records")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        class MockInfo:
            def mode_is_json(self):
                return True

        result = schema.encode(df, MockInfo())

        assert result == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]

    @pytest.mark.skipif(
        not pytest.importorskip("pandas", reason="pandas not installed"),
        reason="pandas not installed",
    )
    def test_encode_columns_orient(self):
        """Test encoding DataFrame with columns orient."""
        import pandas as pd

        from _bentoml_sdk.validators import DataframeSchema

        schema = DataframeSchema(orient="columns")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        class MockInfo:
            def mode_is_json(self):
                return True

        result = schema.encode(df, MockInfo())

        assert result == {"a": [1, 2], "b": [3, 4]}
