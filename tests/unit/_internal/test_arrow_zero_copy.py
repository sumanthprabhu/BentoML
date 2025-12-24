"""
Tests for zero-copy functionality in the arrow module.

This module tests the Arrow IPC serialization/deserialization with zero-copy optimizations.
"""

from __future__ import annotations

import io

import pytest


@pytest.mark.skipif(
    not pytest.importorskip("pyarrow", reason="pyarrow not installed"),
    reason="pyarrow not installed",
)
class TestModelToArrowSchema:
    """Tests for model_to_arrow_schema function."""

    def test_simple_model(self):
        """Test converting a simple Pydantic model to Arrow schema."""
        import pyarrow as pa
        from pydantic import BaseModel

        from _bentoml_impl.arrow import model_to_arrow_schema

        class SimpleModel(BaseModel):
            name: str
            value: int

        schema = model_to_arrow_schema(SimpleModel)

        assert isinstance(schema, pa.Schema)
        assert len(schema) == 2
        assert schema.field("name").type == pa.utf8()
        assert schema.field("value").type == pa.int64()

    def test_model_with_float(self):
        """Test converting model with float field."""
        import pyarrow as pa
        from pydantic import BaseModel

        from _bentoml_impl.arrow import model_to_arrow_schema

        class FloatModel(BaseModel):
            score: float

        schema = model_to_arrow_schema(FloatModel)

        assert schema.field("score").type == pa.float64()

    def test_model_with_boolean(self):
        """Test converting model with boolean field."""
        import pyarrow as pa
        from pydantic import BaseModel

        from _bentoml_impl.arrow import model_to_arrow_schema

        class BoolModel(BaseModel):
            active: bool

        schema = model_to_arrow_schema(BoolModel)

        assert schema.field("active").type == pa.bool_()

    def test_model_with_optional(self):
        """Test converting model with optional field."""
        from typing import Optional

        from pydantic import BaseModel

        from _bentoml_impl.arrow import model_to_arrow_schema

        class OptionalModel(BaseModel):
            name: Optional[str] = None

        schema = model_to_arrow_schema(OptionalModel)

        # Optional fields should be nullable
        assert schema.field("name").nullable is True

    def test_model_with_list(self):
        """Test converting model with list field."""
        from typing import List

        import pyarrow as pa
        from pydantic import BaseModel

        from _bentoml_impl.arrow import model_to_arrow_schema

        class ListModel(BaseModel):
            values: List[int]

        schema = model_to_arrow_schema(ListModel)

        assert pa.types.is_list(schema.field("values").type)


@pytest.mark.skipif(
    not pytest.importorskip("pyarrow", reason="pyarrow not installed"),
    reason="pyarrow not installed",
)
class TestSerializeToArrow:
    """Tests for serialize_to_arrow function."""

    def test_serialize_simple_model(self):
        """Test serializing a simple model to Arrow."""
        from pydantic import BaseModel

        from _bentoml_impl.arrow import serialize_to_arrow

        class SimpleModel(BaseModel):
            name: str
            value: int

        model = SimpleModel(name="test", value=42)
        buffer = io.BytesIO()

        serialize_to_arrow(model, buffer)

        assert buffer.tell() > 0

    def test_serialize_with_zero_copy_enabled(self):
        """Test serializing with zero_copy enabled."""
        from pydantic import BaseModel

        from _bentoml_impl.arrow import serialize_to_arrow

        class SimpleModel(BaseModel):
            name: str
            value: int

        model = SimpleModel(name="test", value=42)
        buffer = io.BytesIO()

        serialize_to_arrow(model, buffer, zero_copy=True)

        assert buffer.tell() > 0

    def test_serialize_with_zero_copy_disabled(self):
        """Test serializing with zero_copy disabled."""
        from pydantic import BaseModel

        from _bentoml_impl.arrow import serialize_to_arrow

        class SimpleModel(BaseModel):
            name: str
            value: int

        model = SimpleModel(name="test", value=42)
        buffer = io.BytesIO()

        serialize_to_arrow(model, buffer, zero_copy=False)

        assert buffer.tell() > 0


@pytest.mark.skipif(
    not pytest.importorskip("pyarrow", reason="pyarrow not installed"),
    reason="pyarrow not installed",
)
class TestDeserializeFromArrow:
    """Tests for deserialize_from_arrow function."""

    def test_roundtrip_simple_model(self):
        """Test roundtrip serialization of a simple model."""
        from pydantic import BaseModel

        from _bentoml_impl.arrow import deserialize_from_arrow
        from _bentoml_impl.arrow import serialize_to_arrow

        class SimpleModel(BaseModel):
            name: str
            value: int

        model = SimpleModel(name="test", value=42)
        buffer = io.BytesIO()

        serialize_to_arrow(model, buffer)
        buffer.seek(0)

        result = deserialize_from_arrow(SimpleModel, buffer)

        assert result.name == "test"
        assert result.value == 42

    def test_roundtrip_with_zero_copy(self):
        """Test roundtrip with zero_copy enabled."""
        from pydantic import BaseModel

        from _bentoml_impl.arrow import deserialize_from_arrow
        from _bentoml_impl.arrow import serialize_to_arrow

        class SimpleModel(BaseModel):
            name: str
            score: float

        model = SimpleModel(name="test", score=3.14)
        buffer = io.BytesIO()

        serialize_to_arrow(model, buffer, zero_copy=True)
        buffer.seek(0)

        result = deserialize_from_arrow(SimpleModel, buffer, zero_copy=True)

        assert result.name == "test"
        assert abs(result.score - 3.14) < 0.001

    def test_roundtrip_complex_model(self):
        """Test roundtrip with a more complex model."""
        from typing import List

        from pydantic import BaseModel

        from _bentoml_impl.arrow import deserialize_from_arrow
        from _bentoml_impl.arrow import serialize_to_arrow

        class ComplexModel(BaseModel):
            name: str
            values: List[int]

        model = ComplexModel(name="test", values=[1, 2, 3])
        buffer = io.BytesIO()

        serialize_to_arrow(model, buffer)
        buffer.seek(0)

        result = deserialize_from_arrow(ComplexModel, buffer)

        assert result.name == "test"
        assert result.values == [1, 2, 3]


@pytest.mark.skipif(
    not pytest.importorskip("pyarrow", reason="pyarrow not installed"),
    reason="pyarrow not installed",
)
class TestIsNullable:
    """Tests for is_nullable function."""

    def test_nullable_field(self):
        """Test detecting nullable field."""
        from _bentoml_impl.arrow import is_nullable

        field = {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ]
        }

        assert is_nullable(field) is True

    def test_non_nullable_field(self):
        """Test detecting non-nullable field."""
        from _bentoml_impl.arrow import is_nullable

        field = {"type": "string"}

        assert is_nullable(field) is False

    def test_anyof_without_null(self):
        """Test anyOf without null type."""
        from _bentoml_impl.arrow import is_nullable

        field = {
            "anyOf": [
                {"type": "string"},
                {"type": "integer"},
            ]
        }

        assert is_nullable(field) is False


@pytest.mark.skipif(
    not pytest.importorskip("pyarrow", reason="pyarrow not installed"),
    reason="pyarrow not installed",
)
class TestTypeToArrow:
    """Tests for _type_to_arrow function."""

    def test_integer_type(self):
        """Test converting integer type."""
        import pyarrow as pa

        from _bentoml_impl.arrow import _type_to_arrow

        result = _type_to_arrow("integer")
        assert result == pa.int64()

    def test_number_type(self):
        """Test converting number type."""
        import pyarrow as pa

        from _bentoml_impl.arrow import _type_to_arrow

        result = _type_to_arrow("number")
        assert result == pa.float64()

    def test_string_type(self):
        """Test converting string type."""
        import pyarrow as pa

        from _bentoml_impl.arrow import _type_to_arrow

        result = _type_to_arrow("string")
        assert result == pa.utf8()

    def test_boolean_type(self):
        """Test converting boolean type."""
        import pyarrow as pa

        from _bentoml_impl.arrow import _type_to_arrow

        result = _type_to_arrow("boolean")
        assert result == pa.bool_()

    def test_unsupported_type(self):
        """Test converting unsupported type raises error."""
        from _bentoml_impl.arrow import _type_to_arrow

        with pytest.raises(TypeError, match="unsupported type"):
            _type_to_arrow("unsupported")


@pytest.mark.skipif(
    not pytest.importorskip("pyarrow", reason="pyarrow not installed"),
    reason="pyarrow not installed",
)
class TestFieldSchemaToArrow:
    """Tests for _field_schema_to_arrow function."""

    def test_tensor_field(self):
        """Test converting tensor field schema."""
        import pyarrow as pa

        from _bentoml_impl.arrow import _field_schema_to_arrow

        field = {
            "type": "tensor",
            "dtype": "float32",
        }

        result = _field_schema_to_arrow(field, {})

        assert pa.types.is_list(result)

    def test_tensor_field_without_dtype(self):
        """Test tensor field without dtype raises error."""
        from _bentoml_impl.arrow import _field_schema_to_arrow

        field = {"type": "tensor"}

        with pytest.raises(ValueError, match="dtype must be specified"):
            _field_schema_to_arrow(field, {})

    def test_datetime_field(self):
        """Test converting datetime field schema."""
        import pyarrow as pa

        from _bentoml_impl.arrow import _field_schema_to_arrow

        field = {
            "type": "string",
            "format": "date-time",
        }

        result = _field_schema_to_arrow(field, {})

        assert pa.types.is_timestamp(result)

    def test_binary_field(self):
        """Test converting binary field schema."""
        import pyarrow as pa

        from _bentoml_impl.arrow import _field_schema_to_arrow

        field = {
            "type": "string",
            "format": "binary",
        }

        result = _field_schema_to_arrow(field, {})

        assert result == pa.binary()

    def test_nested_object_field(self):
        """Test converting nested object field schema."""
        import pyarrow as pa

        from _bentoml_impl.arrow import _field_schema_to_arrow

        field = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "integer"},
            },
        }

        result = _field_schema_to_arrow(field, {})

        assert pa.types.is_struct(result)

    def test_map_field(self):
        """Test converting map/dict field schema."""
        import pyarrow as pa

        from _bentoml_impl.arrow import _field_schema_to_arrow

        field = {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        }

        result = _field_schema_to_arrow(field, {})

        assert pa.types.is_map(result)

    def test_ref_field(self):
        """Test converting field with $ref."""
        import pyarrow as pa

        from _bentoml_impl.arrow import _field_schema_to_arrow

        field = {"$ref": "#/$defs/MyType"}
        ref_defs = {"MyType": {"type": "string"}}

        result = _field_schema_to_arrow(field, ref_defs)

        assert result == pa.utf8()
