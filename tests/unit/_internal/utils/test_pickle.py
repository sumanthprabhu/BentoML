from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    import bentoml._internal.external_typing as ext


def test_pep574_restore() -> None:
    import numpy as np
    import pandas as pd

    from bentoml._internal.utils.pickle import pep574_dumps
    from bentoml._internal.utils.pickle import pep574_loads

    arr1: ext.NpNDArray = np.random.uniform(size=(20, 20))
    arr2: ext.NpNDArray = np.random.uniform(size=(64, 64))
    arr3: ext.NpNDArray = np.random.uniform(size=(72, 72))

    lst = [arr1, arr2, arr3]

    bs: bytes
    concat_buffer_bs: bytes
    indices: list[int]
    bs, concat_buffer_bs, indices = pep574_dumps(lst)
    restored = t.cast(
        t.List["ext.NpNDArray"], pep574_loads(bs, concat_buffer_bs, indices)
    )
    for idx, arr in enumerate(lst):
        assert np.isclose(arr, restored[idx]).all()

    dic: dict[str, ext.NpNDArray] = dict(a=arr1, b=arr2, c=arr3)
    bs, concat_buffer_bs, indices = pep574_dumps(dic)
    restored = t.cast(
        t.Dict[str, "ext.NpNDArray"], pep574_loads(bs, concat_buffer_bs, indices)
    )
    for key, arr in dic.items():
        assert np.isclose(arr, restored[key]).all()

    df1: ext.PdDataFrame = pd.DataFrame(arr1)
    df2: ext.PdDataFrame = pd.DataFrame(arr2)
    df3: ext.PdDataFrame = pd.DataFrame(arr3)

    df_lst = [df1, df2, df3]

    bs, concat_buffer_bs, indices = pep574_dumps(df_lst)
    restored = t.cast(
        t.List["ext.PdDataFrame"], pep574_loads(bs, concat_buffer_bs, indices)
    )
    for idx, df in enumerate(df_lst):
        assert np.isclose(df.to_numpy(), restored[idx].to_numpy()).all()

    df_dic: dict[str, ext.PdDataFrame] = dict(a=df1, b=df2, c=df3)
    bs, concat_buffer_bs, indices = pep574_dumps(df_dic)
    restored = t.cast(
        t.Dict[str, "ext.PdDataFrame"], pep574_loads(bs, concat_buffer_bs, indices)
    )
    for key, df in df_dic.items():
        assert np.isclose(df.to_numpy(), restored[key].to_numpy()).all()


def test_zero_copy_pickle_buffers() -> None:
    """Test zero-copy pickle serialization with out-of-band buffers."""
    import pickle

    import numpy as np

    # Create large arrays to ensure out-of-band buffer usage
    arr = np.random.uniform(size=(100, 100))

    # Serialize with buffer callback
    buffers: list[pickle.PickleBuffer] = []
    main_bytes = pickle.dumps(arr, protocol=5, buffer_callback=buffers.append)

    # Should have created out-of-band buffers for large array
    assert len(buffers) > 0

    # Reconstruct using buffers
    result = pickle.loads(main_bytes, buffers=buffers)
    np.testing.assert_array_equal(arr, result)


def test_zero_copy_preserves_dtype() -> None:
    """Test that zero-copy pickle preserves array dtype."""
    import pickle

    import numpy as np

    for dtype in [np.float32, np.float64, np.int32, np.int64, np.complex128]:
        arr = np.array([1, 2, 3, 4], dtype=dtype)

        buffers: list[pickle.PickleBuffer] = []
        main_bytes = pickle.dumps(arr, protocol=5, buffer_callback=buffers.append)
        result = pickle.loads(main_bytes, buffers=buffers)

        assert result.dtype == dtype
        np.testing.assert_array_equal(arr, result)


def test_zero_copy_large_tensor_bypasses_main_stream() -> None:
    """Test that large tensors use out-of-band buffers."""
    import pickle

    import numpy as np

    # Small array - may or may not use out-of-band
    small_arr = np.array([1, 2, 3])
    small_buffers: list[pickle.PickleBuffer] = []
    _ = pickle.dumps(small_arr, protocol=5, buffer_callback=small_buffers.append)

    # Large array - should use out-of-band
    large_arr = np.random.rand(1000, 1000)
    large_buffers: list[pickle.PickleBuffer] = []
    large_main = pickle.dumps(
        large_arr, protocol=5, buffer_callback=large_buffers.append
    )

    # Large array should have out-of-band buffers
    assert len(large_buffers) > 0

    # Main pickle stream for large array should be much smaller than the data
    large_data_size = large_arr.nbytes
    assert len(large_main) < large_data_size


def test_zero_copy_memoryview_buffer() -> None:
    """Test that PickleBuffer.raw() returns memoryview."""
    import pickle

    import numpy as np

    arr = np.random.rand(100, 100)
    buffers: list[pickle.PickleBuffer] = []
    pickle.dumps(arr, protocol=5, buffer_callback=buffers.append)

    for buff in buffers:
        raw = buff.raw()
        assert isinstance(raw, memoryview)


def test_zero_copy_multiple_arrays() -> None:
    """Test zero-copy with multiple arrays in same structure."""
    import pickle

    import numpy as np

    data = {
        "arr1": np.random.rand(50, 50),
        "arr2": np.random.rand(100, 100),
        "scalar": 42,
        "nested": {"arr3": np.random.rand(25, 25)},
    }

    buffers: list[pickle.PickleBuffer] = []
    main_bytes = pickle.dumps(data, protocol=5, buffer_callback=buffers.append)
    result = pickle.loads(main_bytes, buffers=buffers)

    np.testing.assert_array_equal(data["arr1"], result["arr1"])
    np.testing.assert_array_equal(data["arr2"], result["arr2"])
    assert data["scalar"] == result["scalar"]
    np.testing.assert_array_equal(data["nested"]["arr3"], result["nested"]["arr3"])
