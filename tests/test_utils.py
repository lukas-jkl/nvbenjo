import pytest
import torch

from nvbenjo.utils import EXAMPLE_VALID_SHAPES, format_num, format_seconds, get_rnd_from_shape_s, NoBatchShapeError


def test_format_seconds():
    assert format_seconds(1.001) == "1.001 s"
    assert format_seconds(0.001001) == "1.001 ms"
    assert format_seconds(0.000001001) == "1.001 us"


def test_format_num():
    assert format_num(1000) == "1.00 K"
    assert format_num(1000000) == "1.00 M"
    assert format_num(1000000000) == "1.00 G"
    assert format_num(1000000000000) == "1.00 T"
    assert format_num(1000000000000000) == "1.00 P"
    assert format_num(1024, bytes=True) == "1.00 KB"
    assert format_num(1048576, bytes=True) == "1.00 MB"
    assert format_num(1073741824, bytes=True) == "1.00 GB"
    assert format_num(1099511627776, bytes=True) == "1.00 TB"
    assert format_num(1125899906842624, bytes=True) == "1.00 PB"


def test_get_rnd_shape_examples():
    for example_shape in EXAMPLE_VALID_SHAPES:
        _ = get_rnd_from_shape_s(example_shape, batch_size=12)
    _ = get_rnd_from_shape_s(({"name": "input1", "type": "float", "shape": ("B", 6), "value": 1},), batch_size=12)
    with pytest.raises(NoBatchShapeError, match="Please ensure that the shape contains an identifier"):
        _ = get_rnd_from_shape_s(
            ({"name": "input1", "type": "float", "shape": (6,), "value": [1, 2, 3, 4, 5, 6]},), batch_size=12
        )
    _ = get_rnd_from_shape_s(
        (
            {"name": "input1", "type": "float", "shape": (6,), "value": [1, 2, 3, 4, 5, 6]},
            {"name": "input1", "type": "float", "shape": (6, "B"), "value": 1},
        ),
        batch_size=12,
    )


def test_get_rnd_shape_invalid():
    example_shape = ("B", 3, 224, 224)
    with pytest.raises(ValueError):
        _ = get_rnd_from_shape_s(example_shape, batch_size=12, min_val=0, max_val=1, dtype="invalid_dtype")
    with pytest.raises(ValueError):
        _ = get_rnd_from_shape_s(example_shape, batch_size=12, min_val="invalid", max_val=1)
    with pytest.raises(ValueError):
        _ = get_rnd_from_shape_s(example_shape, batch_size=12, min_val=0, max_val="invalid")
    with pytest.raises(ValueError):
        _ = get_rnd_from_shape_s((1, 3, 224, 224), batch_size=12)  # missing batch size identifier

    with pytest.raises(ValueError):
        _ = get_rnd_from_shape_s(tuple(), batch_size=12, min_val=0, max_val=1)


def test_get_rnd_shape_valid():
    example_shape = ("B", 3, 224, 224)
    rnd_tensor, set_individual_dtype = get_rnd_from_shape_s(
        example_shape, batch_size=12, min_val=0, max_val=1, dtype="float32"
    )
    assert torch.all(rnd_tensor >= 0) and torch.all(rnd_tensor <= 1)
    assert not set_individual_dtype
    assert rnd_tensor.shape == (12, 3, 224, 224)
    assert rnd_tensor.dtype == torch.float32

    rnd_tensor, _ = get_rnd_from_shape_s(example_shape, batch_size=12, min_val=0, max_val=1, dtype="int32")
    assert rnd_tensor.shape == (12, 3, 224, 224)
    assert rnd_tensor.dtype == torch.int32

    example_shape = (("B", 3, 224, 224), (1, 3, 12, 13))
    rnd_tensor, _ = get_rnd_from_shape_s(example_shape, batch_size=12, min_val=0, max_val=1, dtype="float32")
    assert rnd_tensor[0].shape == (12, 3, 224, 224)
    assert rnd_tensor[1].shape == (1, 3, 12, 13)

    example_shape = (
        {"name": "input1", "type": "float", "shape": ("B", 3, 224, 224), "min_max": (3, 12)},
        {"name": "input2", "type": "int", "shape": (1, 3), "min_max": (0, 1)},
    )
    rnd_tensor, set_individual_dtype = get_rnd_from_shape_s(example_shape, batch_size=12, dtype="float32")
    assert set_individual_dtype
    assert rnd_tensor["input1"].shape == (12, 3, 224, 224)
    assert rnd_tensor["input1"].dtype == torch.float32
    assert torch.all(rnd_tensor["input1"] >= 3) and torch.all(rnd_tensor["input1"] <= 12)
    assert rnd_tensor["input2"].shape == (1, 3)
    assert rnd_tensor["input2"].dtype == torch.int32
    assert torch.all(rnd_tensor["input2"] >= 0) and torch.all(rnd_tensor["input2"] <= 1)
