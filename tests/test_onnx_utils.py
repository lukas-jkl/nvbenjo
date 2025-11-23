import typing as ty
from dataclasses import dataclass

import pytest
import torch

try:
    from nvbenjo import onnx_utils
except ImportError as e:
    if "onnxruntime" in str(e):
        pytest.skip("onnxruntime is not installed, skipping ONNX utils tests.", allow_module_level=True)
    else:
        raise
from nvbenjo.cfg import OnnxRuntimeConfig


@dataclass
class FakeOnnxInput:
    name: str
    type: str
    shape: list[ty.Union[int, str]]


def test_get_model():
    with pytest.raises(
        ValueError, match="Invalid model tests/data/doesnotexist.onnx. Must be a valid ONNX path ending with .onnx"
    ):
        _ = onnx_utils.get_model(
            "tests/data/doesnotexist.onnx", device=torch.device("cpu"), runtime_config=OnnxRuntimeConfig()
        )


@pytest.mark.parametrize(
    "user_input_shapes",
    [
        ("B", 3, 224, 224),
        (("B", 3, 224, 224),),
        {"name": "input", "type": "float", "shape": ("B", 3, 224, 224)},
        {"name": "input", "type": "float"},
        {"name": "input"},
    ],
)
def test_get_rnd_input_batch(user_input_shapes):
    inputs = [FakeOnnxInput(name="input", type="tensor(float)", shape=["B", 3, 224, 224])]
    user_input_shapes = ("B", 3, 224, 224)
    batch_size = 4
    rnd_inputs = onnx_utils.get_rnd_input_batch(inputs, user_input_shapes, batch_size)
    assert isinstance(rnd_inputs, dict)
    assert len(rnd_inputs) == len(inputs)
    for inp_name, inp in rnd_inputs.items():
        assert isinstance(inp, torch.Tensor)
        assert inp.shape == (batch_size, 3, 224, 224)
        assert inp.dtype == torch.float32
        assert inp_name == "input"


def test_invalid_get_rnd_input_batch():
    inputs = [
        FakeOnnxInput(name="input1", type="tensor(float)", shape=["B", 3, 224, 224]),
        FakeOnnxInput(name="input2", type="tensor(int64)", shape=[1, 10]),
    ]
    batch_size = 4

    # single shape but multiple model inputs
    user_input_shapes = ("B", 3, 224, 224)
    with pytest.raises(ValueError, match="The model has multiple inputs, but the provided input is a single shape."):
        _ = onnx_utils.get_rnd_input_batch(inputs, user_input_shapes, batch_size)

    # mismatching number of shapes
    user_input_shapes = (("B", 3, 224, 224), ("B", 10), (1, 5))
    with pytest.raises(ValueError, match="The model has 2 inputs, but the provided input has 3 shapes."):
        _ = onnx_utils.get_rnd_input_batch(inputs, user_input_shapes, batch_size)

    # invalid input name
    user_input_shapes = (
        {"name": "invalid_input", "type": "float", "shape": ("B", 3, 224, 224)},
        {"name": "input2", "type": "int", "shape": (1, 10)},
    )
    with pytest.raises(ValueError, match="The model does not have an input named invalid_input."):
        _ = onnx_utils.get_rnd_input_batch(inputs, user_input_shapes, batch_size)

    # invalid num shapes
    user_input_shapes = ({"name": "input1", "type": "float", "shape": ("B", 3, 224, 2.3)},)
    with pytest.raises(ValueError, match="The model has 2 inputs, but the provided input has 1 shapes."):
        _ = onnx_utils.get_rnd_input_batch(inputs, user_input_shapes, batch_size)

    # invalid shape
    user_input_shapes = (
        {"name": "input1", "type": "float", "shape": ("B", 3, 224, 2.3)},
        {"name": "input2", "type": "float", "shape": ("B", 3, 224, 2.3)},
    )
    with pytest.raises(ValueError, match="Failed to generate random input from shape"):
        _ = onnx_utils.get_rnd_input_batch(inputs, user_input_shapes, batch_size)
