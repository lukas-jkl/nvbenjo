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
from nvbenjo import benchmark
from nvbenjo.cfg import OnnxModelConfig, OnnxRuntimeConfig
from tests.test_torch_python_api import _CountingModel


@dataclass
class FakeOnnxInput:
    name: str
    type: str
    shape: list[int | str]


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


class _CountingSession:
    """Wraps an onnxruntime session and counts how often it was run."""

    def __init__(self, session):
        self._session = session
        self.num_inferences = 0

    def run_with_iobinding(self, *args, **kwargs):
        self.num_inferences += 1
        return self._session.run_with_iobinding(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._session, name)


def test_warmup_runs_model(tmp_path, monkeypatch):
    num_warmup_batches, num_batches = 3, 2
    onnx_path = tmp_path / "counting.onnx"
    torch.onnx.export(
        _CountingModel(),
        args=(torch.randn(2, 16),),
        f=str(onnx_path),
        input_names=["x"],
        output_names=["output"],
        dynamic_axes={"x": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
    )
    session = _CountingSession(
        onnx_utils.get_model(str(onnx_path), device=torch.device("cpu"), runtime_config=OnnxRuntimeConfig())
    )
    monkeypatch.setattr(benchmark, "load_model", lambda *args, **kwargs: session)

    model_cfg = OnnxModelConfig(
        name="counting-onnx",
        type_or_path=f"onnx:{onnx_path}",
        shape=({"name": "x", "shape": ("B", 16)},),
        devices=["cpu"],
        batch_sizes=[1],
        num_warmup_batches=num_warmup_batches,
        num_batches=num_batches,
    )
    benchmark.benchmark_model(model_cfg, measure_memory=False)

    assert session.num_inferences == num_warmup_batches + num_batches
