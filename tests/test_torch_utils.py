from contextlib import nullcontext

import pytest
import torch
from torch import nn

from nvbenjo.cfg import TorchRuntimeConfig
from nvbenjo.torch_utils import (
    _aoti_load_kwargs,
    apply_batch_precision,
    apply_non_amp_model_precision,
    get_amp_ctxt_for_precision,
    get_model,
    get_model_parameters,
    run_model_with_input,
)
from nvbenjo.utils import CompileMode, PrecisionType


def test_get_model_parameters():
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 10, bias=False)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    num_params = get_model_parameters(model)
    assert num_params == 100


def test_apply_non_amp_model_precision():
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 10, bias=False)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    batch = torch.randn(10, 10)
    model = apply_non_amp_model_precision(model, PrecisionType.FP16)
    batch = apply_batch_precision(batch, PrecisionType.FP16)
    assert model.fc.weight.dtype == torch.float16
    assert batch.dtype == torch.float16

    model = SimpleModel()
    batch = torch.randn(10, 10)
    model = apply_non_amp_model_precision(model, PrecisionType.FP32)
    batch = apply_batch_precision(batch, PrecisionType.FP32)
    assert model.fc.weight.dtype == torch.float32
    assert batch.dtype == torch.float32

    model = SimpleModel()
    batch = torch.randn(10, 10)
    model = apply_non_amp_model_precision(model, PrecisionType.BFLOAT16)
    batch = apply_batch_precision(batch, PrecisionType.BFLOAT16)
    assert model.fc.weight.dtype == torch.bfloat16
    assert batch.dtype == torch.bfloat16

    model = SimpleModel()
    batch = torch.randn(10, 10)
    model = apply_non_amp_model_precision(model, PrecisionType.AMP_FP16)
    batch = apply_batch_precision(batch, PrecisionType.AMP_FP16)
    # only shall apply non-amp precisions
    assert model.fc.weight.dtype == torch.float32
    assert batch.dtype == torch.float32


@pytest.mark.parametrize(
    "precision,expected_dtype",
    [
        (PrecisionType.FP16, torch.float16),
        (PrecisionType.BFLOAT16, torch.bfloat16),
        pytest.param(
            PrecisionType.FP8_E4M3FN,
            getattr(torch, "float8_e4m3fn", None),
            marks=pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="requires PyTorch >= 2.1"),
        ),
        pytest.param(
            PrecisionType.FP8_E5M2,
            getattr(torch, "float8_e5m2", None),
            marks=pytest.mark.skipif(not hasattr(torch, "float8_e5m2"), reason="requires PyTorch >= 2.1"),
        ),
    ],
)
def test_apply_precision(precision, expected_dtype):
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 10, bias=False)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    batch = torch.randn(10, 10)
    model = apply_non_amp_model_precision(model, precision)
    batch = apply_batch_precision(batch, precision)
    assert model.fc.weight.dtype == expected_dtype
    assert batch.dtype == expected_dtype


def test_get_amp_ctxt_for_precision():
    ctxt = get_amp_ctxt_for_precision(PrecisionType.AMP, torch.device("cpu"))
    assert isinstance(ctxt, torch.autocast)

    ctxt = get_amp_ctxt_for_precision(PrecisionType.FP32, torch.device("cpu"))
    assert isinstance(ctxt, nullcontext)


@pytest.mark.parametrize(
    "compile_input,expected_mode",
    [
        (False, CompileMode.NONE),
        (True, CompileMode.TORCH_COMPILE),
        ("torch_compile", CompileMode.TORCH_COMPILE),
        ("aot_compile", CompileMode.AOT_COMPILE),
        ("none", CompileMode.NONE),
        ("AOT_COMPILE", CompileMode.AOT_COMPILE),
    ],
)
def test_runtime_config_compile_mode(compile_input, expected_mode):
    cfg = TorchRuntimeConfig(compile=compile_input)
    assert cfg._compile_mode == expected_mode


def test_runtime_config_compile_invalid():
    with pytest.raises(ValueError):
        TorchRuntimeConfig(compile="invalid_mode")


def test_run_model_with_input_dict_as_single_arg():
    class DictArgModel(nn.Module):
        def forward(self, x):
            return x["a"] + x["b"]

    model = DictArgModel()
    out = run_model_with_input(model, {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])})
    assert torch.equal(out, torch.tensor([3.0]))


class _ConstAttrModel(nn.Module):
    """Model with a plain tensor attribute, which torch.export lifts into a constant."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)
        self.offset = torch.arange(4, dtype=torch.float32)

    def forward(self, x):
        return self.fc(x) + self.offset


def test_load_exported_module_puts_constants_on_device(tmp_path):
    program = torch.export.export(_ConstAttrModel().eval(), (torch.randn(2, 4),))
    path = tmp_path / "model.pt2"
    torch.export.save(program, str(path))

    meta = torch.device("meta")
    module = get_model(f"torchexport:{path}", device=meta, runtime_config=TorchRuntimeConfig())

    devices = {t.device for t in list(module.parameters()) + list(module.buffers())}
    devices |= {v.device for sub in module.modules() for v in sub.__dict__.values() if isinstance(v, torch.Tensor)}
    assert devices == {meta}


def test_aoti_load_kwargs_pins_cuda_device_index():
    kwargs = _aoti_load_kwargs(torch.device("cuda:1"), run_single_threaded=True)
    assert kwargs == {"run_single_threaded": True, "device_index": 1}
    # No index and no CUDA -> nothing to pin, and ``None`` values are dropped.
    assert _aoti_load_kwargs(torch.device("cuda"), run_single_threaded=None) == {}
    assert _aoti_load_kwargs(torch.device("cpu"), run_single_threaded=True) == {"run_single_threaded": True}


def test_load_exported_module_runs_on_other_device(tmp_path):
    """A CPU-exported program must run on the benchmark device (constants + baked asserts)."""
    program = torch.export.export(_ConstAttrModel().eval(), (torch.randn(2, 4),))
    path = tmp_path / "model.pt2"
    torch.export.save(program, str(path))

    meta = torch.device("meta")
    module = get_model(f"torchexport:{path}", device=meta, runtime_config=TorchRuntimeConfig())
    out = module(torch.randn(2, 4, device=meta))
    assert out.device == meta
