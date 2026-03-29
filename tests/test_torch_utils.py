from contextlib import nullcontext

import pytest
import torch
from torch import nn

from nvbenjo.cfg import TorchRuntimeConfig
from nvbenjo.torch_utils import (
    apply_batch_precision,
    apply_non_amp_model_precision,
    get_amp_ctxt_for_precision,
    get_model_parameters,
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
