from contextlib import nullcontext

import torch
from torch import nn

from nvbenjo.torch_utils import (
    apply_batch_precision,
    apply_non_amp_model_precision,
    get_amp_ctxt_for_precision,
    get_model_parameters,
)
from nvbenjo.utils import PrecisionType


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
