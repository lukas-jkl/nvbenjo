from contextlib import nullcontext

import torch
import torch.nn as nn

from nvbenjo.utils import (
    PrecisionType,
    apply_non_amp_model_precision,
    format_num,
    format_seconds,
    get_amp_ctxt_for_precision,
    get_model_parameters,
)


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
    model, batch = apply_non_amp_model_precision(model, batch, PrecisionType.FP16)
    assert model.fc.weight.dtype == torch.float16
    assert batch.dtype == torch.float16

    model = SimpleModel()
    batch = torch.randn(10, 10)
    model, batch = apply_non_amp_model_precision(model, batch, PrecisionType.FP32)
    assert model.fc.weight.dtype == torch.float32
    assert batch.dtype == torch.float32

    model = SimpleModel()
    batch = torch.randn(10, 10)
    model, batch = apply_non_amp_model_precision(model, batch, PrecisionType.BFLOAT16)
    assert model.fc.weight.dtype == torch.bfloat16
    assert batch.dtype == torch.bfloat16

    model = SimpleModel()
    batch = torch.randn(10, 10)
    model, batch = apply_non_amp_model_precision(model, batch, PrecisionType.AMP_FP16)
    # only shall apply non-amp precisions
    assert model.fc.weight.dtype == torch.float32
    assert batch.dtype == torch.float32


def test_get_amp_ctxt_for_precision():
    ctxt = get_amp_ctxt_for_precision(PrecisionType.AMP, torch.device("cpu"))
    assert isinstance(ctxt, torch.autocast)

    ctxt = get_amp_ctxt_for_precision(PrecisionType.FP32, torch.device("cpu"))
    assert isinstance(ctxt, nullcontext)
