from contextlib import nullcontext

import pytest
import torch
from torch import nn

from nvbenjo.benchmark import _run_warmup
from nvbenjo.cfg import TorchRuntimeConfig
from nvbenjo.torch_utils import (
    apply_batch_precision,
    apply_non_amp_model_precision,
    get_amp_ctxt_for_precision,
    get_model,
    get_model_parameters,
    measure_gpu_memory_allocation,
    measure_repeated_inference_timing,
    run_model_with_input,
)
from nvbenjo.utils import CompileMode, PrecisionType


def test_get_model_parameters():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10, bias=False)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    num_params = get_model_parameters(model)
    assert num_params == 100


def test_apply_non_amp_model_precision():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
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
            super().__init__()
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


class _GradProbe(nn.Module):
    """Records whether autograd was active for each forward pass."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.grad_enabled = []
        self.output_requires_grad = []

    def forward(self, x):
        out = self.fc(x)
        self.grad_enabled.append(torch.is_grad_enabled())
        self.output_requires_grad.append(out.requires_grad)
        return out


def test_inference_runs_without_autograd():
    device = torch.device("cpu")
    batch = torch.randn(4, 10)

    model = _GradProbe()
    # parameters require grad, so without a no_grad guard every forward builds a graph
    assert all(p.requires_grad for p in model.parameters())

    _run_warmup(model, batch, device, num_warmup_batches=2, progress_bar=None)
    measure_gpu_memory_allocation(model, batch, device, iterations=2)
    measure_repeated_inference_timing(model, batch, batch_size=4, model_device=device, num_runs=2)

    assert len(model.grad_enabled) == 6
    assert not any(model.grad_enabled)
    assert not any(model.output_requires_grad)


class _BatchNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)

    def forward(self, x):
        return self.bn(x)


def test_get_model_returns_eval_mode_torchvision():
    model = get_model(
        "torchvision:resnet18", device=torch.device("cpu"), runtime_config=TorchRuntimeConfig(), weights=None
    )
    assert not model.training


def test_get_model_returns_eval_mode_saved_models(tmp_path):
    runtime_config = TorchRuntimeConfig()
    device = torch.device("cpu")

    torch_path = tmp_path / "model.pth"
    torch.save(_BatchNormModel().train(), torch_path)
    assert not get_model(str(torch_path), device=device, runtime_config=runtime_config).training

    jit_path = tmp_path / "model.jit"
    torch.jit.save(torch.jit.script(_BatchNormModel().train()), jit_path)
    assert not get_model(str(jit_path), device=device, runtime_config=runtime_config).training
