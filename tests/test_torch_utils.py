from contextlib import contextmanager, nullcontext
from unittest.mock import MagicMock

import pytest
import torch
from packaging.version import Version
from torch import nn

from nvbenjo import torch_utils
from nvbenjo.benchmark import _run_warmup
from nvbenjo.cfg import TorchModelConfig, TorchRuntimeConfig
from nvbenjo.torch_utils import (
    _aoti_load_kwargs,
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


class _ConstAttrModel(nn.Module):
    """Plain tensor attribute -> torch.export lifts it into a constant."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)
        self.offset = torch.arange(4, dtype=torch.float32)

    def forward(self, x):
        return self.fc(x) + self.offset


requires_move_to_device_pass = pytest.mark.skipif(
    Version(torch.__version__) < Version("2.5"), reason="move_to_device_pass requires PyTorch 2.5+"
)


@requires_move_to_device_pass
def test_load_exported_module_puts_constants_on_device(tmp_path):
    program = torch.export.export(_ConstAttrModel().eval(), (torch.randn(2, 4),))
    path = tmp_path / "model.pt2"
    torch.export.save(program, str(path))

    meta = torch.device("meta")
    module = get_model(f"torchexport:{path}", device=meta, runtime_config=TorchRuntimeConfig())

    devices = {t.device for t in list(module.parameters()) + list(module.buffers())}
    devices |= {v.device for sub in module.modules() for v in sub.__dict__.values() if isinstance(v, torch.Tensor)}
    assert devices == {meta}


@pytest.mark.skipif(
    Version(torch.__version__) < Version("2.8"), reason="aoti_load_package device_index requires PyTorch 2.8+"
)
def test_aoti_load_kwargs_pins_cuda_device_index():
    kwargs = _aoti_load_kwargs(torch.device("cuda:1"), run_single_threaded=True)
    assert kwargs == {"run_single_threaded": True, "device_index": 1}


def test_aoti_load_kwargs_without_device_index():
    assert _aoti_load_kwargs(torch.device("cuda"), run_single_threaded=None) == {}
    assert _aoti_load_kwargs(torch.device("cpu"), run_single_threaded=True) == {"run_single_threaded": True}


@requires_move_to_device_pass
def test_load_exported_module_runs_on_other_device(tmp_path):
    """A CPU-exported program must run on the benchmark device."""
    program = torch.export.export(_ConstAttrModel().eval(), (torch.randn(2, 4),))
    path = tmp_path / "model.pt2"
    torch.export.save(program, str(path))

    meta = torch.device("meta")
    module = get_model(f"torchexport:{path}", device=meta, runtime_config=TorchRuntimeConfig())
    out = module(torch.randn(2, 4, device=meta))
    assert out.device == meta


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


def test_get_model_torchvision():
    model = get_model(
        "torchvision:resnet18", device=torch.device("cpu"), runtime_config=TorchRuntimeConfig(), weights=None
    )
    assert not model.training

    # an unknown name must fail here, not return None and blow up later at call time
    with pytest.raises(ValueError, match="Invalid torchvision model reznet18"):
        get_model("torchvision:reznet18", device=torch.device("cpu"), runtime_config=TorchRuntimeConfig(), weights=None)


def test_get_model_returns_eval_mode_saved_models(tmp_path):
    runtime_config = TorchRuntimeConfig()
    device = torch.device("cpu")

    torch_path = tmp_path / "model.pth"
    torch.save(_BatchNormModel().train(), torch_path)
    assert not get_model(str(torch_path), device=device, runtime_config=runtime_config).training

    jit_path = tmp_path / "model.jit"
    torch.jit.save(torch.jit.script(_BatchNormModel().train()), jit_path)
    assert not get_model(str(jit_path), device=device, runtime_config=runtime_config).training


@contextmanager
def _recording_device_ctxt(device, entered):
    entered.append(device)
    yield


def test_timing_loop_runs_under_device_ctxt(monkeypatch):
    entered: list[torch.device] = []
    monkeypatch.setattr(torch_utils, "device_ctxt", lambda device: _recording_device_ctxt(device, entered))

    model = nn.Linear(4, 2).eval()
    measure_repeated_inference_timing(
        model, torch.randn(2, 4), batch_size=2, model_device=torch.device("cpu"), num_runs=2
    )

    assert entered == [torch.device("cpu")]


def test_cuda_graphed_model_replays_under_device_ctxt(monkeypatch):
    # replay() takes no stream or device argument, so the captured device has to be made current
    entered: list[torch.device] = []
    monkeypatch.setattr(torch_utils, "device_ctxt", lambda device: _recording_device_ctxt(device, entered))

    graph = MagicMock()
    static_input = torch.zeros(2, 3)
    graphed = torch_utils._CudaGraphedModel(graph, static_input, "captured-output", torch.device("cuda:1"))

    result = graphed(torch.ones(2, 3))

    assert entered == [torch.device("cuda:1")]
    graph.replay.assert_called_once()
    assert result == "captured-output"
    # the input was copied into the captured buffer inside the context, before the replay
    assert torch.equal(static_input, torch.ones(2, 3))


def test_aot_cache_path_distinguishes_matmul_precision(tmp_path):
    model_cfg = TorchModelConfig(name="m", type_or_path="torchvision:resnet18")
    paths = {
        precision: torch_utils._aot_cache_path(
            cache_dir=str(tmp_path),
            model_cfg=model_cfg,
            batch_size=1,
            runtime_cfg=TorchRuntimeConfig(matmul_precision=precision),
            device=torch.device("cpu"),
        )
        for precision in (None, "highest", "high", "medium")
    }

    assert len(set(paths.values())) == len(paths), paths


def test_cuda_graph_capture_rejects_zero_warmup():
    # cuda_graphs + num_warmup_batches=0 is a legal config that used to die inside cuDNN with
    # CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED, aborting every remaining combination
    with pytest.raises(ValueError, match="at least one warm-up iteration"):
        torch_utils._cuda_graph_capture(None, None, torch.device("cuda:0"), num_warmup_iters=0)

