import glob
import json
import os

import pytest
import torch
from torch import nn

from nvbenjo import benchmark, cfg, torch_utils
from nvbenjo.utils import PrecisionType

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="precision options are CUDA-only")
requires_tf32 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="TF32 tensor cores require compute capability 8.0+",
)


@requires_tf32
@pytest.mark.parametrize(
    ("matmul_precision", "uses_tf32"),
    [("highest", False), ("high", True), ("medium", True)],
)
def test_matmul_precision_ctxt_switches_tf32(matmul_precision, uses_tf32):
    a = torch.randn(256, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    reference = a.double() @ b.double()

    with torch_utils.matmul_precision_ctxt(matmul_precision):
        result = a @ b

    error = ((result.double() - reference).abs().max() / reference.abs().max()).item()
    assert (error > 1e-5) == uses_tf32, f"{matmul_precision} gave {error:.2e}"


@requires_cuda
@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        (PrecisionType.AMP, torch.float16),
        (PrecisionType.AMP_FP16, torch.float16),
        pytest.param(
            PrecisionType.AMP_BFLOAT16,
            torch.bfloat16,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
                reason="bfloat16 autocast needs hardware support",
            ),
        ),
        (PrecisionType.FP32, torch.float32),
    ],
)
def test_amp_precision_runs_in_expected_dtype(precision, expected_dtype):
    device = torch.device("cuda")
    model = nn.Linear(64, 64).to(device).eval()
    batch = torch.randn(8, 64, device=device)

    with torch.no_grad(), torch_utils.get_amp_ctxt_for_precision(precision=precision, device=device):
        output = model(batch)

    assert output.dtype == expected_dtype


@requires_tf32
def test_matmul_precision_reaches_the_benchmarked_kernels(tmp_path):
    def profiled_kernels(matmul_precision: str) -> set[str]:
        prefix = os.path.join(tmp_path, matmul_precision)
        model_path = os.path.join(tmp_path, "model.pt")
        torch.save(nn.Linear(1024, 1024), model_path)
        model_cfg = cfg.TorchModelConfig(
            name="linear",
            type_or_path=model_path,
            shape=(("B", 1024),),
            devices=["cuda:0"],
            batch_sizes=[256],
            num_warmup_batches=1,
            num_batches=2,
            runtime_options={
                "rt": cfg.TorchRuntimeConfig(
                    compile=False,
                    precision=PrecisionType.FP32,
                    matmul_precision=matmul_precision,
                    enable_profiling=True,
                    profiling_prefix=prefix,
                )
            },
        )
        benchmark.benchmark_models({"model": model_cfg})

        traces = glob.glob(f"{prefix}*.json")
        assert traces, f"enable_profiling wrote no trace for {matmul_precision}"

        kernels = set()
        for trace in traces:
            with open(trace) as fh:
                for event in json.load(fh).get("traceEvents", []):
                    if event.get("cat", "").lower() == "kernel":
                        kernels.add(event["name"])
        return kernels

    fp32_kernels = profiled_kernels("highest")
    tf32_kernels = profiled_kernels("medium")

    if not fp32_kernels or not tf32_kernels:
        pytest.skip("profiler captured no CUDA kernels")

    assert fp32_kernels.isdisjoint(tf32_kernels), f"{fp32_kernels} vs {tf32_kernels}"
