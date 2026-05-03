import os
from tempfile import TemporaryDirectory

import pytest
import torch
import torch.nn as nn

from nvbenjo import benchmark, cfg
from nvbenjo.utils import PrecisionType


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 8)

    def forward(self, x):
        return self.fc(x)


def test_pytorch_simple():
    with TemporaryDirectory() as tmpdir:
        num_batches = 2
        model_cfg = cfg.TorchModelConfig(
            name="torch-shufflenet-v2-x0-5",
            type_or_path="torchvision:shufflenet_v2_x0_5",
            shape=(("B", 3, 224, 224),),
            devices=["cpu"],
            batch_sizes=[1],
            num_warmup_batches=1,
            num_batches=num_batches,
            runtime_options={
                "test1": cfg.TorchRuntimeConfig(
                    compile=False,
                    precision=PrecisionType.FP32,
                    enable_profiling=True,
                    profiling_prefix=os.path.join(tmpdir, "profile_"),
                    profiler_kwargs={"profile_memory": True, "record_shapes": True},
                ),
            },
            custom_batchmetrics={
                "fps": 1.0,
            },
        )
        results = benchmark.benchmark_models({"model_1": model_cfg})
        assert not results.empty
        assert "model_1" in results.model.to_numpy()
        assert "test1" in results.runtime_options.to_numpy()
        assert len(results.time_inference.to_numpy()) == num_batches

        # Check that profiling files were created
        profile_files = os.listdir(tmpdir)
        assert len(profile_files) == 1
        assert profile_files[0].startswith("profile_")
        assert profile_files[0].endswith(".json")


@pytest.mark.skipif(torch.__version__ < "2.6", reason="aoti_compile_and_package requires PyTorch 2.6+")
def test_aot_cache_skips_recompile(tmp_path):
    torch.manual_seed(0)
    model_path = tmp_path / "tiny.pt"
    torch.save(_Tiny(), model_path)
    cache_dir = tmp_path / "cache"

    def make_cfg(batch_size: int) -> cfg.TorchModelConfig:
        return cfg.TorchModelConfig(
            name="aot-cache-tiny",
            type_or_path=str(model_path),
            shape=(("B", 16),),
            devices=["cpu"],
            batch_sizes=[batch_size],
            num_warmup_batches=1,
            num_batches=1,
            runtime_options={
                "aot": cfg.TorchRuntimeConfig(
                    compile="aot_compile",
                    precision=PrecisionType.FP32,
                    cache_dir=str(cache_dir),
                ),
            },
        )

    # First run: compile + cache write
    benchmark.benchmark_models({"model_1": make_cfg(1)})
    files = sorted(os.listdir(cache_dir))
    assert len(files) == 1, f"expected 1 cached package, got {files}"
    cache_file = cache_dir / files[0]
    first_mtime = cache_file.stat().st_mtime
    firs    t_size = cache_file.stat().st_size

    # Second run with identical config: cache hit, file untouched
    benchmark.benchmark_models({"model_1": make_cfg(1)})
    assert sorted(os.listdir(cache_dir)) == files
    assert cache_file.stat().st_mtime == first_mtime
    assert cache_file.stat().st_size == first_size

    # Different batch_size → different key → second cache file appears
    benchmark.benchmark_models({"model_1": make_cfg(2)})
    assert len(os.listdir(cache_dir)) == 2
