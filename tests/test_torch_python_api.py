import os
from tempfile import TemporaryDirectory

from nvbenjo import benchmark, cfg
from nvbenjo.utils import PrecisionType


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
