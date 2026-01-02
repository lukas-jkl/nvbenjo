from nvbenjo import cfg
from nvbenjo.utils import PrecisionType
from nvbenjo import benchmark


def test_pytorch_simple():
    model_cfg = cfg.TorchModelConfig(
        name="torch-shufflenet-v2-x0-5",
        type_or_path="torchvision:shufflenet_v2_x0_5",
        shape=(("B", 3, 224, 224),),
        devices=["cpu"],
        batch_sizes=[1],
        num_warmup_batches=1,
        num_batches=2,
        runtime_options={
            "test1": cfg.TorchRuntimeConfig(compile=False, precision=PrecisionType.FP32),
        },
    )
    results = benchmark.benchmark_models({"model_1": model_cfg})
    assert not results.empty
    assert "model_1" in results.model.to_numpy()
    assert "test1" in results.runtime_options.to_numpy()
