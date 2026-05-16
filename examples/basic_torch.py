"""Basic PyTorch benchmark comparing precision modes."""

import torch

from nvbenjo import benchmark, cfg
from nvbenjo.utils import PrecisionType

device = "cuda" if torch.cuda.is_available() else "cpu"

model_cfg = cfg.TorchModelConfig(
    name="resnet50",
    type_or_path="torchvision:resnet50",
    shape=(("B", 3, 224, 224),),
    devices=(device,),
    batch_sizes=(1, 8),
    num_warmup_batches=2,
    num_batches=5,
    runtime_options={
        "fp32": cfg.TorchRuntimeConfig(precision=PrecisionType.FP32),
    },
)
results = benchmark.benchmark_models({"resnet50": model_cfg})
# results is a pandas DataFrame with latency, throughput, and memory columns
print(results[["model", "runtime_options", "batch_size", "time_inference"]].to_string())
