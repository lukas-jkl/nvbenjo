"""Basic ONNX benchmark with runtime options."""

import os

from nvbenjo import benchmark, cfg

model_path = os.path.expanduser("~/Downloads/resnet50-v2-7.onnx")
if not os.path.isfile(model_path):
    raise SystemExit(
        f"ONNX model not found at {model_path}. Download it with:\n"
        "  wget -O ~/Downloads/resnet50-v2-7.onnx "
        "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx"
    )

model_cfg = cfg.OnnxModelConfig(
    name="resnet50-onnx",
    type_or_path=model_path,
    shape=({"name": "data", "type": "float", "shape": ("B", 3, 224, 224), "min_max": (0, 1)},),
    devices=("cpu",),
    batch_sizes=(1, 8),
    num_warmup_batches=2,
    num_batches=5,
    runtime_options={
        "default": cfg.OnnxRuntimeConfig(
            intra_op_num_threads=2,
            graph_optimization_level="ORT_ENABLE_BASIC",
        ),
    },
)
results = benchmark.benchmark_models({"resnet50": model_cfg})
print(results[["model", "runtime_options", "batch_size", "time_inference"]].to_string())
