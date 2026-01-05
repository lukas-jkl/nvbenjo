# Nvbenjo

[![Nox](https://img.shields.io/badge/%F0%9F%A6%8A-Nox-D85E00.svg)](https://github.com/wntrblm/nox)
![Ruff](https://github.com/lukas-jkl/nvbenjo/actions/workflows/ruff.yml/badge.svg)
![Tests](https://github.com/lukas-jkl/nvbenjo/actions/workflows/test.yml/badge.svg)

Nvbenjo is a utility for benchmarking inference of deep learning models on NVIDIA GPUs.
It supports models in [Onnx](https://onnx.ai/) format as well as [PyTorch](https://pytorch.org/) models.
Nvbenjo generates comprehensive benchmark results including:
- **CSV file** with all measurement data (latency, throughput, memory usage, etc.)
- **Plots** visualizing the benchmark results

## Demo
![Demo](./assets/demo.gif)

## Installing

```bash
pip install nvbenjo
```

If you need a specific version of PyTorch or want to benchmark Onnx models adapt your install:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install onnx onnxruntime-gpu
pip install nvbenjo
```

## Usage

```bash
# Specify models to run in the command line
nvbenjo \
"+nvbenjo.models={\
    efficientnet: {type_or_path: 'torchvision:efficientnet_b0',  shape:['B',3,224,224],  batch_sizes: [16,32]},\
    resnet:       {type_or_path: 'torchvision:wide_resnet101_2', shape: ['B',3,224,224], batch_sizes: [16,32]}\
}"

# or better, specify your own config (or one of the pre-defined config files)
nvbenjo -cn small
nvbenjo -cn="/my/config/path/myconfig.yaml"

# override single arguments of your config
nvbenjo -cn="/my/config/path/myconfig.yaml" nvbenjo.models.mymodel.num_batches=10

# show current config and help
nvbenjo -cn="/my/config/path/myconfig.yaml" --help
```

Take a look at the [Documentation](https://nvbenjo.readthedocs.io) for more information as well as example configurations.

## Development

Example using uv:

```bash
uv sync --extra dev --extra onnx-cpu # or gpu
uv run nvbenjo

# for a quick run
uv run nvbenjo -cn small

# tests
uv run pytest
uv run nox
```
