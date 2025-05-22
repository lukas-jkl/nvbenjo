# Nvbenjo

![Ruff](https://github.com/lukas-jkl/nvbenjo/actions/workflows/ruff.yml/badge.svg)
![Tests](https://github.com/lukas-jkl/nvbenjo/actions/workflows/test.yml/badge.svg)

A tool for evaluating system performance against pytorch models. 


## Usage

```bash
nvbenjo

# with custom output directory
nvbenjo output_dir=testout

# override single arguments
nvbenjo output_dir=testout nvbenjo.models.0.num_batches=10

# custom model configuration
nvbenjo 'nvbenjo.models=['\
'   {name: "efficientnet", type_or_path: "efficientnet_b0", shape: ["B", 3, 224, 224], batch_sizes: [16, 32], precisions: [FP32, FP16]},'\
'   {name: "custommodel", type_or_path: "wide_resnet101_2", shape: ["B", 3, 224, 224], batch_sizes: [16, 32], precisions: [AMP, FP16]}'\
']'

nvbenjo 'nvbenjo.models=['\
'   {name: "efficientnet", type_or_path: "efficientnet_b0", shape: [["B", 3, 224, 224], ["B", 3, 224, 224]], batch_sizes: [16, 32], precisions: [FP32, FP16]},'\
']'
# or specify your own config
nvbenjo -cn customconfig
```


## Development

Example using uv:

```bash
uv sync --extra dev
uv run main.py

# for a quick run on CPU
uv run main.py -cn small
```