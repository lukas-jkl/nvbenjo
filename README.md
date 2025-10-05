# Nvbenjo

[![Nox](https://img.shields.io/badge/%F0%9F%A6%8A-Nox-D85E00.svg)](https://github.com/wntrblm/nox)
![Ruff](https://github.com/lukas-jkl/nvbenjo/actions/workflows/ruff.yml/badge.svg)
![Tests](https://github.com/lukas-jkl/nvbenjo/actions/workflows/test.yml/badge.svg)

A tool for evaluating system performance against pytorch models. 


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
# if the config is in the same working directory
nvbenjo -cn="myconfig.yaml"
# Otherwise specify the directory and config file
#TODO: this is an annoying limitation of hydra we can prob. hack this into a sinlge arg by patching main?
nvbenjo -cd="/my/config/path" -cn="myconfig.yaml" 

# override single arguments of your config
nvbenjo -cd="/my/config/path" -cn="myconfig.yaml" nvbenjo.models.mymodel.num_batches=10
```


## Development

Example using uv:

```bash
uv sync --extra dev
uv run main.py

# for a quick run on CPU
uv run main.py -cn small
```