# Nvbenjo Documentation

Nvbenjo is a utility for benchmarking deep learning models on NVIDIA GPUs.
It supports models in [Onnx](https://onnx.ai/) format as well as [PyTorch](https://pytorch.org/) models.

```{toctree}
:maxdepth: 2
:caption: Contents

self
configuration
python_api
```

(usage)=
## Usage
Nvbenjo can be used as command line tool and uses [hydra](https://hydra.cc/) configuration. 

Specify configuration directly from the command line:
```bash
nvbenjo \
"+nvbenjo.models={\
    efficientnet: {type_or_path: 'torchvision:efficientnet_b0',  shape:['B',3,224,224],  batch_sizes: [16,32]},\
    resnet:       {type_or_path: 'torchvision:wide_resnet101_2', shape: ['B',3,224,224], batch_sizes: [16,32]}\
}"
```

### Usage with Config File

Or better, specify your own config (or one of the pre-defined config files)
```
nvbenjo -cn small
nvbenjo -cn="/my/config/path/myconfig.yaml"
```

Override single arguments of your config
```
nvbenjo -cn="/my/config/path/myconfig.yaml" nvbenjo.models.mymodel.num_batches=10
```


```{eval-rst}
.. literalinclude:: ../src/nvbenjo/conf/example.yaml
   :language: yaml
   :lines: 1-45
   :caption: Example Configuration
```

### Usage with Python API

See the [Python API Reference](python_api.md) for detailed documentation of all available functions and classes and [Configuration Reference](configuration.md) for config objects.


```python
from nvbenjo import cfg
from nvbenjo.utils import PrecisionType
from nvbenjo import benchmark


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
```