# Configuration

Nvbenjo uses [Hydra](https://hydra.cc/) for configuration using the dataclasses listed below which you may use with the [Python API](python_api.md). 
See [Examples](examples.md) for configuration file examples to use with the command line interface.

## Main configuration classes

```{eval-rst}
.. autoclass:: nvbenjo.cfg.BenchConfig
   :members:
   
.. autoclass:: nvbenjo.cfg.NvbenjoConfig
   :members:
```

## Pytorch

```{eval-rst}
.. autoclass:: nvbenjo.cfg.TorchModelConfig
   :members:

.. autoclass:: nvbenjo.cfg.TorchRuntimeConfig
   :members:

```

## Onnx

```{eval-rst}
.. autoclass:: nvbenjo.cfg.OnnxModelConfig
   :members:
   
.. autoclass:: nvbenjo.cfg.OnnxRuntimeConfig
   :members:
```