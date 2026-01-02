## Configuration

Nvbenjo uses [Hydra](https://hydra.cc/) for configuration using the dataclasses listed below. 

### Main configuration classes

```{eval-rst}
.. autoclass:: nvbenjo.cfg.BenchConfig
   :members:
   
.. autoclass:: nvbenjo.cfg.NvbenjoConfig
   :members:
```

### Pytorch

```{eval-rst}
.. autoclass:: nvbenjo.cfg.TorchModelConfig
   :members:

.. autoclass:: nvbenjo.cfg.TorchRuntimeConfig
   :members:

```

### Onnx

```{eval-rst}
.. autoclass:: nvbenjo.cfg.OnnxModelConfig
   :members:
   
.. autoclass:: nvbenjo.cfg.OnnxRuntimeConfig
   :members:
```