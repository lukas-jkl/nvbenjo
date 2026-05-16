# Python API Reference

## Benchmark Module

```{eval-rst}
.. autofunction:: nvbenjo.benchmark.benchmark_models

.. autofunction:: nvbenjo.benchmark.load_model
```

## PyTorch Utilities

```{eval-rst}
.. autofunction:: nvbenjo.torch_utils.get_model

.. autofunction:: nvbenjo.torch_utils.measure_gpu_memory_allocation

.. autofunction:: nvbenjo.torch_utils.measure_repeated_inference_timing
```

## ONNX Utilities

```{eval-rst}
.. autofunction:: nvbenjo.onnx_utils.get_model

.. autofunction:: nvbenjo.onnx_utils.measure_gpu_memory_allocation

.. autofunction:: nvbenjo.onnx_utils.measure_repeated_inference_timing
```

## System Information

```{eval-rst}
.. autofunction:: nvbenjo.system_info.get_system_info

.. autofunction:: nvbenjo.system_info.get_gpu_info
```

## Examples

### PyTorch

```{eval-rst}
.. literalinclude:: ../examples/basic_torch.py
   :language: python
   :caption: Basic PyTorch benchmark
```

### ONNX

```{eval-rst}
.. literalinclude:: ../examples/basic_onnx.py
   :language: python
   :caption: Basic ONNX benchmark
```