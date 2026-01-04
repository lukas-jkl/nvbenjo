import logging
import os
import time
import typing as ty
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Callable, Optional

import pandas as pd
import torch
import torch.nn as nn
import torchvision

from nvbenjo import console
from nvbenjo.cfg import TorchRuntimeConfig
from nvbenjo.utils import AMP_PREFIX, TRANSFER_WARNING, PrecisionType, TensorLike

logger = logging.getLogger(__name__)


def get_model(
    type_or_path: str, device: torch.device, runtime_config: TorchRuntimeConfig, verbose=False, **kwargs
) -> ty.Any:
    """Load PyTorch model.

    Parameters
    ----------
    type_or_path : str
        Model type or path. This can be:

        - a valid path to a saved torch model (saved with torch.save or torch.jit.save)
        - a valid huggingface AutoModel (named 'huggingface:<model-name>') see https://huggingface.co/docs/transformers/model_doc/auto
        - a valid torchvision model (named 'torchvision:<model-name>') see `torchvision.models.list_models()`

    device : torch.device
        Device to load the model onto.
    runtime_config : TorchRuntimeConfig
        Runtime configuration for the model.
    verbose : bool, optional
        Whether to print verbose output, by default False

    Returns
    -------
    ty.Any
        Loaded model.

    Examples
    --------
    >>> model = get_model("torchvision:resnet18", device=torch.device("cpu"), runtime_config=TorchRuntimeConfig())
    >>> model = get_model("/path/to/model.pth", device=torch.device("cuda"), runtime_config=TorchRuntimeConfig())
    >>> model = get_model("/path/to/jitmodel.jit", device=torch.device("cuda"), runtime_config=TorchRuntimeConfig())
    >>> model = get_model("huggingface:bert-base-uncased", device=torch.device("cpu"), runtime_config=TorchRuntimeConfig())
    """
    type_or_path = os.path.expanduser(type_or_path)
    if os.path.isfile(type_or_path):
        if verbose and console is not None:
            console.print(f"Loading torch model {type_or_path}")
        try:
            model = torch.load(type_or_path, map_location=device, weights_only=False)
            model.eval()
            return model
        except Exception:
            try:
                return torch.jit.load(type_or_path, map_location=device)
            except Exception:
                if torch.__version__ > "2.1":
                    program = torch.export.load(type_or_path)
                    module = program.module()
                    module = module.to(device)
                    return module
                else:
                    raise

    if type_or_path.startswith("huggingface:"):
        type_or_path = type_or_path[len("huggingface:") :]
        if verbose and console is not None:
            console.print(f"Loading huggingface model {type_or_path}")
        from transformers import AutoModel  # type: ignore

        return AutoModel.from_pretrained(type_or_path).to(device)
    elif type_or_path.startswith("torchvision:"):
        type_or_path = type_or_path[len("torchvision:") :]
        available_torchvision_models = torchvision.models.list_models()
        if type_or_path in available_torchvision_models:
            if verbose and console is not None:
                console.print(f"Loading torchvision model {type_or_path}")
            return torchvision.models.get_model(type_or_path, **kwargs).to(device)
    else:
        available_torchvision_models = torchvision.models.list_models()
        raise ValueError(
            (
                f"Invalid model {type_or_path}. Must be: \n"
                "- a valid path \n"
                "- a valid huggingface AutoModel (named 'huggingface:<model-name>')  \n"
                f"- a valid torchvision model (named 'torchvision:<model-name>') from {available_torchvision_models} \n"
            )
        )


def run_model_with_input(model: nn.Module, input: TensorLike) -> TensorLike:
    if isinstance(input, (list, tuple)):
        return model(*input)  # type: ignore
    elif isinstance(input, dict):
        return model(**input)  # type: ignore
    else:
        return model(input)  # type: ignore


def transfer_to_device(result: ty.Any, to_device: torch.device) -> ty.Any:
    if hasattr(result, "to"):
        return result.to(to_device)
    if isinstance(result, Sequence):
        return [transfer_to_device(ri, to_device=to_device) for ri in result]
    elif hasattr(result, "items"):
        return {k: transfer_to_device(v, to_device=to_device) for k, v in result.items()}
    else:
        raise ValueError(f"Unsupported result type: {type(result)} could not transfer to {to_device}")


def apply_batch_precision(batch: TensorLike, precision: PrecisionType) -> TensorLike:
    def _apply_batch_precision(batch_tensor: torch.Tensor):
        if AMP_PREFIX not in precision.value:
            if precision == PrecisionType.FP16:
                batch_tensor = batch_tensor.half()
            elif precision == PrecisionType.BFLOAT16:
                batch_tensor = batch_tensor.bfloat16()
            else:
                if precision != PrecisionType.FP32:
                    raise ValueError(f"Invalid precision type {precision}.")
        return batch_tensor

    if isinstance(batch, torch.Tensor):
        batch = _apply_batch_precision(batch)
    elif isinstance(batch, (list, tuple)):
        batch = tuple(_apply_batch_precision(b) for b in batch)
    elif isinstance(batch, dict):
        batch = {k: _apply_batch_precision(v) for k, v in batch.items()}
    else:
        raise ValueError(f"Unsupported batch type: {type(batch)}. Must be a Tensor, Tuple, or Dict.")

    return batch


def apply_non_amp_model_precision(
    model: nn.Module,
    precision: PrecisionType,
) -> nn.Module:
    if AMP_PREFIX not in precision.value:
        if precision == PrecisionType.FP16:
            model = model.half()
        elif precision == PrecisionType.BFLOAT16:
            model = model.bfloat16()
        else:
            if precision != PrecisionType.FP32:
                raise ValueError(f"Invalid precision type {precision}.")

    return model


def get_amp_ctxt_for_precision(precision: PrecisionType, device: torch.device) -> AbstractContextManager:
    if AMP_PREFIX in precision.value:
        valid_values = [PrecisionType.AMP, PrecisionType.AMP_FP16, PrecisionType.AMP_BFLOAT16]
        if precision not in valid_values:
            raise ValueError(f"Invalid AMP precision type {precision} must be one of {valid_values}")

        if precision in [PrecisionType.AMP]:
            ctxt = torch.autocast(device_type=device.type, enabled=True)
        elif precision in [PrecisionType.AMP_FP16]:
            ctxt = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)
        elif precision in [PrecisionType.AMP_BFLOAT16]:
            ctxt = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True)
        else:
            raise ValueError(f"Invalid precision type {precision}.")
    else:
        ctxt = nullcontext()
    return ctxt


def get_model_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_memory_allocation(model: nn.Module, batch: TensorLike, device: torch.device, iterations: int = 3) -> int:
    """Measure the peak memory usage during inference

    Parameters
    ----------
    model : nn.Module
        The model to benchmark.
    batch : TensorLike
        Sample input to the model.
    device : torch.device
        The device where the model is located and shall be used for benchmarking.
    iterations : int, optional
        Number of iterations to run for measuring memory allocation, by default 3

    Returns
    -------
    int
        Maximum memory allocated during inference in bytes.
    """
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)
    # before_run_allocation = torch.cuda.memory_allocated(device=device)

    batch = transfer_to_device(batch, to_device=device)
    model = model.to(device)
    for _ in range(iterations):
        r = run_model_with_input(model, batch)
    try:
        _ = transfer_to_device(r, to_device=torch.device("cpu"))
    except Exception:
        console.print(TRANSFER_WARNING)

    if device.type == "cuda":
        logger.debug(torch.cuda.memory_summary(device=device, abbreviated=True))
        # after_batch_allocation = torch.cuda.memory_allocated(device=device)
        max_batch_allocation = torch.cuda.max_memory_allocated(device=device)
    else:
        max_batch_allocation = -1

    return max_batch_allocation


def measure_repeated_inference_timing(
    model: nn.Module,
    sample: TensorLike,
    batch_size: int,
    model_device: torch.device,
    transfer_to_device_fn: Callable = transfer_to_device,
    num_runs: int = 100,
    progress_callback: Optional[Callable] = None,
) -> pd.DataFrame:
    """Measure inference times.

    Parameters
    ----------
    model : nn.Module
        The model to benchmark.
    sample : TensorLike
        Sample input to the model.
    batch_size : int
        The batch size of the sample.
    model_device : torch.device
        The device where the model is located and shall be used for benchmarking.
    transfer_to_device_fn : Callable, optional
        Function to transfer data to the specified device, by default transfer_to_device
    num_runs : int, optional
        Number of inference runs to perform, by default 100
    progress_callback : Optional[Callable], optional
        Callback function to report progress, by default None

    Returns
    -------
    pd.DataFrame
        DataFrame containing timing results.

    Examples
    --------
    Measure Inference::

        import torch
        from nvbenjo.torch_utils import measure_repeated_inference_timing
        from nvbenjo.torch_utils import get_model
        from nvbenjo.cfg import TorchRuntimeConfig

        model = get_model("torchvision:resnet18", device=torch.device("cpu"), runtime_config=TorchRuntimeConfig())
        sample = torch.randn(2, 3, 224, 224)  # batch size 2
        results = measure_repeated_inference_timing(
            model=model,
            sample=sample,
            batch_size=2,
            model_device=torch.device("cpu"),
            num_runs=2
        )

    """
    time_cpu_to_device = []
    time_inference = []
    time_device_to_cpu = []
    time_total = []
    results_raw = []

    for _ in range(num_runs):
        start_on_cpu = time.perf_counter()
        device_sample = transfer_to_device_fn(sample, model_device)

        if model_device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            stop_event = torch.cuda.Event(enable_timing=True)
            start_event.record()  # For GPU timing
        start_on_device = time.perf_counter()  # For CPU timing

        device_result = run_model_with_input(model, device_sample)

        if model_device.type == "cuda":
            stop_event.record()
            torch.cuda.synchronize()
            # elapsed_on_device = stop_event.elapsed_time(start_event)
            elapsed_on_device = start_event.elapsed_time(stop_event) / 1000.0
            stop_on_device = time.perf_counter()
        else:
            stop_on_device = time.perf_counter()
            elapsed_on_device = stop_on_device - start_on_device

        try:
            transfer_to_device_fn(device_result, torch.device("cpu"))
        except Exception:
            console.print(TRANSFER_WARNING)
        stop_on_cpu = time.perf_counter()

        assert elapsed_on_device > 0

        time_cpu_to_device.append(start_on_device - start_on_cpu)
        time_inference.append(elapsed_on_device)
        time_device_to_cpu.append(stop_on_cpu - stop_on_device)
        time_total.append(stop_on_cpu - start_on_cpu)
        results_raw.append(
            {
                "time_cpu_to_device": start_on_device - start_on_cpu,
                "time_inference": elapsed_on_device,
                "time_device_to_cpu": stop_on_cpu - stop_on_device,
                "time_total": stop_on_cpu - start_on_cpu,
                "time_total_batch_normalized": (stop_on_cpu - start_on_cpu) / batch_size,
            }
        )
        if progress_callback is not None:
            progress_callback()

    results_raw = pd.DataFrame(results_raw)

    return results_raw
