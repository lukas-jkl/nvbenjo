import hashlib
import json
import logging
import os
import re
import time
import typing as ty
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import torch
import torch.nn as nn
import torchvision
from packaging.version import Version

try:
    # PyTorch's aoti_load_package reaches for torch._inductor.codecache without
    # importing it, so register the attribute up front when available.
    import torch._inductor.codecache  # noqa: F401
except ImportError:
    pass
from rich.progress import Progress

from nvbenjo import console
from nvbenjo.cfg import TorchModelConfig, TorchRuntimeConfig
from nvbenjo.utils import AMP_PREFIX, TRANSFER_WARNING, PrecisionType, TensorLike, progress_task

logger = logging.getLogger(__name__)


def get_model(
    type_or_path: str, device: torch.device, runtime_config: TorchRuntimeConfig, verbose=False, **kwargs
) -> ty.Any:
    """Load PyTorch model.

    Parameters
    ----------
    type_or_path : str
        Model type or path. Supports prefixes to specify the model source:

        - ``torchvision:<name>`` -- Load a torchvision model (e.g. ``torchvision:resnet50``), see `torchvision.models.list_models()`
        - ``huggingface:<name>`` -- Load a HuggingFace AutoModel (e.g. ``huggingface:bert-base-uncased``), see https://huggingface.co/docs/transformers/model_doc/auto
        - ``jit:<path>`` -- Load a TorchScript/JIT model
        - ``torchexport:<path>`` -- Load a ``torch.export`` saved model
        - ``aot:<path>`` -- Load a pre-compiled AOT model
        - *(no prefix)* -- Path to a model saved with ``torch.save`` or ``torch.jit.save``

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
    >>> model = get_model("jit:/path/to/model.pt", device=torch.device("cuda"), runtime_config=TorchRuntimeConfig())
    >>> model = get_model("torchexport:/path/to/model.pt2", device=torch.device("cuda"), runtime_config=TorchRuntimeConfig())
    >>> model = get_model("aot:/path/to/model.pt2", device=torch.device("cuda"), runtime_config=TorchRuntimeConfig())
    >>> model = get_model("huggingface:bert-base-uncased", device=torch.device("cpu"), runtime_config=TorchRuntimeConfig())
    """
    type_or_path = os.path.expanduser(type_or_path)
    if type_or_path.startswith("jit:"):
        if verbose and console is not None:
            console.print(f"Loading jit model {type_or_path}")
        type_or_path = type_or_path[len("jit:") :]
        return torch.jit.load(os.path.expanduser(type_or_path), map_location=device)
    elif type_or_path.startswith("torchexport:"):
        if verbose and console is not None:
            console.print(f"Loading torchexport model {type_or_path}")
        type_or_path = type_or_path[len("torchexport:") :]
        program = torch.export.load(os.path.expanduser(type_or_path))
        module = program.module()
        module = module.to(device)
        return module
    elif type_or_path.startswith("aot:"):
        if verbose and console is not None:
            console.print(f"Loading AOT model {type_or_path}")
        type_or_path = type_or_path[len("aot:") :]
        return torch._inductor.aoti_load_package(os.path.expanduser(type_or_path))
    elif os.path.isfile(type_or_path):
        # Path and no prefix -> try different methods
        if verbose and console is not None:
            console.print(f"Loading torch model {type_or_path}")
        try:
            model = torch.load(os.path.expanduser(type_or_path), map_location=device, weights_only=False)
            model.eval()
            return model
        except Exception:
            try:
                return torch.jit.load(os.path.expanduser(type_or_path), map_location=device)
            except Exception:
                if Version(torch.__version__) > Version("2.1"):
                    try:
                        program = torch.export.load(os.path.expanduser(type_or_path))
                        module = program.module()
                        module = module.to(device)
                        return module
                    except Exception:
                        return torch._inductor.aoti_load_package(os.path.expanduser(type_or_path))
                else:
                    raise

    if type_or_path.startswith("huggingface:"):
        type_or_path = type_or_path[len("huggingface:") :]
        if verbose and console is not None:
            console.print(f"Loading huggingface model {type_or_path}")
        from transformers import AutoModel  # type: ignore

        return AutoModel.from_pretrained(os.path.expanduser(type_or_path)).to(device)
    elif type_or_path.startswith("torchvision:"):
        type_or_path = type_or_path[len("torchvision:") :]
        available_torchvision_models = torchvision.models.list_models()
        if type_or_path in available_torchvision_models:
            if verbose and console is not None:
                console.print(f"Loading torchvision model {type_or_path}")
            model = torchvision.models.get_model(type_or_path, **kwargs).to(device)
            model.eval()
            return model
    else:
        available_torchvision_models = torchvision.models.list_models()
        raise ValueError(
            f"Invalid model {type_or_path}. Must be: \n"
            "- a valid path to a saved torch model\n"
            "- 'jit:<path>' for a TorchScript/JIT model\n"
            "- 'torchexport:<path>' for a torch.export model\n"
            "- 'aot:<path>' for a pre-compiled AOT model\n"
            "- 'huggingface:<model-name>' for a HuggingFace AutoModel\n"
            f"- 'torchvision:<model-name>' from {available_torchvision_models}\n"
        )


def run_model_with_input(model: nn.Module | Callable, input: TensorLike) -> TensorLike:
    if isinstance(input, (list, tuple)):
        return model(*input)
    elif isinstance(input, dict):
        return model(**{str(k): v for k, v in input.items()})
    else:
        return model(input)


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


def measure_memory_allocation(
    model: nn.Module | Callable, batch: TensorLike, device: torch.device, iterations: int = 3
) -> int:
    """Measure the peak memory usage during inference

    Parameters
    ----------
    model : nn.Module | Callable
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
    if isinstance(model, nn.Module):
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


def _file_meta(type_or_path: str) -> Optional[dict]:
    path = type_or_path
    for prefix in ("jit:", "torchexport:", "aot:"):
        if path.startswith(prefix):
            path = path[len(prefix) :]
            break
    path = os.path.expanduser(path)
    if os.path.isfile(path):
        st = os.stat(path)
        return {"name": os.path.basename(path), "size": st.st_size, "mtime": st.st_mtime}
    return None


def _aot_cache_path(
    cache_dir: str,
    model_cfg: TorchModelConfig,
    batch_size: int,
    runtime_cfg: TorchRuntimeConfig,
    device: torch.device,
) -> Path:
    key_parts: dict = {
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "type_or_path": model_cfg.type_or_path,
        "model_kwargs": sorted(model_cfg.kwargs.items()),
        "file_meta": _file_meta(model_cfg.type_or_path),
        "shape": list(model_cfg.shape),
        "batch_size": batch_size,
        "precision": runtime_cfg.precision.value,
        "compile_kwargs": sorted((k, v) for k, v in runtime_cfg.compile_kwargs.items() if k != "package_path"),
        "device_type": device.type,
    }
    if device.type == "cuda":
        key_parts["sm"] = torch.cuda.get_device_capability(device)
        key_parts["device_name"] = torch.cuda.get_device_name(device)
    digest = hashlib.sha256(json.dumps(key_parts, default=str, sort_keys=True).encode()).hexdigest()[:16]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", model_cfg.name)
    return Path(cache_dir).expanduser() / f"{safe_name}_{digest}.pt2"


def _export_program(model: Any, batch: TensorLike, device: torch.device) -> Any:
    if not isinstance(model, nn.Module):
        return model.to(device)
    device_batch = transfer_to_device(batch, device)
    if isinstance(device_batch, dict):
        return torch.export.export(model.to(device), args=(), kwargs=device_batch)
    if isinstance(device_batch, (tuple, list)):
        batch_args = tuple(device_batch)
    else:
        batch_args = (device_batch,)
    return torch.export.export(model.to(device), batch_args)


def _aot_compile_or_load(
    model: Any,
    batch: TensorLike,
    device: torch.device,
    model_cfg: TorchModelConfig,
    batch_size: int,
    runtime_cfg: TorchRuntimeConfig,
    progress_bar: Optional[Progress],
) -> Any:
    cache_path = (
        _aot_cache_path(runtime_cfg.cache_dir, model_cfg, batch_size, runtime_cfg, device)
        if runtime_cfg.cache_dir
        else None
    )

    if cache_path is not None and cache_path.exists():
        with progress_task(progress_bar, f"    Load AOT compiled model {cache_path}...", total=None):
            try:
                return torch._inductor.aoti_load_package(str(cache_path))
            except Exception:
                console.print(f"Failed to load AOT cache {cache_path}, falling back to recompile")
                console.print_exception()

    program = _export_program(model, batch, device)
    program = program.run_decompositions()

    compile_kwargs = dict(runtime_cfg.compile_kwargs)
    if cache_path is not None:
        if "package_path" in compile_kwargs:
            raise ValueError("Cannot set both runtime_config.cache_dir and compile_kwargs['package_path']")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Compile to a sibling tmp path, then atomic-rename. Keep the .pt2 suffix
        # because aoti_compile_and_package validates package_path ends in .pt2.
        tmp_path = cache_path.with_name(f"{cache_path.stem}.tmp{cache_path.suffix}")
        compile_kwargs["package_path"] = str(tmp_path)
        with progress_task(progress_bar, "    AOT compiling...", total=None):
            torch._inductor.aoti_compile_and_package(program, **compile_kwargs)
        os.replace(tmp_path, cache_path)
        return torch._inductor.aoti_load_package(str(cache_path))
    else:
        with progress_task(progress_bar, "    AOT compiling...", total=None):
            package_path = torch._inductor.aoti_compile_and_package(program, **compile_kwargs)
        return torch._inductor.aoti_load_package(package_path)
