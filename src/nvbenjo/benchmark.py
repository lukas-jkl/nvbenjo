import itertools
import logging
import time
import typing as ty
from typing import Any, Callable, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rich import progress
from rich.progress import Progress

import nvbenjo.utils as utils
from nvbenjo import console, torch_utils
from nvbenjo.cfg import BaseModelConfig, TorchModelConfig, OnnxModelConfig, TorchRuntimeConfig, OnnxRuntimeConfig

logger = logging.getLogger(__name__)


def load_model(
    type_or_path: str, device: torch.device, runtime_config: TorchRuntimeConfig | OnnxRuntimeConfig, **kwargs
) -> Any:
    """Load a model, may be a PyTorch or ONNX model based on the runtime configuration.

    Parameters
    ----------
    type_or_path : str
        String specifying the model type or path
    device : torch.device
        Device to load the model onto
    runtime_config : :class:`~nvbenjo.cfg.TorchRuntimeConfig` or :class:`~nvbenjo.cfg.OnnxRuntimeConfig`
        Runtime configuration for the model

    Returns
    -------
    Any
        Loaded model instance
    """
    match runtime_config:
        case OnnxRuntimeConfig():
            from nvbenjo import onnx_utils

            return onnx_utils.get_model(type_or_path, device=device, runtime_config=runtime_config, **kwargs)
        case TorchRuntimeConfig():
            return torch_utils.get_model(type_or_path, device=device, runtime_config=runtime_config, **kwargs)
        case _:
            raise ValueError(f"Unknown runtime config type {type(runtime_config)}")


def _test_load_models(model_cfgs: Dict[str, BaseModelConfig]) -> None:
    loaded_types = []
    logger.info("Checking if models are valid and available")
    for _, model_cfg in model_cfgs.items():
        if model_cfg.type_or_path not in loaded_types:
            initial_runtime_options = list(model_cfg.runtime_options.values())[0]
            _ = load_model(
                model_cfg.type_or_path,
                device=torch.device("cpu"),
                runtime_config=initial_runtime_options,
                **model_cfg.kwargs,
            )
            loaded_types.append(model_cfg.type_or_path)


def benchmark_models(model_cfgs: Dict[str, BaseModelConfig], measure_memory: Optional[bool] = True) -> pd.DataFrame:
    """Benchmark the given models.

    Parameters
    ----------
    model_cfgs : Dict[str, :class:`~nvbenjo.cfg.TorchModelConfig` | :class:`~nvbenjo.cfg.OnnxModelConfig`]
    measure_memory : bool, optional
        Whether to measure memory usage during benchmarking, by default True

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the benchmarking results


    Examples
    --------
    Basic usage with single PyTorch model::

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
            custom_batchmetrics={
                "fps": 1.0,
            },
        )
        results = benchmark.benchmark_models({"model_1": model_cfg})
    """
    _test_load_models(model_cfgs)

    with _get_progress_bar() as progress_bar:
        model_task = progress_bar.add_task("Benchmarking models", total=len(model_cfgs))
        results = []

        for model_name, model_cfg in model_cfgs.items():
            progress_bar.update(model_task, description=f"Benchmarking {model_name}")
            model_results = benchmark_model(model_cfg, progress_bar=progress_bar, measure_memory=measure_memory)
            model_results["model"] = model_name
            if model_cfg.custom_batchmetrics:
                model_results = utils.calculate_batchmetrics(model_results, model_cfg.custom_batchmetrics)
            results.append(model_results)
            progress_bar.advance(model_task)

        results = pd.concat(results)

    return results


def _get_progress_bar() -> Progress:
    return Progress(
        "[progress.description]{task.description}",
        progress.BarColumn(bar_width=80),
        "[progress.percentage]{task.completed}/{task.total}",
        console=console,
    )


def _run_warmup(
    model: nn.Module,
    batch: utils.TensorLike,
    device: torch.device,
    num_warmup_batches: int,
    progress_bar: Optional[Progress],
):
    try:
        if progress_bar is not None:
            warm_up_task = progress_bar.add_task("    Warm-up", total=num_warmup_batches)

        for _ in range(num_warmup_batches):
            batch = torch_utils.transfer_to_device(batch, device)
            r = torch_utils.run_model_with_input(model, batch)
            try:
                _ = torch_utils.transfer_to_device(r, to_device=torch.device("cpu"))
            except Exception:
                console.print(utils.TRANSFER_WARNING)

            if progress_bar is not None:
                progress_bar.advance(warm_up_task)

    finally:
        if progress_bar is not None:
            progress_bar.remove_task(warm_up_task)


def _measure_timings(
    model: Any,
    batch: utils.TensorLike,
    batch_size: int,
    device: torch.device,
    num_batches: int,
    progress_bar: Optional[Progress],
    timing_function: Callable = torch_utils.measure_repeated_inference_timing,
    profiler: Optional[torch.profiler.profile] = None,
) -> pd.DataFrame:
    if progress_bar is not None:
        measure_task = progress_bar.add_task(
            "    Inference",
            total=num_batches,
        )

    def progress_callback():
        if progress_bar is not None:
            progress_bar.advance(measure_task)
        else:
            pass
        if profiler is not None:
            profiler.step()

    try:
        cur_raw_results = timing_function(
            model,
            batch,
            batch_size,
            device,
            num_runs=num_batches,
            progress_callback=progress_callback,
        )
    finally:
        if progress_bar is not None:
            progress_bar.remove_task(measure_task)
    return cur_raw_results


def _get_device(runtime_config: OnnxRuntimeConfig | TorchRuntimeConfig, device: str, console) -> torch.device:
    device_chosen = torch.device(device)
    match device_chosen.type:
        case "cpu":
            match runtime_config:
                case TorchRuntimeConfig():
                    return device_chosen
                case OnnxRuntimeConfig():
                    # If CUDA is available and the execution provider is set to CUDA or TensorRT, use CUDA
                    from nvbenjo import onnx_utils

                    available_providers = onnx_utils.ort.get_available_providers()  # type: ignore
                    if (
                        torch.cuda.is_available()
                        and (
                            "CUDAExecutionProvider" in available_providers
                            or "TensorrtExecutionProvider" in available_providers
                        )
                        # Check for
                        # [("CUDAExecutionProvider", {"device_id": 0})] style entries
                        and runtime_config.execution_providers
                        and any(
                            provider in ["CUDAExecutionProvider", "TensorrtExecutionProvider"]
                            or provider is not None
                            and provider[0] in ["CUDAExecutionProvider", "TensorrtExecutionProvider"]
                            for provider in runtime_config.execution_providers  # type: ignore
                        )
                    ):
                        device_id = 0
                        for provider in runtime_config.execution_providers:  # type: ignore
                            if len(provider) == 2 and provider[0] in [
                                "CUDAExecutionProvider",
                                "TensorrtExecutionProvider",
                            ]:
                                device_id = provider[1].get("device_id", 0)  # type: ignore
                        device_chosen = torch.device(f"cuda:{device_id}")
            return device_chosen
        case "cuda":
            match runtime_config:
                case TorchRuntimeConfig():
                    if torch.cuda.is_available():
                        return device_chosen
                    else:
                        console.print("[yellow]CUDA is not available. Running on CPU.[/yellow]")
                        return torch.device("cpu")

                case OnnxRuntimeConfig():
                    from nvbenjo import onnx_utils

                    available_providers = onnx_utils.ort.get_available_providers()  # type: ignore
                    if torch.cuda.is_available() and (
                        "CUDAExecutionProvider" in available_providers
                        or "TensorrtExecutionProvider" in available_providers
                    ):
                        return device_chosen
                    else:
                        if torch.cuda.is_available():
                            console.print(
                                "[yellow]CUDAExecutionProvider is not available in onnxruntime. Running on CPU.[/yellow]"
                            )
                        else:
                            console.print("[yellow]CUDA is not available. Running on CPU.[/yellow]")
                        return torch.device("cpu")
                case _:
                    raise ValueError(f"Unknown runtime config type {type(runtime_config)}")
        case _:
            raise ValueError(f"Invalid device {device}. Must be one of cpu or cuda")

    return device_chosen


# TODO: !! two seperate functions torch and onnx
def benchmark_model(
    model_cfg: BaseModelConfig,
    measure_memory: Optional[bool] = True,
    progress_bar: Optional[Progress] = None,
) -> pd.DataFrame:
    """Benchmark one model configuration.

    Parameters
    ----------
    model_cfg : :class:`~nvbenjo.cfg.TorchModelConfig` or :class:`~nvbenjo.cfg.OnnxModelConfig`
        Model configuration to benchmark
    measure_memory : bool, optional
        Whether to measure memory usage during benchmarking, by default True
    progress_bar : Progress, optional
        Progress bar instance to display benchmarking progress, by default None

    Returns
    -------
    pd.DataFrame
        Benchmarking results as a pandas DataFrame
    """
    results = []
    num_model_parameters = None
    precision_batch_oom = {}

    if progress_bar is None:
        progress_bar = _get_progress_bar()
    console = progress_bar.console

    iter_cfgs = list(itertools.product(*[model_cfg.devices, model_cfg.batch_sizes, model_cfg.runtime_options.items()]))
    bench_task = progress_bar.add_task("Running Benchmark", total=len(iter_cfgs))
    for device, batch_size, (runtime_option_name, runtime_cfg) in iter_cfgs:
        if precision_batch_oom.get(runtime_option_name, np.inf) < batch_size:
            # already went oom for these runtime options with smaller batch size -> skip bigger one
            progress_bar.advance(bench_task)
            continue
        try:
            device = _get_device(runtime_cfg, device, console)
            progress_bar.update(
                bench_task, description=f"  Device {device} | batch-size: {batch_size} | {runtime_option_name}"
            )

            model = load_model(model_cfg.type_or_path, device=device, runtime_config=runtime_cfg, **model_cfg.kwargs)
            if isinstance(model_cfg, TorchModelConfig):
                batch: utils.TensorLike
                batch, set_dtype = utils.get_rnd_from_shape_s(shape=model_cfg.shape, batch_size=batch_size)

                if num_model_parameters is None:
                    num_model_parameters = torch_utils.get_model_parameters(model)

                model = torch_utils.apply_non_amp_model_precision(
                    model, precision=model_cfg.runtime_options[runtime_option_name].precision
                )
                if runtime_cfg.compile:
                    model = torch.compile(model, **runtime_cfg.compile_kwargs)

                if not isinstance(model, nn.Module):
                    raise ValueError(f"Expected a torch.nn.Module but got {type(model)}")

                # only apply precision to input if no precision is specified
                if not set_dtype:
                    batch = torch_utils.apply_batch_precision(batch, precision=runtime_cfg.precision)
                else:
                    if not isinstance(batch, dict):
                        raise ValueError(f"Batch {batch} must be a dict if set_dtype is used.")
                    batch = ty.cast(
                        dict[str, torch.Tensor],
                        {
                            k: (
                                torch_utils.apply_batch_precision(
                                    ty.cast(torch.Tensor, v), precision=runtime_cfg.precision
                                )
                                if not set_dtype[k]
                                else ty.cast(torch.Tensor, v)
                            )
                            for k, v in batch.items()
                        },
                    )

                with torch_utils.get_amp_ctxt_for_precision(precision=runtime_cfg.precision, device=device):
                    _run_warmup(model, batch, device, model_cfg.num_warmup_batches, progress_bar)
                    if measure_memory:
                        memory_alloc = torch_utils.measure_memory_allocation(model, batch, device)
                    else:
                        memory_alloc = 0
                    if runtime_cfg.enable_profiling:
                        if "activities" not in runtime_cfg.profiler_kwargs:
                            runtime_cfg.profiler_kwargs["activities"] = (
                                [
                                    torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA,
                                ]
                                if device.type == "cuda"
                                else [torch.profiler.ProfilerActivity.CPU]
                            )
                        profiler = torch.profiler.profile(
                            **runtime_cfg.profiler_kwargs,
                        )
                        profiler.start()
                    else:
                        profiler = None
                    cur_results = _measure_timings(
                        model=model,
                        batch=batch,
                        batch_size=batch_size,
                        device=device,
                        num_batches=model_cfg.num_batches,
                        progress_bar=progress_bar,
                        timing_function=torch_utils.measure_repeated_inference_timing,
                    )
                    if profiler is not None:
                        profiler.stop()
                        time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
                        profiler.export_chrome_trace(
                            f"{runtime_cfg.profiling_prefix}_{device}_{batch_size}_{time_str}.json"
                        )
            elif isinstance(model_cfg, OnnxModelConfig):
                from nvbenjo import onnx_utils

                batch = onnx_utils.get_rnd_input_batch(model.get_inputs(), model_cfg.shape, batch_size)

                memory_alloc = 0
                num_model_parameters = 0
                set_dtype = False
                if measure_memory:
                    memory_alloc = onnx_utils.measure_memory_allocation(model, batch, device)
                else:
                    memory_alloc = 0
                cur_results = _measure_timings(
                    model,
                    batch,
                    batch_size,
                    device,
                    model_cfg.num_batches,
                    progress_bar,
                    timing_function=onnx_utils.measure_repeated_inference_timing,
                )
            else:
                raise ValueError(f"Unknown model config type {type(model_cfg)}")

            del model
            del batch
            torch.cuda.empty_cache()

            cur_results["memory_bytes"] = memory_alloc
            cur_results["model"] = model_cfg.name
            cur_results["batch_size"] = batch_size
            cur_results["runtime_options"] = runtime_option_name
            cur_results["device"] = str(device)
            results.append(cur_results)
        except torch.cuda.OutOfMemoryError:
            console.print(
                f"[red]Out of memory for batch size {batch_size} and runtime_options {runtime_option_name} on device {str(device)}[/red]"
            )
            precision_batch_oom[runtime_option_name] = batch_size
            continue
        except Exception as e:
            if "Failed to allocate memory" in str(e) or "ALLOC_FAILED" in str(e):
                console.print(
                    f"[red]Out of memory for batch size {batch_size} and runtime_options {runtime_option_name} on device {str(device)}[/red]"
                )
                precision_batch_oom[runtime_option_name] = batch_size
                continue
            else:
                raise e
        finally:
            progress_bar.advance(bench_task)

    if progress_bar is not None:
        progress_bar.remove_task(bench_task)

    return pd.concat(results)
