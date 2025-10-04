import itertools
import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rich import progress
from rich.progress import Progress

import nvbenjo.utils as utils
from nvbenjo import console, torch_utils
from nvbenjo.cfg import ModelConfig

logger = logging.getLogger(__name__)


def load_model(type_or_path: str, device: torch.device, **kwargs) -> tuple[Any, str]:
    if type_or_path.endswith(".onnx"):
        from nvbenjo import onnx_utils

        return onnx_utils.get_model(type_or_path, device=device, **kwargs), "onnx"
    else:
        return torch_utils.get_model(type_or_path, device=device, **kwargs), "torch"


def _test_load_models(model_cfgs: list[ModelConfig]) -> None:
    loaded_types = []
    logger.info("Checking if models are valid and available")
    for model_cfg in model_cfgs:
        if model_cfg.type_or_path not in loaded_types:
            _, _ = load_model(model_cfg.type_or_path, device=torch.device("cpu"), verbose=True, **model_cfg.kwargs)
            loaded_types.append(model_cfg.type_or_path)


def benchmark_models(
    model_cfgs: list[ModelConfig], measure_memory: Optional[bool] = True, profile: Optional[bool] = False
) -> pd.DataFrame:
    _test_load_models(model_cfgs)

    with _get_progress_bar() as progress_bar:
        model_task = progress_bar.add_task("Benchmarking models", total=len(model_cfgs))
        results = []

        for model_cfg in model_cfgs:
            progress_bar.update(model_task, description=f"Benchmarking {model_cfg.name}")
            model_results = benchmark_model(
                model_cfg, progress_bar=progress_bar, measure_memory=measure_memory, profile=profile
            )
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
):
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


def _get_device(model_type: str, device: str, console) -> torch.device:
    device_chosen = torch.device(device)
    if device_chosen.type == "cpu":
        return device_chosen
    elif device_chosen.type == "cuda":
        if model_type == "torch":
            if torch.cuda.is_available():
                return device_chosen
            else:
                console.print("[yellow]CUDA is not available. Running on CPU.[/yellow]")
                return torch.device("cpu")
        if model_type == "onnx":
            from nvbenjo import onnx_utils

            available_providers = onnx_utils.ort.get_available_providers()  # type: ignore
            if torch.cuda.is_available() and "CUDAExecutionProvider" in available_providers:
                return device_chosen
            else:
                if torch.cuda.is_available():
                    console.print(
                        "[yellow]CUDAExecutionProvider is not available in onnxruntime. Running on CPU.[/yellow]"
                    )
                else:
                    console.print("[yellow]CUDA is not available. Running on CPU.[/yellow]")
                return torch.device("cpu")
    else:
        raise ValueError(f"Invalid device {device}. Must be one of cpu or cuda")
    return device_chosen


def benchmark_model(
    model_cfg: ModelConfig,
    profile: Optional[bool] = False,
    measure_memory: Optional[bool] = True,
    progress_bar: Optional[Progress] = None,
) -> pd.DataFrame:
    results = []
    num_model_parameters = None
    precision_batch_oom = {}

    if profile:
        raise NotImplementedError("Profiling is not yet implemented.")

    if progress_bar is None:
        progress_bar = _get_progress_bar()
    console = progress_bar.console
    _, model_type = load_model(model_cfg.type_or_path, device=torch.device("cpu"), **model_cfg.kwargs)

    iter_cfgs = list(itertools.product(*[model_cfg.devices, model_cfg.batch_sizes, model_cfg.precisions]))
    bench_task = progress_bar.add_task("Running Benchmark", total=len(iter_cfgs))
    for device, batch_size, precision in iter_cfgs:
        if precision_batch_oom.get(precision, np.inf) < batch_size:
            # already went oom for this precision with smaller batch size -> skip bigger one
            progress_bar.advance(bench_task)
            continue
        try:
            device = _get_device(model_type, device, console)
            progress_bar.update(
                bench_task, description=f"  Device {device} | batch-size: {batch_size} | {precision.name}"
            )

            model, model_type = load_model(model_cfg.type_or_path, device=device, **model_cfg.kwargs)
            if model_type == "torch":
                batch, set_dtype = utils.get_rnd_from_shape_s(shape=model_cfg.shape, batch_size=batch_size)

                if num_model_parameters is None:
                    num_model_parameters = torch_utils.get_model_parameters(model)

                model = torch_utils.apply_non_amp_model_precision(model, precision=precision)

                # only apply precision to input if no precision is specified
                if not set_dtype:
                    batch = torch_utils.apply_batch_precision(batch, precision=precision)
                else:
                    batch = {
                        k: torch_utils.apply_batch_precision(v, precision=precision) if not set_dtype[k] else v
                        for k, v in batch.items()
                    }

                with torch_utils.get_amp_ctxt_for_precision(precision=precision, device=device):
                    _run_warmup(model, batch, device, model_cfg.num_warmup_batches, progress_bar)
                    if measure_memory:
                        memory_alloc = torch_utils.measure_memory_allocation(model, batch, device)
                    else:
                        memory_alloc = 0
                    cur_results = _measure_timings(
                        model,
                        batch,
                        batch_size,
                        device,
                        model_cfg.num_batches,
                        progress_bar,
                        timing_function=torch_utils.measure_repeated_inference_timing,
                    )
            elif model_type == "onnx":
                if utils.AMP_PREFIX in precision.value:
                    raise ValueError(f"ONNX models do not support AMP precision {precision}")                    

                from nvbenjo import onnx_utils

                batch = onnx_utils.get_rnd_input_batch(model.get_inputs(), model_cfg.shape, batch_size)

                memory_alloc = 0
                num_model_parameters = 0
                set_dtype = False

                # only apply precision to input if no precision is specified
                if not set_dtype:
                    batch = torch_utils.apply_batch_precision(batch, precision=precision)
                else:
                    batch = {
                        k: torch_utils.apply_batch_precision(v, precision=precision) if not set_dtype[k] else v
                        for k, v in batch.items()
                    }
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
                raise ValueError(f"Unknown model type {model_type}")

            del model
            del batch
            torch.cuda.empty_cache()

            cur_results["memory_bytes"] = memory_alloc
            cur_results["model"] = model_cfg.name
            cur_results["batch_size"] = batch_size
            cur_results["precision"] = precision.value
            cur_results["device"] = str(device)
            results.append(cur_results)
        except torch.cuda.OutOfMemoryError:
            console.print(
                f"[red]Out of memory for batch size {batch_size} and precision {precision} on device {str(device)}[/red]"
            )
            precision_batch_oom[precision] = batch_size
            continue
        finally:
            progress_bar.advance(bench_task)

    return pd.concat(results)
