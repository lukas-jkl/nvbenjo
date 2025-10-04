import itertools
import logging
from typing import Optional, Callable, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rich import progress
from rich.progress import Progress

import nvbenjo.utils as utils
from nvbenjo.cfg import ModelConfig
from nvbenjo import console

from nvbenjo import torch_utils

logger = logging.getLogger(__name__)


def load_model(type_or_path: str, device: torch.device, **kwargs):
    if type_or_path.endswith(".onnx"):
        from nvbenjo import onnx_utils

        return onnx_utils.get_model(type_or_path, device=device, **kwargs)
    else:
        return torch_utils.get_model(type_or_path, device=device, **kwargs)


def _test_load_models(model_cfgs: list[ModelConfig]) -> None:
    loaded_types = []
    logger.info("Checking if models are valid and available")
    for model_cfg in model_cfgs:
        if model_cfg.type_or_path not in loaded_types:
            _ = load_model(model_cfg.type_or_path, device=torch.device("cpu"), verbose=True, **model_cfg.kwargs)
            loaded_types.append(model_cfg.type_or_path)


def benchmark_models(model_cfgs: list[ModelConfig]) -> pd.DataFrame:
    _test_load_models(model_cfgs)

    with _get_progress_bar() as progress_bar:
        model_task = progress_bar.add_task("Benchmarking models", total=len(model_cfgs))
        results = []

        for model_cfg in model_cfgs:
            progress_bar.update(model_task, description=f"Benchmarking {model_cfg.name}")
            model_results = benchmark_model(model_cfg, progress_bar)
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
            _ = torch_utils.transfer_to_device(r, to_device=torch.device("cpu"))
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


def benchmark_model(model_cfg: ModelConfig, progress_bar: Optional[Progress] = None) -> pd.DataFrame:
    results = []
    num_model_parameters = None
    precision_batch_oom = {}

    if progress_bar is None:
        progress_bar = _get_progress_bar()
    console = progress_bar.console
    model = load_model(model_cfg.type_or_path, device=torch.device("cpu"), **model_cfg.kwargs)

    iter_cfgs = list(itertools.product(*[model_cfg.device_indices, model_cfg.batch_sizes, model_cfg.precisions]))
    bench_task = progress_bar.add_task("Running Benchmark", total=len(iter_cfgs))
    for device_idx, batch_size, precision in iter_cfgs:
        if precision_batch_oom.get(precision, np.inf) < batch_size:
            # already went oom for this precision with smaller batch size -> skip bigger one
            progress_bar.advance(bench_task)
            continue
        try:
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{device_idx}")
            else:
                console.print("[yellow]CUDA is not available. Running on CPU.[/yellow]")
                device = torch.device("cpu")

            progress_bar.update(
                bench_task, description=f"  Device {device} | batch-size: {batch_size} | {precision.name}"
            )

            model = load_model(model_cfg.type_or_path, device=device, **model_cfg.kwargs)
            if isinstance(model, nn.Module):
                batch, set_dtype = utils.get_rnd_from_shape_s(shape=model_cfg.shape, batch_size=batch_size)

                if num_model_parameters is None:
                    num_model_parameters = torch_utils.get_model_parameters(model)

                model = torch_utils.apply_non_amp_model_precision(model, precision=precision)
                # only apply precision to input if no precision is specified
                if not set_dtype:
                    batch = torch_utils.apply_batch_precision(batch, precision=precision)

                with torch_utils.get_amp_ctxt_for_precision(precision=precision, device=device):
                    _run_warmup(model, batch, device, model_cfg.num_warmup_batches, progress_bar)
                    memory_alloc = torch_utils.measure_memory_allocation(model, batch, device)
                    cur_results = _measure_timings(
                        model,
                        batch,
                        batch_size,
                        device,
                        model_cfg.num_batches,
                        progress_bar,
                        timing_function=torch_utils.measure_repeated_inference_timing,
                    )
            else:
                from nvbenjo import onnx_utils

                batch = onnx_utils.get_rnd_input_batch(model.get_inputs(), model_cfg.shape, batch_size)

                memory_alloc = 0
                num_model_parameters = 0
                # maybe use a onnx util instead that only applies to batch and respects which types are possible?
                # maybe we can check automatically if model supports fp16/bf16
                batch = torch_utils.apply_batch_precision(batch, precision=precision)
                memory_alloc = onnx_utils.measure_memory_allocation(model, batch, device)
                cur_results = _measure_timings(
                    model,
                    batch,
                    batch_size,
                    device,
                    model_cfg.num_batches,
                    progress_bar,
                    timing_function=onnx_utils.measure_repeated_inference_timing,
                )

            cur_results["memory_bytes"] = memory_alloc
            cur_results["model"] = model_cfg.name
            cur_results["batch_size"] = batch_size
            cur_results["precision"] = precision.value
            cur_results["device_idx"] = device_idx
            results.append(cur_results)
        except torch.cuda.OutOfMemoryError:
            console.print(
                f"[red]Out of memory for batch size {batch_size} and precision {precision} on device {device_idx}[/red]"
            )
            precision_batch_oom[precision] = batch_size
            continue
        finally:
            progress_bar.advance(bench_task)

    return pd.concat(results)
