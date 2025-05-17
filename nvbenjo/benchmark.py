import itertools
import logging
import os
import time
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from rich import progress
from rich.progress import Progress
from torch.utils.data import Dataset

import nvbenjo.utils as utils
from nvbenjo.cfg import ModelConfig
from nvbenjo import console

logger = logging.getLogger(__name__)


class DummyDataset(Dataset):
    def __init__(self, size, shape, dtype, min_max):
        self.size = size
        self.shape = shape
        self.min_max = min_max
        self.dtype = dtype

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.min_max is not None:
            min_val, max_val = self.min_max
        else:
            min_val, max_val = 0, 1

        if self.dtype is None or any(s in self.dtype for s in ["float", "double"]):
            dtype = self.dtype if self.dtype is not None else "float32"
            diff = max_val - min_val
            rnd = torch.randn(size=self.shape, dtype=getattr(torch, dtype))
            rnd = rnd * diff + min_val
        elif any(s in self.dtype for s in ["int", "long"]):
            rnd = torch.randint(low=int(min_val), high=int(max_val), size=self.shape, dtype=getattr(torch, self.dtype))
        else:
            raise ValueError(f"Invalid dtype {self.dtype}. Must be one of int, long, float, double")

        return rnd, None


def get_model(type_or_path: str, device: torch.device, **kwargs) -> nn.Module:
    if os.path.isfile(type_or_path):
        return torch.load(type_or_path, map_location=device, weights_only=False)

    if type_or_path.startswith("huggingface:"):
        logger.info("Attempting to load HuggingFace model")
        from transformers import AutoModel

        type_or_path = type_or_path[len("huggingface:") :]
        return AutoModel.from_pretrained(type_or_path).to(device)

    available_torchvision_models = torchvision.models.list_models()
    if type_or_path in available_torchvision_models:
        return torchvision.models.get_model(type_or_path, **kwargs).to(device)
    else:
        raise ValueError(
            (
                f"Invalid model {type_or_path}. Must be: \n"
                "- a valid path \n"
                "- a valid huggingface AutoModel (named 'huggingface:<model-name>')  \n"
                f"- or one of these available torchvision models: {available_torchvision_models}"
            )
        )


def measure_memory_allocation(model: nn.Module, batch: torch.Tensor, device: torch.device):
    if device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats(device=device)
    # before_run_allocation = torch.cuda.memory_allocated(device=device)

    batch = batch.to(device)
    model = model.to(device)
    r = model(batch)
    _ = utils.transfer_to_device(r, to_device="cpu")

    logger.debug(torch.cuda.memory_summary(device=device, abbreviated=True))

    # after_batch_allocation = torch.cuda.memory_allocated(device=device)
    max_batch_allocation = torch.cuda.max_memory_allocated(device=device)

    return max_batch_allocation


def measure_repeated_inference_timing(
    model: nn.Module,
    sample: torch.Tensor,
    model_device: torch.device,
    transfer_to_device_fn=utils.transfer_to_device,
    num_runs: int = 100,
    progress_callback: Optional[Callable] = None,
) -> dict:
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

        device_result = model(device_sample)

        if model_device.type == "cuda":
            stop_event.record()
            torch.cuda.synchronize()
            # elapsed_on_device = stop_event.elapsed_time(start_event)
            elapsed_on_device = start_event.elapsed_time(stop_event) / 1000.0
            stop_on_device = time.perf_counter()
        else:
            stop_on_device = time.perf_counter()
            elapsed_on_device = stop_on_device - start_on_device

        transfer_to_device_fn(device_result, "cpu")
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
                "time_total_batch_normalized": (stop_on_cpu - start_on_cpu) / sample.shape[0],
            }
        )
        if progress_callback is not None:
            progress_callback()

    results_raw = pd.DataFrame(results_raw)

    return results_raw


def benchmark_models(model_cfgs: list[ModelConfig]) -> Tuple[pd.DataFrame, dict]:
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


def _get_progress_bar() -> Optional[Progress]:
    return Progress(
        "[progress.description]{task.description}",
        progress.BarColumn(bar_width=80),
        "[progress.percentage]{task.completed}/{task.total}",
        console=console,
    )


def _run_warmup(
    model: nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    num_warmup_batches: int,
    progress_bar: Optional[Progress],
):
    try:
        if progress_bar is not None:
            warm_up_task = progress_bar.add_task("    Warm-up", total=num_warmup_batches)

        for _ in range(num_warmup_batches):
            r = model(batch.to(device))
            _ = utils.transfer_to_device(r, to_device="cpu")
            if progress_bar is not None:
                progress_bar.advance(warm_up_task)

    finally:
        if progress_bar is not None:
            progress_bar.remove_task(warm_up_task)


def _measure_timings(
    model: nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    num_batches: int,
    progress_bar: Optional[Progress],
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
        cur_raw_results = measure_repeated_inference_timing(
            model,
            batch,
            device,
            num_runs=num_batches,
            progress_callback=progress_callback,
        )
    finally:
        if progress_bar is not None:
            progress_bar.remove_task(measure_task)
    return cur_raw_results


def benchmark_model(model_cfg: ModelConfig, progress_bar=Optional[Progress]) -> Tuple[pd.DataFrame, dict]:
    results = []
    num_model_parameters = None
    precision_batch_oom = {}

    model = get_model(model_cfg.type_or_path, device="cpu", **model_cfg.kwargs)

    if progress_bar is None:
        progress_bar = _get_progress_bar()
    console = progress_bar.console

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

            shape = tuple(batch_size if s == "B" else s for s in model_cfg.shape)
            dset = DummyDataset(
                size=model_cfg.num_batches + model_cfg.num_warmup_batches + 1,
                shape=shape,
                dtype=model_cfg.input_dtype,
                min_max=model_cfg.input_min_max,
            )
            batch, _ = dset[0]

            model = get_model(model_cfg.type_or_path, device=device, **model_cfg.kwargs)
            if num_model_parameters is None:
                num_model_parameters = utils.get_model_parameters(model)

            # only apply precision to input if no precision is specified
            if model_cfg.input_dtype:
                model, _ = utils.apply_non_amp_model_precision(model, None, precision=precision)
            else:
                model, batch = utils.apply_non_amp_model_precision(model, batch, precision=precision)

            with utils.get_amp_ctxt_for_precision(precision=precision, device=device):
                _run_warmup(model, batch, device, model_cfg.num_warmup_batches, progress_bar)
                memory_alloc = measure_memory_allocation(model, batch, device)
                cur_results = _measure_timings(model, batch, device, model_cfg.num_batches, progress_bar)

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
