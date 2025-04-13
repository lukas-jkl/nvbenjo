import itertools
import logging
import os
import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

import nvbenjo.utils as utils
from nvbenjo.cfg import ModelConfig

logger = logging.getLogger(__name__)


class DummyDataset(Dataset):
    def __init__(self, size, shape):
        self.size = size
        self.shape = shape

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(self.shape), None


def get_model(type_or_path: str, device: torch.device, **kwargs) -> nn.Module:
    if os.path.isfile(type_or_path):
        return torch.load(type_or_path, map_location=device)

    available_torchvision_models = torchvision.models.list_models()
    if type_or_path in available_torchvision_models:
        return torchvision.models.get_model(type_or_path, **kwargs).to(device)
    else:
        raise ValueError(f"Invalid model {type_or_path}. Must be a valid path or one of {available_torchvision_models}")


def get_model_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_memory_allocation(model: nn.Module, batch: torch.Tensor, device: torch.device):
    torch.cuda.reset_peak_memory_stats(device=device)
    # before_run_allocation = torch.cuda.memory_allocated(device=device)

    batch = batch.to(device)
    model = model.to(device)
    _ = model(batch).to(device)

    logger.debug(torch.cuda.memory_summary(device=device, abbreviated=True))

    # after_batch_allocation = torch.cuda.memory_allocated(device=device)
    max_batch_allocation = torch.cuda.max_memory_allocated(device=device)

    return max_batch_allocation


def warm_up(model, batch, device, num_batches):
    for _ in tqdm(range(num_batches), desc="Warm-up batches", leave=False):
        _ = model(batch.to(device)).to("cpu")
    return


def measure_repeated_inference_timing(
    model: nn.Module,
    sample: torch.Tensor,
    model_device: torch.device,
    precision: str = "fp32",
    transfer_to_device_fn=torch.Tensor.to,
    num_runs: int = 100,
) -> dict:
    time_cpu_to_device = []
    time_inference = []
    time_device_to_cpu = []
    time_total = []

    results_raw = []

    for _ in tqdm(range(num_runs), desc="Measuring inference", leave=False):
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
            }
        )

    results_raw = pd.DataFrame(results_raw)

    human_readable_results = {}
    for key in ["time_cpu_to_device", "time_inference", "time_device_to_cpu", "time_total"]:
        s_per_batch = results_raw[key]
        batches_per_s = 1.0 / s_per_batch
        key = key.replace("time_", "", 1)
        human_readable_results[key] = {
            "batches_per_second": (
                f"{utils.format_num(batches_per_s.mean())} +/- {utils.format_num(batches_per_s.std())} "
                f" [{utils.format_num(batches_per_s.min())}, {utils.format_num(batches_per_s.max())}]"
            ),
            "batch_latency": (
                f"{utils.format_seconds(s_per_batch.mean())} +/- {utils.format_seconds(s_per_batch.std())}"
                f" [{utils.format_seconds(s_per_batch.min())}, {utils.format_seconds(s_per_batch.max())}]"
            ),
        }
    logger.debug("\n" + yaml.dump(human_readable_results, width=1000) + "\n")
    return results_raw, human_readable_results


def benchmark_model(model_cfg: ModelConfig) -> Tuple[pd.DataFrame, dict]:
    raw_results = []
    human_readable_results = {}
    model = get_model(model_cfg.type_or_path, device="cpu", **model_cfg.kwargs)

    num_model_parameters = None
    precision_batch_oom = {}
    iter_cfgs = tqdm(list(itertools.product(*[model_cfg.device_indices, model_cfg.batch_sizes, model_cfg.precisions])))
    for device_idx, batch_size, precision in iter_cfgs:
        if precision_batch_oom.get(precision, np.inf) < batch_size:
            # already went oom for this precision with smaller batch size -> skip bigger one
            continue
        try:
            device = torch.device(f"cuda:{device_idx}")
            iter_cfgs.set_description(f"Device {str(device)} | Batch Size: {batch_size} | Precision: {precision}")

            shape = tuple(batch_size if s == "B" else s for s in model_cfg.shape)
            dset = DummyDataset(size=model_cfg.num_batches + model_cfg.num_warmup_batches + 1, shape=shape)

            model = get_model(model_cfg.type_or_path, device=device, **model_cfg.kwargs)
            if num_model_parameters is None:
                num_model_parameters = get_model_parameters(model)

            batch, _ = dset[0]

            model, batch = utils.apply_non_amp_model_precision(model, batch, precision=precision)
            with utils.get_amp_ctxt_for_precision(precision=precision, device=device):
                warm_up(model, batch, device, num_batches=model_cfg.num_warmup_batches)
                memory_alloc = measure_memory_allocation(model, batch, device)
                cur_raw_results, cur_human_readable_results = measure_repeated_inference_timing(
                    model, batch, device, precision=precision, num_runs=model_cfg.num_batches
                )
            cur_raw_results["memory_bytes"] = memory_alloc
            cur_human_readable_results["memory"] = utils.format_num(memory_alloc, bytes=True)

            cur_raw_results["model"] = model_cfg.name
            cur_raw_results["batch_size"] = batch_size
            cur_raw_results["precision"] = precision
            cur_raw_results["device_idx"] = device_idx
            human_readable_results[f"{model_cfg.name}_b{batch_size}_{precision}"] = cur_human_readable_results
            raw_results.append(cur_raw_results)
        except torch.cuda.OutOfMemoryError:
            logger.info(f"Out of memory for batch size {batch_size} and precision {precision}")
            precision_batch_oom[precision] = batch_size
            continue

    raw_results = pd.concat(raw_results)
    return raw_results, human_readable_results
