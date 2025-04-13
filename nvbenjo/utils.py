from contextlib import AbstractContextManager, nullcontext

import torch
import torch.nn as nn


def format_num(num: int, bytes: bool = False) -> str:
    """Scale bytes to its proper format, e.g. 1253656 => '1.20MB'"""
    factor = 1024 if bytes else 1000
    suffix = "B" if bytes else ""
    for unit in ["", " K", " M", " G", " T", " P"]:
        if num < factor:
            return f"{num:.2f}{unit}{suffix}"
        num /= factor


def format_seconds(time_seconds: float):
    if time_seconds > 1:
        return f"{time_seconds:.3f} s"
    else:
        time_ms = time_seconds * 1000
        if time_ms > 1:
            return f"{time_ms:.3f} ms"
        else:
            time_us = time_ms * 1000
            return f"{time_us:.3f} us"


def get_amp_ctxt_for_precision(precision: str, device: torch.device) -> AbstractContextManager:
    if "amp" in precision:
        valid_values = ["amp", "amp_fp16", "amp_bfloat16"]
        if precision not in valid_values:
            raise ValueError(f"Invalid AMP precision type {precision} must be one of {valid_values}")

        if precision in ["amp"]:
            ctxt = torch.autocast(device_type=device.type, enabled=True)
        elif precision in ["amp_fp16"]:
            ctxt = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)
        elif precision in ["amp_bfloat16"]:
            ctxt = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True)
        else:
            raise ValueError(f"Invalid precision type {precision}.")
    else:
        ctxt = nullcontext()
    return ctxt


def apply_non_amp_model_precision(model: nn.Module, batch: torch.Tensor, precision: str):
    if "amp" not in precision:
        if precision == "fp16":
            model = model.half()
            batch = batch.half()
        elif precision == "bfloat16":
            model = model.bfloat16()
            batch = batch.bfloat16()
        else:
            assert precision == "fp32"
        return model, batch
    else:
        return model, batch
