from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from enum import Enum
import typing as ty

import torch
import torch.nn as nn

AMP_PREFIX = "amp"


class PrecisionType(Enum):
    AMP = f"{AMP_PREFIX}"
    AMP_FP16 = f"{AMP_PREFIX}_fp16"
    AMP_BFLOAT16 = f"{AMP_PREFIX}_bfloat16"
    FP32 = "fp32"
    FP16 = "fp16"
    BFLOAT16 = "bfloat16"
    LONG = "long"


def format_num(num: int, bytes: bool = False) -> str:
    """Scale bytes to its proper format, e.g. 1253656 => '1.20MB'"""
    if num is None:
        return num
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


def apply_non_amp_model_precision(model: nn.Module, batch: ty.Optional[torch.Tensor], precision: PrecisionType):
    if AMP_PREFIX not in precision.value:
        if precision == PrecisionType.FP16:
            model = model.half()
            if batch is not None:
                batch = batch.half()
        elif precision == PrecisionType.BFLOAT16:
            model = model.bfloat16()
            if batch is not None:
                batch = batch.bfloat16()
        else:
            if precision != PrecisionType.FP32:
                raise ValueError(f"Invalid precision type {precision}.")
        return model, batch
    else:
        return model, batch


def get_model_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def transfer_to_device(result: any, to_device: torch.device):
    if hasattr(result, "to"):
        return result.to(to_device)
    if isinstance(result, Sequence):
        return [transfer_to_device(ri, to_device=to_device) for ri in result]
    elif hasattr(result, "items"):
        return {k: transfer_to_device(v, to_device=to_device) for k, v in result.items()}
    else:
        raise ValueError(f"Unsupported result type: {type(result)} could not transfer to {to_device}")
