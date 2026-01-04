from __future__ import annotations

import typing as ty
import pandas as pd
from enum import Enum
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig

import torch

BATCH_SIZE_IDENTIFIERS = ("B", "batch_size")
TRANSFER_WARNING = (
    "[yellow]Warning: Could not transfer model output to CPU. Time to CPU measures will be incorrect.[/yellow]"
)

SingleShape = tuple[int | str, ...] | dict[str, ty.Any]
MultiShape = tuple[SingleShape, ...]
Shape = SingleShape | MultiShape
TensorLike = torch.Tensor | tuple[torch.Tensor, ...] | dict[str, torch.Tensor]
ProviderType = ty.Union[str, ty.Tuple[str, dict[ty.Any, ty.Any]]]

EXAMPLE_VALID_SHAPES: list[Shape] = [
    ("B", 3, 224, 224),
    (("B", 3, 224, 224), ("B", 10)),
    ({"name": "input1", "type": "float", "shape": ("B", 3, 224, 224), "min_max": (0, 1)},),
    (
        {"name": "input1", "type": "float", "shape": ("B", 3, 224, 224), "min_max": (0, 1)},
        {"name": "input2", "type": "int", "shape": (1, 3)},
        {"name": "input3", "type": "int", "shape": (), "value": 42},
    ),
]

AMP_PREFIX = "amp"


class NoBatchShapeError(ValueError):
    pass


class PrecisionType(Enum):
    AMP = f"{AMP_PREFIX}"
    AMP_FP16 = f"{AMP_PREFIX}_fp16"
    AMP_BFLOAT16 = f"{AMP_PREFIX}_bfloat16"
    FP32 = "fp32"
    FP16 = "fp16"
    BFLOAT16 = "bfloat16"
    LONG = "long"


def format_num(num: ty.Union[int, float], bytes: bool = False) -> ty.Union[str, None]:
    """Scale bytes to its proper format, e.g. 1253656 => '1.20MB'"""
    if num is None:
        return num
    factor = 1024 if bytes else 1000
    suffix = "B" if bytes else ""
    for unit in ["", " K", " M", " G", " T", " P"]:
        if num < factor:
            return f"{num:.2f}{unit}{suffix}"
        num /= factor


def format_seconds(time_seconds: float) -> str:
    if time_seconds > 1:
        return f"{time_seconds:.3f} s"
    else:
        time_ms = time_seconds * 1000
        if time_ms > 1:
            return f"{time_ms:.3f} ms"
        else:
            time_us = time_ms * 1000
            return f"{time_us:.3f} us"


def _get_rnd(
    shape_tuple: tuple[int], dtype: ty.Optional[str], min_val: int, max_val: int, value: ty.Optional[ty.Any] = None
) -> torch.Tensor:
    if dtype is None or any(s in dtype for s in ["float", "double"]):
        dtype = dtype if dtype is not None else "float32"
        if value:
            if isinstance(value, (ListConfig, list)):
                rnd = torch.tensor(value, dtype=getattr(torch, dtype))
            else:
                rnd = torch.full(size=shape_tuple, fill_value=value, dtype=getattr(torch, dtype))
        else:
            rnd = torch.distributions.Uniform(min_val, max_val).sample(shape_tuple).to(dtype=getattr(torch, dtype))
    elif any(s in dtype for s in ["int", "long"]):
        if value:
            if isinstance(value, (ListConfig, list)):
                rnd = torch.tensor(value, dtype=getattr(torch, dtype))
            else:
                rnd = torch.full(size=shape_tuple, fill_value=value, dtype=getattr(torch, dtype))
        else:
            rnd = torch.randint(low=int(min_val), high=int(max_val), size=shape_tuple, dtype=getattr(torch, dtype))
    else:
        raise ValueError(f"Invalid dtype {dtype}. Must be one of int, long, float, double")
    return rnd


def _check_shape_dict(si: dict | DictConfig) -> None:
    if "name" not in si:
        raise ValueError(f"The shape definition {si} must contain a 'name' key.")
    if "dtype" in si:
        raise ValueError("The 'dtype' is not valid. Did you mean 'type'?")
    if "shape" not in si:
        raise ValueError(f"The shape definition {si} must contain a 'shape' key.")
    for k in si.keys():
        if k not in ["name", "type", "shape", "min_max", "value"]:
            raise ValueError(f"Invalid key {k} in shape definition {si}.")
    if "min_max" in si.keys() and "value" in si.keys():
        raise ValueError(f"Invalid shape definition {si} can only specify min_max or value.")


def get_rnd_from_shape_s(
    shape: Shape, batch_size: int, dtype=None, min_val=0, max_val=1
) -> tuple[TensorLike, dict[str, bool]]:
    try:
        depends_on_batch = False
        set_individual_types = {}

        def _get_rnd_batch(shape_tuple):
            if any(s in shape_tuple for s in BATCH_SIZE_IDENTIFIERS):
                nonlocal depends_on_batch
                depends_on_batch = True
            return tuple(batch_size if s in BATCH_SIZE_IDENTIFIERS else s for s in shape_tuple)

        if not isinstance(shape, dict) and all(isinstance(si, (str, int)) for si in shape):
            # simple shape e.g. (B, 3, 224, 224)
            rnd_input = _get_rnd(shape_tuple=_get_rnd_batch(shape), dtype=dtype, min_val=min_val, max_val=max_val)
        elif all(isinstance(si, (tuple, list, ListConfig)) for si in shape):
            # tuple of shapes e.g. ((B, 3, 224, 224), (B, 10))
            rnd_input = tuple(
                _get_rnd(_get_rnd_batch(si), dtype=dtype, min_val=min_val, max_val=max_val) for si in shape
            )
        elif all(isinstance(si, (dict, DictConfig)) for si in shape):
            rnd_input = {}
            for si in shape:
                # check is redundant but helps type checker
                if not isinstance(si, (dict, DictConfig)):
                    raise ValueError(f"Shape item {si} must be of type dict.")

                # name='input', type='float', shape=['B', 320000, 1], min_max=(0, 1)
                _check_shape_dict(si)
                name = si["name"]
                set_individual_types[name] = "type" in si

                rnd_input[name] = _get_rnd(
                    shape_tuple=_get_rnd_batch(si["shape"]),
                    dtype=si.get("type", dtype),
                    min_val=si.get("min_max", (min_val, max_val))[0],
                    max_val=si.get("min_max", (min_val, max_val))[1],
                    value=si.get("value", None),
                )
        else:
            raise ValueError(
                (
                    f"Invalid shape {shape}.\n "
                    "Example valid inputs:\n " + "\n - ".join([str(s) for s in EXAMPLE_VALID_SHAPES])
                )
            )

        if not depends_on_batch:
            raise NoBatchShapeError(
                f"Shape {shape} does not depend on batch size. "
                f"Please ensure that the shape contains an identifier for the batch size: {BATCH_SIZE_IDENTIFIERS}."
            )
    except NoBatchShapeError as e:
        raise e
    except Exception as e:
        raise ValueError(
            f"Failed to generate random input from shape {shape} with batch size {batch_size}. "
            f"Please ensure that the shape is valid.\n"
            + "Example valid inputs:\n "
            + "\n - ".join([str(s) for s in EXAMPLE_VALID_SHAPES])
        ) from e

    return rnd_input, set_individual_types


def calculate_batchmetrics(results: pd.DataFrame, custom_batchmetrics: dict[str, float]) -> pd.DataFrame:
    """Calculate custom batch metrics and add them to the results DataFrame.

    Parameters
    ----------
    results : pd.DataFrame
        The benchmark results DataFrame.
    custom_batchmetrics : dict[str, float]
        Dictionary of custom batch metrics to calculate. The key is the metric name and the value is the multiplier.

    Returns
    -------
    pd.DataFrame
        The updated results DataFrame with custom batch metrics added.
    """
    for metric_name, value in custom_batchmetrics.items():
        results[metric_name] = value / results["time_total_batch_normalized"]
    return results
