import os
import time
import typing as ty

import onnxruntime as ort
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig

from nvbenjo import console
from nvbenjo.torch_utils import transfer_to_device
from nvbenjo.utils import EXAMPLE_VALID_SHAPES, check_shape_dict, get_rnd_from_shape_s


def get_model(type_or_path: str, device: torch.device, verbose=False, **kwargs) -> ort.InferenceSession:
    if not type_or_path.endswith(".onnx") or not os.path.isfile(type_or_path):
        raise ValueError(f"Invalid model {type_or_path}. Must be a valid ONNX path ending with .onnx")

    if verbose and console is not None:
        console.print(f"Loading ONNX model {type_or_path}")

    if "providers" not in kwargs:
        if device.type == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        kwargs["providers"] = providers

    sess = ort.InferenceSession(type_or_path, **kwargs)
    return sess


def get_rnd_input_batch(onnx_session_inputs, shape: tuple, batch_size: int) -> ty.Dict[str, torch.Tensor]:
    def strip_type_string(s: str) -> str:
        if s.startswith("tensor(") and s.endswith(")"):
            s = s[len("tensor(") : -1]
        return s

    if not isinstance(shape, dict) and all(isinstance(si, (str, int)) for si in shape):
        # simple shape e.g. (B, 3, 224, 224)
        if len(onnx_session_inputs) != 1:
            raise ValueError(
                "The model has multiple inputs, but the provided shape is a single shape. Please provide a list of shapes or a dict of shapes."
            )
        model_input = onnx_session_inputs[0]
        rnd_shape = ({"name": model_input.name, "type": strip_type_string(model_input.type), "shape": shape},)
    elif all(isinstance(si, (tuple, list, ListConfig)) for si in shape):
        # tuple of shapes e.g. ((B, 3, 224, 224), (B, 10))
        if len(onnx_session_inputs) != len(shape):
            raise ValueError(
                f"The model has {len(onnx_session_inputs)} inputs, but the provided shape has {len(shape)} shapes. Please provide a list of shapes or a dict of shapes."
            )
        rnd_shape = tuple(
            {"name": model_input.name, "type": strip_type_string(model_input.type), "shape": si}
            for model_input, si in zip(onnx_session_inputs, shape)
        )
    elif all(isinstance(si, (dict, DictConfig)) for si in shape):
        onnx_inputs_by_name = {inp.name: inp for inp in onnx_session_inputs}
        rnd_shape = tuple(dict(si) for si in shape)  # convert from DictConfig to dict
        if len(onnx_session_inputs) != len(shape):
            raise ValueError(
                f"The model has {len(onnx_session_inputs)} inputs, but the provided shape has {len(shape)} shapes. Please provide a list of shapes or a dict of shapes."
            )
        for si in rnd_shape:
            check_shape_dict(si)
            # # name='input', type='float', shape=['B', 320000, 1], min_max=(0, 1)
            if si["name"] not in onnx_inputs_by_name:
                raise ValueError(
                    f"The model does not have an input named {si['name']}. Available inputs are: {list(onnx_inputs_by_name.keys())}"
                )
            if "type" not in si:
                si["type"] = strip_type_string(onnx_inputs_by_name[si["name"]].type)
            if "shape" not in si:
                si["shape"] = onnx_inputs_by_name[si["name"]].shape
    else:
        raise ValueError(
            (
                f"Invalid shape {shape}.\n "
                "Example valid inputs:\n " + "\n - ".join([str(s) for s in EXAMPLE_VALID_SHAPES])
            )
        )
    batch, _ = get_rnd_from_shape_s(shape=rnd_shape, batch_size=batch_size)
    if not isinstance(batch, dict):
        raise ValueError("Internal Error was unable to generate dict of inputs for ONNX model.")
    return batch


def measure_repeated_inference_timing(
    model: ort.InferenceSession,
    sample: ty.Dict[str, torch.Tensor],
    batch_size: int,
    model_device: torch.device,
    transfer_to_device_fn=transfer_to_device,
    num_runs: int = 100,
    progress_callback: ty.Optional[ty.Callable] = None,
) -> pd.DataFrame:
    time_cpu_to_device = []
    time_inference = []
    time_device_to_cpu = []
    time_total = []
    results_raw = []

    # Convert ONNX type to PyTorch dtype
    dtype_map = {
        "tensor(float)": torch.float32,
        "tensor(float16)": torch.float16,
        "tensor(int64)": torch.int64,
        "tensor(int32)": torch.int32,
        "tensor(bool)": torch.bool,
    }

    # TODO: !! just run without io mapping first so we get the correct shapes?
    # TODO:; Make this work for multiple inputs
    # TODO: !! need to support auto-dedetcint
    onnx_model_outputs = model.get_outputs()

    # run model once so we know the output shapes
    outputs = model.run(None, {n: d.cpu().numpy() for n, d in sample.items()})
    output_shapes = {}
    for onnx_output, output_shape in zip(onnx_model_outputs, outputs):
        output_shapes[onnx_output.name] = output_shape.shape
    del outputs

    for _ in range(num_runs):
        start_on_cpu = time.perf_counter()
        device_sample = transfer_to_device_fn(sample, model_device)

        if model_device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            stop_event = torch.cuda.Event(enable_timing=True)
            start_event.record()  # For GPU timing
        start_on_device = time.perf_counter()  # For CPU timing

        io_binding = model.io_binding()
        device_id = 0 if model_device.index is None else model_device.index

        if isinstance(device_sample, dict):
            for i, (name, input) in enumerate(device_sample.items()):
                io_binding.bind_input(
                    name=name,
                    device_type=model_device.type,
                    device_id=device_id,
                    element_type=str(input.dtype).strip("torch."),
                    shape=input.shape,
                    buffer_ptr=input.data_ptr(),
                )
        else:
            raise ValueError(f"Invalid input type {type(device_sample)}. Must be one of list, tuple, dict")

        device_result = []
        for i, output in enumerate(onnx_model_outputs):
            torch_dtype = dtype_map.get(output.type, torch.float32)  # default to float32 if type not found
            output_tensor = torch.empty(size=output_shapes[output.name], dtype=torch_dtype, device=model_device)
            io_binding.bind_output(
                name=output.name,
                device_type=model_device.type,
                device_id=device_id,
                element_type=str(output_tensor.dtype).strip("torch."),
                shape=output_tensor.shape,
                buffer_ptr=output_tensor.data_ptr(),
            )
            device_result.append(output_tensor)

        model.run_with_iobinding(io_binding)

        if model_device.type == "cuda":
            stop_event.record()
            torch.cuda.synchronize()
            elapsed_on_device = start_event.elapsed_time(stop_event) / 1000.0
            stop_on_device = time.perf_counter()
        else:
            stop_on_device = time.perf_counter()
            elapsed_on_device = stop_on_device - start_on_device

        transfer_to_device_fn(device_result, torch.device("cpu"))
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
