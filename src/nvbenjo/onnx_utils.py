import os
import time
import nvitop
import threading
import typing as ty

from .cfg import OnnxRuntimeConfig

try:
    import onnxruntime as ort  # type: ignore
except ImportError:
    raise ImportError("onnxruntime is not installed. Please install onnx and onnxruntime or onnxruntime-gpu.")

import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from . import console
from .torch_utils import transfer_to_device
from .utils import EXAMPLE_VALID_SHAPES, TRANSFER_WARNING, _check_shape_dict, get_rnd_from_shape_s, Shape


def get_model(
    type_or_path: str, device: torch.device, runtime_config: OnnxRuntimeConfig, **kwargs
) -> ort.InferenceSession:
    """Load a onnx model into InferenceSession.

    Parameters
    ----------
    type_or_path : str
        Path to the ONNX model file.
    device : torch.device
        Device to run the ONNX model on.
    runtime_config : OnnxRuntimeConfig
        Runtime configuration for the ONNX model.
    **kwargs
        Additional keyword arguments to pass to the InferenceSession.

    Returns
    -------
    ort.InferenceSession
        Loaded ONNX InferenceSession.
    """
    type_or_path = os.path.expanduser(type_or_path)
    if not type_or_path.endswith(".onnx") or not os.path.isfile(type_or_path):
        raise ValueError(f"Invalid model {type_or_path}. Must be a valid ONNX path ending with .onnx")

    if runtime_config.execution_providers is None:
        if device.type == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():  # type: ignore
                raise RuntimeError(
                    "CUDAExecutionProvider is not available in onnxruntime. Please install onnxruntime-gpu or run on CPU."
                )
            else:
                providers = [("CUDAExecutionProvider", {"device_id": device.index}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
    else:
        providers = runtime_config.execution_providers

    # Convert list in providers to tuple if needed
    if isinstance(providers, ListConfig):
        providers = OmegaConf.to_container(providers)
    for i, provider in enumerate(providers):  # type: ignore
        if isinstance(provider, (list, ListConfig)):
            providers[i] = tuple(provider)  # type: ignore

    session_options = ort.SessionOptions()  # type: ignore
    session_options.log_severity_level = runtime_config.log_severity_level
    session_options.intra_op_num_threads = runtime_config.intra_op_num_threads
    session_options.inter_op_num_threads = runtime_config.inter_op_num_threads
    session_options.enable_profiling = runtime_config.enable_profiling
    if runtime_config.enable_profiling and runtime_config.profiling_prefix is not None:
        session_options.profile_file_prefix = runtime_config.profiling_prefix
    session_options.graph_optimization_level = getattr(
        ort.GraphOptimizationLevel,  # type: ignore
        runtime_config.graph_optimization_level,
    )

    sess = ort.InferenceSession(
        type_or_path,
        sess_options=session_options,
        providers=providers,  # type: ignore
        provider_options=runtime_config.provider_options,
        **kwargs,
    )
    return sess


def _sample_gpu_memory(
    device: torch.device, stop_event: threading.Event, max_mem: list[int], sample_time_s: float = 0.010
):
    if device.type == "cuda":
        gpu = nvitop.Device(device.index)
    else:
        max_mem[0] = 1
    while not stop_event.is_set():
        if device.type == "cuda":
            mem = gpu.memory_used()
            if isinstance(mem, int) and mem > max_mem[0]:
                max_mem[0] = mem
        time.sleep(sample_time_s)


def measure_memory_allocation(
    model: ort.InferenceSession, sample: dict[str, torch.Tensor], device: torch.device, iterations: int = 3
) -> int:
    """Measure the memory usage during inference

    Parameters
    ----------
    model : ort.InferenceSession
        The ONNX model to benchmark.
    sample : dict[str, torch.Tensor]
        Sample input data for the model.
    device : torch.device
        Device to run the model on.
    iterations : int, optional
        Number of iterations to run for measuring memory allocation, by default 3

    Returns
    -------
    int
        Maximum memory allocated during inference in bytes.
    """
    max_mem = [-1]
    stop_event = threading.Event()

    # Start memory sampling thread so we can measure peak memory usage in parallel
    sampler = threading.Thread(target=_sample_gpu_memory, args=(device, stop_event, max_mem))
    sampler.start()
    time.sleep(0.01)  # give sampler some time to start

    try:
        for _ in range(iterations):
            _ = model.run(None, {n: d.cpu().numpy() for n, d in sample.items()})
    finally:
        stop_event.set()
        sampler.join()

    max_mem = max_mem[0]
    if max_mem == -1:
        raise RuntimeError("Memory measurement failed!")

    return max_mem


def _get_formated_input_info(onnx_inputs) -> str:
    return "\n".join(
        f"- name: {onnx_input.name}, dtype: {onnx_input.type}, shape: {onnx_input.shape}{', default: ' + str(onnx_input.value) if hasattr(onnx_inputs, 'value') else ''}"
        for onnx_input in onnx_inputs
    )


def get_rnd_input_batch(onnx_session_inputs, shape: Shape, batch_size: int) -> dict[str, torch.Tensor]:
    def strip_type_string(s: str) -> str:
        if s.startswith("tensor(") and s.endswith(")"):
            s = s[len("tensor(") : -1]
        return s

    if not isinstance(shape, dict) and all(isinstance(si, (str, int)) for si in shape):
        # simple shape e.g. (B, 3, 224, 224)
        if len(onnx_session_inputs) != 1:
            raise ValueError(
                "The model has multiple inputs, but the provided input is a single shape."
                f"Model Inputs: \n{_get_formated_input_info(onnx_session_inputs)}"
            )
        model_input = onnx_session_inputs[0]
        rnd_shape = ({"name": model_input.name, "type": strip_type_string(model_input.type), "shape": shape},)
    elif all(isinstance(si, (tuple, list, ListConfig)) for si in shape):
        # tuple of shapes e.g. ((B, 3, 224, 224), (B, 10))
        if len(onnx_session_inputs) != len(shape):
            raise ValueError(
                f"The model has {len(onnx_session_inputs)} inputs, but the provided input has {len(shape)} shapes. Please provide a list of shapes or a dict of shapes."
                f"Model Inputs: \n{_get_formated_input_info(onnx_session_inputs)}"
            )
        rnd_shape = tuple(
            {"name": model_input.name, "type": strip_type_string(model_input.type), "shape": si}
            for model_input, si in zip(onnx_session_inputs, shape)
        )
    elif all(isinstance(si, (dict, DictConfig)) for si in shape):
        onnx_inputs_by_name = {inp.name: inp for inp in onnx_session_inputs}
        rnd_shape = tuple(dict(si) for si in shape)  # convert from DictConfig to dict # type: ignore
        if len(onnx_session_inputs) != len(shape):
            raise ValueError(
                f"The model has {len(onnx_session_inputs)} inputs, but the provided input has {len(shape)} shapes. Please provide a list of shapes or a dict of shapes."
                f"Model Inputs: \n{_get_formated_input_info(onnx_session_inputs)}"
            )
        for si in rnd_shape:
            _check_shape_dict(si)
            # # name='input', type='float', shape=['B', 320000, 1], min_max=(0, 1)
            if si["name"] not in onnx_inputs_by_name:
                raise ValueError(
                    f"The model does not have an input named {si['name']}."
                    f"Model Inputs: \n{_get_formated_input_info(onnx_session_inputs)}"
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
    return batch  # type: ignore


def measure_repeated_inference_timing(
    model: ort.InferenceSession,
    sample: dict[str, torch.Tensor],
    batch_size: int,
    model_device: torch.device,
    transfer_to_device_fn: ty.Callable = transfer_to_device,
    num_runs: int = 100,
    progress_callback: ty.Optional[ty.Callable] = None,
) -> pd.DataFrame:
    """Measure inference times.

    Parameters
    ----------
    model : ort.InferenceSession
        The ONNX model to benchmark.
    sample : dict[str, torch.Tensor]
        Sample input data for the model.
    batch_size : int
        Batch size for the input data.
    model_device : torch.device
        Device to run the model on.
    transfer_to_device_fn : Callable, optional
        Function to transfer data to the device, by default transfer_to_device
    num_runs : int, optional
        Number of runs to perform for timing, by default 100
    progress_callback : ty.Optional[ty.Callable], optional
        Callback function for p

    Returns
    -------
    pd.DataFrame
        DataFrame containing timing results.
    """
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

    onnx_model_outputs = model.get_outputs()

    # run model once so we know the output shapes
    outputs = model.run(None, {n: d.cpu().numpy() for n, d in sample.items()})
    output_shapes = {
        onnx_output.name: output_shape.shape for onnx_output, output_shape in zip(onnx_model_outputs, outputs)
    }
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

        try:
            transfer_to_device_fn(device_result, torch.device("cpu"))
        except Exception:
            console.print(TRANSFER_WARNING)
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
