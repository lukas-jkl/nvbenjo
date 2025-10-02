from nvbenjo import console
import onnxruntime as ort
import torch
import time
import os
import pandas as pd
from nvbenjo.torch_utils import transfer_to_device
import typing as ty


def get_model(
    type_or_path: str, device: torch.device, verbose=False, **kwargs
) -> ort.capi.onnxruntime_inference_collection.InferenceSession:
    if not type_or_path.endswith(".onnx") and os.path.isfile(type_or_path):
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
    sess.io_binding()
    return sess


def measure_repeated_inference_timing(
    model: ort.capi.onnxruntime_inference_collection.InferenceSession,
    sample: torch.Tensor,
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

    for _ in range(num_runs):
        start_on_cpu = time.perf_counter()
        device_sample = transfer_to_device_fn(sample, model_device)

        if model_device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            stop_event = torch.cuda.Event(enable_timing=True)
            start_event.record()  # For GPU timing
        start_on_device = time.perf_counter()  # For CPU timing

        io_binding = model.io_binding()
        inputs = model.get_inputs()
        device_id = 0 if model_device.index is None else model_device.index
        if isinstance(device_sample, (list, tuple)):
            for i, input in enumerate(device_sample):
                io_binding.bind_input(
                    name=inputs[i].name,
                    device_type=model_device.type,
                    device_id=device_id,
                    element_type=input.numpy().dtype,
                    shape=input.shape,
                    buffer_ptr=input.data_ptr(),
                )
        elif isinstance(device_sample, dict):
            for i, (name, input) in enumerate(device_sample.items()):
                io_binding.bind_input(
                    name=name,
                    device_type=model_device.type,
                    device_id=device_id,
                    element_type=input.numpy().dtype,
                    shape=input.shape,
                    buffer_ptr=input.data_ptr(),
                )
        else:
            raise ValueError(f"Invalid input type {type(device_sample)}. Must be one of list, tuple, dict")

        outputs = model.get_outputs()
        device_result = []
        for output in outputs:
            s = output.shape
            for i, si in enumerate(s):
                if si == "batch_size":
                    s[i] = batch_size

            # Convert ONNX type to PyTorch dtype
            dtype_map = {
                "tensor(float)": torch.float32,
                "tensor(float16)": torch.float16,
                "tensor(int64)": torch.int64,
                "tensor(int32)": torch.int32,
                "tensor(bool)": torch.bool,
            }
            torch_dtype = dtype_map.get(output.type, torch.float32)  # default to float32 if type not found

            output_tensor = torch.empty(size=s, dtype=torch_dtype, device=model_device)
            io_binding.bind_output(
                name=output.name,
                device_type=model_device.type,
                device_id=device_id,
                element_type=output_tensor.numpy().dtype,
                shape=output_tensor.shape,
                buffer_ptr=output_tensor.data_ptr(),
            )
            device_result.append(output_tensor)

        model.run_with_iobinding(io_binding)

        if model_device.type == "cuda":
            stop_event.record()
            torch.cuda.synchronize()
            # elapsed_on_device = stop_event.elapsed_time(start_event)
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
