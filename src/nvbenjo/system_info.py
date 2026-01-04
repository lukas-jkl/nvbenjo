from __future__ import annotations

from platform import uname
from typing import Any
import logging

import pynvml
from cpuinfo import get_cpu_info
import psutil

from .utils import format_num

logger = logging.getLogger(__name__)


def _get_architecture_name_from_version(version: int) -> str:
    version_names = {
        pynvml.NVML_DEVICE_ARCH_KEPLER: "Kepler",
        pynvml.NVML_DEVICE_ARCH_MAXWELL: "Maxwell",
        pynvml.NVML_DEVICE_ARCH_PASCAL: "Pascal",
        pynvml.NVML_DEVICE_ARCH_VOLTA: "Volta",
        pynvml.NVML_DEVICE_ARCH_TURING: "Truing",
        pynvml.NVML_DEVICE_ARCH_AMPERE: "Ampere",
        pynvml.NVML_DEVICE_ARCH_ADA: "Ada",
        pynvml.NVML_DEVICE_ARCH_HOPPER: "Hopper",
        pynvml.NVML_DEVICE_ARCH_BLACKWELL: "Blackwell",
    }
    return f"Version {version} ({version_names.get(version, 'Unknown')})"


def get_gpu_info() -> list[dict[str, Any]]:
    """Retrieve information about GPUs in the system.

    Includes information such as name, architecture, memory, clock speeds, CUDA capability, and driver version.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries containing GPU information.
    """
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    infos = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        compute_capability = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        clock_gpu = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        clock_mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        # clock_sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        # clock_video = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_VIDEO)

        infos.append(
            {
                "idx": i,
                "name": pynvml.nvmlDeviceGetName(handle),
                "architecture": _get_architecture_name_from_version(pynvml.nvmlDeviceGetArchitecture(handle)),
                "memory": format_num(pynvml.nvmlDeviceGetMemoryInfo(handle).total, bytes=True),
                "clock_gpu": f"{clock_gpu} Mhz",
                "clock_mem": f"{clock_mem} Mhz",
                "cuda_capability": f"{compute_capability[0]}.{compute_capability[1]}",
                "driver": str(pynvml.nvmlSystemGetDriverVersion()),
            }
        )
    pynvml.nvmlShutdown()
    return infos


def get_gpu_power_usage(device_index: int) -> float:
    pynvml.nvmlInit()
    usage = pynvml.nvmlDeviceGetPowerUsage(pynvml.nvmlDeviceGetHandleByIndex(device_index))
    pynvml.nvmlShutdown()
    return usage


def get_system_info() -> dict[str, Any]:
    """Retrieve system information.

    Collects information about the operating system, CPU, memory, and GPU.

    Returns
    -------
    dict[str, Any]
        A dictionary containing system information.
    """
    sys = uname()
    cpu = get_cpu_info()
    svmem = psutil.virtual_memory()
    try:
        gpus = get_gpu_info()
    except pynvml.NVMLError_LibraryNotFound:  # type: ignore
        logger.warning("NVIDIA driver not found")
        gpus = {}
    if hasattr(psutil, "cpu_freq") and psutil.cpu_freq() is not None:
        cpufreq = psutil.cpu_freq().max
    else:
        cpufreq = 0.0
    return {
        "os": {"system": sys.system, "node": sys.node, "release": sys.release, "version": sys.version},
        "cpu": {
            "model": cpu["brand_raw"],
            "architecture": cpu["arch_string_raw"],
            "cores": {
                "physical": psutil.cpu_count(logical=False),
                "total": psutil.cpu_count(logical=True),
            },
            "frequency": f"{(cpufreq / 1000):.2f} GHz",
        },
        "memory": format_num(svmem.total, bytes=True),
        "gpus": gpus,
    }
