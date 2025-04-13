import typing as ty
from platform import uname
from typing import Any, Dict

import pynvml
from cpuinfo import get_cpu_info
from psutil import cpu_count, cpu_freq, virtual_memory

from nvbenjo.utils import format_num


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


def get_gpu_info() -> ty.List[ty.Dict[str, any]]:
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


def get_system_info() -> Dict[str, Any]:
    sys = uname()
    cpu = get_cpu_info()
    svmem = virtual_memory()
    return {
        "os": {"system": sys.system, "node": sys.node, "release": sys.release, "version": sys.version},
        "cpu": {
            "model": cpu["brand_raw"],
            "architecture": cpu["arch_string_raw"],
            "cores": {
                "physical": cpu_count(logical=False),
                "total": cpu_count(logical=True),
            },
            "frequency": f"{(cpu_freq().max / 1000):.2f} GHz",
        },
        "memory": format_num(svmem.total, bytes=True),
        "gpus": get_gpu_info(),
    }
