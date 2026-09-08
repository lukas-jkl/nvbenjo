import logging
from logging import NullHandler

from rich.console import Console

console = Console()

from .benchmark import benchmark_model
from .system_info import get_cpu_info, get_gpu_info, get_system_info

__all__ = ["benchmark_model", "get_cpu_info", "get_gpu_info", "get_system_info"]


logging.getLogger(__name__).addHandler(NullHandler())
