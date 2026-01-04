import logging
from logging import NullHandler
from rich.console import Console


console = Console()

from .benchmark import benchmark_model  # noqa: E402
from .system_info import get_cpu_info, get_gpu_info, get_system_info  # noqa: E402

__all__ = [get_system_info, get_gpu_info, get_cpu_info, benchmark_model]


logging.getLogger(__name__).addHandler(NullHandler())
