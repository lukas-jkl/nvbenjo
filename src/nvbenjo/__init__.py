import logging
import os
import sys
from logging import NullHandler

from rich.console import Console

# rich falls back to 80 columns without a TTY, which truncates the results table
console = Console(width=None if sys.stdout.isatty() or os.environ.get("COLUMNS") else 200)

from .benchmark import benchmark_model
from .system_info import get_cpu_info, get_gpu_info, get_system_info

__all__ = ["benchmark_model", "get_cpu_info", "get_gpu_info", "get_system_info"]


logging.getLogger(__name__).addHandler(NullHandler())
