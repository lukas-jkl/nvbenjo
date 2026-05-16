import os
from os.path import join
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from rich import box
import seaborn as sns
from rich.bar import Bar
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rich.measure import Measurement
from rich.console import Console, ConsoleOptions, RenderResult

from . import console
from .utils import format_num, format_seconds


class MemoryBar:
    """A composite bar showing torch allocator memory (red) and remaining process memory (white)."""

    FULL_BLOCK = "█"

    def __init__(self, torch_mem: float, gpu_mem: float, max_mem: float, width: int = 80):
        self.torch_mem = torch_mem
        self.gpu_mem = gpu_mem
        self.max_mem = max_mem
        self.width = width

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = min(self.width, options.max_width)
        if self.max_mem <= 0:
            yield Text(" " * width)
            return
        gpu_chars = round(self.gpu_mem / self.max_mem * width)
        # If torch_mem == gpu_mem show all as white
        if self.torch_mem < self.gpu_mem:
            torch_chars = round(self.torch_mem / self.max_mem * width)
        else:
            torch_chars = 0
        grey_chars = gpu_chars - torch_chars
        empty_chars = width - gpu_chars

        bar = Text()
        if torch_chars > 0:
            bar.append(self.FULL_BLOCK * torch_chars, style="red")
        if grey_chars > 0:
            bar.append(self.FULL_BLOCK * grey_chars, style="bright_white")
        if empty_chars > 0:
            bar.append(" " * empty_chars)
        yield bar

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        return Measurement(1, self.width)


def _format_memory_label(torch_mem: float, gpu_mem: float) -> Text:
    """Format memory label: 'torch_val (gpu_val)' with red/white styling, or just white if equal."""
    label = Text()
    if torch_mem < gpu_mem:
        label.append(str(format_num(torch_mem, bytes=True)), style="red")
        label.append(f" ({format_num(gpu_mem, bytes=True)})", style="bright_white")
    else:
        label.append(str(format_num(gpu_mem, bytes=True)), style="bright_white")
    return label


def _has_mem(results: pd.Series | pd.DataFrame, key: str):
    if isinstance(results, pd.Series):
        return key in results.index and not pd.isnull(results[key])
    return key in results.columns and not results[key].isnull().all()


def visualize_results(
    results: pd.DataFrame,
    output_dir: str,
    keys: List[str] = [
        "time_cpu_to_device",
        "time_device_to_cpu",
        "time_inference",
        "time_total_batch_normalized",
        "torch_memory_bytes",
        "gpu_memory_bytes",
    ],
    hue="runtime_options",
    col="batch_size",
    kind="bar",
):
    sns.set_style("whitegrid")
    for model in results.model.unique():
        mult_devices = len(results.device.unique()) > 1
        for device in results.device.unique():
            model_device_results = results[(results.model == model) & (results.device == device)]
            if len(model_device_results) == 0:
                continue
            for key in keys:
                if key in model_device_results.columns and not model_device_results[key].isnull().all():
                    sns.catplot(
                        data=model_device_results,
                        x="model",
                        y=key,
                        hue=hue,
                        col=col,
                        kind=kind,
                        palette="dark",
                        alpha=0.6,
                    )
                    device_stem = f"{device}_" if mult_devices else ""
                    os.makedirs(join(output_dir, model), exist_ok=True)
                    plt.savefig(join(output_dir, model, f"{device_stem}{key}.png"))
                    plt.close()

    if len(results.device.unique()) == 1 and len(results.model.unique()) > 1:
        for key in keys:
            if key in results.columns and not results[key].isnull().all():
                sns.catplot(
                    data=results.drop_duplicates().reset_index(drop=True),
                    y=key,
                    hue=hue,
                    col=col,
                    kind=kind,
                    row="model",
                    sharey=True,
                    palette="dark",
                    alpha=0.6,
                )
                device_stem = f"{device}_" if mult_devices else ""
                os.makedirs(join(output_dir, "summary"), exist_ok=True)
                plt.savefig(join(output_dir, "summary", f"summary_{key}.png"))
                plt.close()

    # TODO: maybe also check if only one model type then do same for device


def print_system_info(system_info: dict):
    text_color = "white"
    os_info = system_info["os"]
    os_string = os_info["system"].replace("Linux", "Linux 🐧")
    cpu_info = system_info["cpu"]
    cuda_info = system_info["cuda"]
    gpu_infos = cuda_info.get("gpus", [])
    driver_version = cuda_info.get("driver_version", "None")
    cudnn_version = cuda_info.get("cudnn_version", "None")
    torch_version = cuda_info.get("torch_version", "None")

    title = Text("System Information", style="bold cyan")

    content = Text()
    content.append("\n")
    content.append(f"💻️ {os_info['node']}\n", style="bold")
    content.append("OS:   ", style="bold yellow")
    content.append(f"{os_string} - {os_info['version']} ({os_info['release']})\n", style=text_color)
    content.append("CPU:  ", style="bold magenta")
    content.append(f"{cpu_info['model']} ({cpu_info['architecture']})\t", style=text_color)
    content.append("Cores: ", style=f"{text_color} bold")
    content.append(f"{cpu_info['cores']}\n", style=text_color)

    content.append("GPUs", style="bold green")
    if len(gpu_infos) > 0:
        content.append(f" (Driver {driver_version}, Torch {torch_version}, CuDNN {cudnn_version})\n", style="green")
        for gpu_info in gpu_infos:
            content.append("   ", style="bold blue")
            content.append(f"{gpu_info['name']} @ {gpu_info['clock_gpu']} ", style=text_color)
            content.append(f"({gpu_info['memory']} @ {gpu_info['clock_mem']})", style=text_color)
            content.append(f" - {gpu_info['architecture']} cap {gpu_info['cuda_capability']}\n", style=text_color)
    else:
        content.append("  None\n", style=text_color)

    console.print(Panel(content, title=title, border_style="blue", padding=(0, 2)))


def _print_device_results(model_results: pd.Series | pd.DataFrame, model: str, device: str, custom_metric_keys: List):
    # Create a rich table for each model+device combination
    table = Table(
        title=f"Model: {model} on Device: {device}",
        show_header=True,
        header_style="bold",
        show_lines=True,
        title_style="bold",
    )

    # Get grouped results
    device_results = model_results[model_results.device == device]
    device_results = device_results.drop(columns=["device"])
    device_results = device_results.groupby(["model", "runtime_options", "batch_size"]).mean()
    device_results["device"] = device  # Add device column back for display
    print_result = device_results.reset_index()

    # Remove columns where all values are None
    print_result = print_result.dropna(axis="columns", how="all")

    # Merge memory columns into a single combined column
    has_process_mem = _has_mem(print_result, "gpu_memory_bytes")
    has_torch_mem = _has_mem(print_result, "torch_memory_bytes")
    if has_process_mem and has_torch_mem:
        print_result["memory"] = list(zip(print_result["torch_memory_bytes"], print_result["gpu_memory_bytes"]))
        print_result = print_result.drop(columns=["torch_memory_bytes", "gpu_memory_bytes"])
        memory_col_header = "Memory: Torch (Process)"
    elif has_process_mem:
        print_result["memory"] = print_result["gpu_memory_bytes"]
        print_result = print_result.drop(columns=["gpu_memory_bytes"])
        memory_col_header = "Process Memory"
    elif has_torch_mem:
        print_result["memory"] = print_result["torch_memory_bytes"]
        print_result = print_result.drop(columns=["torch_memory_bytes"])
        memory_col_header = "Torch Memory"
    else:
        raise RuntimeError("Invalid Memory Column")

    # Format values for display
    for column in print_result.columns:
        if column == "memory":
            if has_process_mem and has_torch_mem:
                print_result[column] = print_result[column].apply(
                    lambda x: f"{format_num(x[0], bytes=True)} ({format_num(x[1], bytes=True)})"
                )
            else:
                print_result[column] = print_result[column].apply(lambda x: f"{format_num(x, bytes=True)}")
        elif column == "time_total_batch_normalized":
            top3 = print_result.time_total_batch_normalized.nsmallest(3).index
            print_result[column] = print_result[column].apply(format_seconds)
            for i, emoji in enumerate(["🥇", "🥈", "🥉"][: len(top3)]):
                print_result.loc[top3[i], column] = f"{emoji} {print_result.loc[top3[i], column]}"
        elif column.startswith("time"):
            print_result[column] = print_result[column].apply(format_seconds)
        elif column == "device":
            print_result[column] = print_result[column].apply(lambda x: f"{x}")
        elif column in custom_metric_keys:
            print_result[column] = print_result[column].apply(lambda x: f"{x:.3f}")

    # Add columns to the table
    for col in print_result.columns:
        # Set column styles based on data type
        if col == "model":
            style = "bold green"
        elif col == "runtime_options":
            style = "bold blue"
        elif col == "batch_size":
            style = "bold yellow"
        elif col == "time_total_batch_normalized":
            style = "bold cyan"
        elif col.startswith("time"):
            style = None
        elif col == "memory":
            col = memory_col_header
            style = None
        else:
            style = None

        # Format column names for better display
        display_name = col.replace("_", " ").title()
        table.add_column(header=display_name, style=style, justify="right")

    # Add rows to the table
    for _, row in print_result.iterrows():
        table.add_row(*[str(value) for value in row.values])

    # Display the table in a panel
    console.print(Panel(table, border_style="dim", padding=(0, 1)))


def _print_summary_plot(results: pd.Series | pd.DataFrame, custom_metric_keys: List):
    default_metric = "time_total_batch_normalized"
    default_metric_title = "Time Batch Normalized"
    if not custom_metric_keys:
        first_metric = default_metric
        metric_title = default_metric_title
    else:
        first_metric = custom_metric_keys[0]
        metric_title = first_metric

    table = Table(
        show_header=True,
        header_style="bold",
        box=box.SIMPLE,
        padding=(0, 1),
        title_justify="left",
        title_style="bold",
    )
    table.add_column("Model", style="bold")
    table.add_column("Runtime Options", style="bold")
    table.add_column("Batch Size", style="bold")
    table.add_column("Device", style="bold")
    table.add_column(metric_title, header_style="bold cyan")
    table.add_column("", justify="right")
    table.add_column("", justify="right")
    mem_header = Text()

    if _has_mem(results, "torch_memory_bytes"):
        mem_header.append("Torch Memory", style="bold red")
        mem_header.append(" / ", style="bold")
    mem_header.append("Process Memory", style="bold bright_white")
    table.add_column(mem_header)
    table.add_column("", justify="right")

    max_first_metric = results[first_metric].max().item()
    max_process_mem = (
        results.gpu_memory_bytes.max().item()
        if _has_mem(results, "gpu_memory_bytes")
        else results.torch_memory_bytes.max().item()
    )
    for model in results.model.unique():
        model_results = results[results.model == model]
        for device in model_results.device.unique():
            # Get grouped results and mean over runs
            device_results = model_results[model_results.device == device]
            device_results = device_results.drop(columns=["device"])
            device_results = device_results.groupby(["model", "runtime_options", "batch_size"]).mean()
            device_results["device"] = device  # Add device column back for display
            print_result = device_results.reset_index()

            # Remove columns where all values are None
            print_result = print_result.dropna(axis="columns", how="all")

            # Get grouped results
            device_results = model_results[model_results.device == device]

            print_result = print_result.sort_values(first_metric)
            for _, res in print_result.iterrows():
                first_val = res[first_metric]
                gpu_mem_val = res.gpu_memory_bytes
                torch_mem_val = res.torch_memory_bytes if _has_mem(res, "torch_memory_bytes") else gpu_mem_val
                table.add_row(
                    Text(model, style="green"),
                    Text(res.runtime_options, style="blue"),
                    Text(str(res.batch_size), style="yellow"),
                    res.device,
                    Bar(begin=0, size=max_first_metric, end=first_val, width=80, color="cyan"),
                    Text(
                        format_seconds(first_val) if first_metric == default_metric else str(format_num(first_val)),
                        style="cyan",
                    ),
                    "   ",
                    MemoryBar(torch_mem=torch_mem_val, gpu_mem=gpu_mem_val, max_mem=max_process_mem, width=80),
                    _format_memory_label(torch_mem_val, gpu_mem_val),
                )

    console.print(Panel(table, border_style="dim", padding=(0, 1)))


def print_results(results: pd.DataFrame, custom_metric_keys: List[str] = []):
    for model in results.model.unique():
        model_results = results[results.model == model]
        for device in model_results.device.unique():
            _print_device_results(model_results, model, device, custom_metric_keys)

    _print_summary_plot(results, custom_metric_keys)
