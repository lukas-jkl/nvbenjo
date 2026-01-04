import os
from os.path import join
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import console
from .utils import format_num, format_seconds


def visualize_results(
    results: pd.DataFrame,
    output_dir: str,
    keys: List[str] = [
        "time_cpu_to_device",
        "time_device_to_cpu",
        "time_inference",
        "time_total_batch_normalized",
        "memory_bytes",
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
    gpu_infos = system_info["gpus"]
    driver_version = set(gpu_info["driver"] for gpu_info in gpu_infos)
    driver_version = driver_version.pop() if len(driver_version) == 1 else driver_version

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
        content.append(f" (Driver {driver_version})\n", style="green")
        for gpu_info in gpu_infos:
            content.append("   ", style="bold blue")
            content.append(f"{gpu_info['name']} @ {gpu_info['clock_gpu']} ", style=text_color)
            content.append(f"({gpu_info['memory']} @ {gpu_info['clock_mem']})", style=text_color)
            content.append(f" - {gpu_info['architecture']}\n", style=text_color)
    else:
        content.append("  None\n", style=text_color)

    console.print(Panel(content, title=title, border_style="blue", padding=(1, 4)))


def print_results(results: pd.DataFrame, custom_metric_keys: List[str] = []):
    console.print("\n")
    for model in results.model.unique():
        model_results = results[results.model == model]
        for device in model_results.device.unique():
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

            # Format values for display
            for column in print_result.columns:
                if column == "time_total_batch_normalized":
                    top3 = print_result.time_total_batch_normalized.nsmallest(3).index
                    print_result[column] = print_result[column].apply(format_seconds)
                    for i, emoji in enumerate(["🥇", "🥈", "🥉"][: len(top3)]):
                        print_result.loc[top3[i], column] = f"{emoji} {print_result.loc[top3[i], column]}"
                elif column.startswith("time"):
                    print_result[column] = print_result[column].apply(format_seconds)
                elif "bytes" in column:
                    print_result[column] = print_result[column].apply(format_num, bytes=True)
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
                elif "memory" in col:
                    style = "red"
                else:
                    style = None

                # Format column names for better display
                display_name = col.replace("_", " ").title()
                table.add_column(display_name, style=style, justify="right")

            # Add rows to the table
            for _, row in print_result.iterrows():
                table.add_row(*[str(value) for value in row.values])

            # Display the table in a panel
            console.print(Panel(table, border_style="dim", padding=(1, 2)))
            console.print("\n")
