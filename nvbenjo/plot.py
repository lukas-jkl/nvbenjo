from os.path import join
from typing import List
from tabulate import tabulate

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nvbenjo.utils import format_num, format_seconds


def visualize_results(
    results: pd.DataFrame,
    output_dir: str,
    keys: List[str] = ["time_cpu_to_device", "time_inference", "time_device_to_cpu", "time_total", "memory_bytes"],
    hue="precision",
    col="batch_size",
    kind="bar",
):
    sns.set_style("whitegrid")
    for model in results.model.unique():
        mult_devices = len(results.device_idx.unique()) > 1
        for device_idx in results.device_idx.unique():
            model_device_results = results[results.model == model][results.device_idx == device_idx]
            for key in keys:
                sns.catplot(data=model_device_results, x="model", y=key, hue=hue, col=col, kind=kind)
                device_stem = f"{device_idx}_" if mult_devices else ""
                plt.savefig(join(output_dir, f"{model}_{device_stem}{key}.png"))


def print_results(
    results: pd.DataFrame,
):
    print("\n")
    for model in results.model.unique():
        model_results = results[results.model == model]
        for device_idx in model_results.device_idx.unique():
            print(f"Model: {model} Device: {device_idx}")
            device_results = model_results[model_results.device_idx == device_idx]
            device_results = device_results.groupby(["model", "precision", "batch_size"]).mean()
            for column in device_results.columns:
                if "time" in column:
                    device_results[column] = device_results[column].apply(format_seconds)
                elif "bytes" in column:
                    device_results[column] = device_results[column].apply(format_num, bytes=True)
                elif column == "device_idx":
                    device_results[column] = device_results[column].apply(lambda x: f"{int(x)}")
                else:
                    device_results[column] = device_results[column]
                print_result = device_results.reset_index()

            print(
                tabulate(
                    print_result, headers=print_result.columns, showindex=False, tablefmt="fancy_grid", floatfmt=".3f"
                )
            )
            print("\n")
