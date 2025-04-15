from os.path import join
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
