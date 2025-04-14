import logging

import hydra
import pandas as pd
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm
from hydra.core.config_store import ConfigStore


from nvbenjo.cfg import BenchConfig
import os
from os.path import join

from nvbenjo.benchmark import benchmark_model
from importlib.resources import files

logger = logging.getLogger("nvbenjo")

cs = ConfigStore.instance()
cs.store(name="base_config", node=BenchConfig)


# @hydra.main(version_base=None, config_path="./conf", config_name="default")
@hydra.main(version_base=None, config_path=os.path.join(files("nvbenjo").joinpath("conf")), config_name="default")
def nvbenjo(cfg: BenchConfig):
    run(cfg)


def run(cfg: BenchConfig) -> None:
    if cfg.output_dir is None:
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    else:
        output_dir = cfg.output_dir

    nvbenjo_cfg = cfg.nvbenjo
    raw_results = []
    human_readable_results = {}
    logger.info(f"Starting benchmark, output-dir {output_dir}")

    model_iter = tqdm(nvbenjo_cfg.models, leave=True)
    for model_cfg in model_iter:
        model_iter.set_description(model_cfg.name)
        model_raw_results, model_human_readable_results = benchmark_model(model_cfg)
        raw_results.append(model_raw_results)
        human_readable_results.update(model_human_readable_results)

    raw_results = pd.concat(raw_results)

    with open(join(output_dir, "out.yaml"), "w") as f:
        yaml.dump(human_readable_results, f, width=1000.0)
    raw_results.to_csv(join(output_dir, "out.csv"))
    with open(join(output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    nvbenjo()
