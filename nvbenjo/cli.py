import logging
import os
from importlib.resources import files
from os.path import join

import hydra
import yaml
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from nvbenjo import plot
from nvbenjo.benchmark import benchmark_models
from nvbenjo.cfg import BenchConfig
from nvbenjo.system_info import get_system_info

logger = logging.getLogger("nvbenjo")

cs = ConfigStore.instance()
cs.store(name="base_config", node=BenchConfig)


@hydra.main(version_base=None, config_path=os.path.join(files("nvbenjo").joinpath("conf")), config_name="default")
def nvbenjo(cfg: BenchConfig):
    run(cfg)


def run(cfg: BenchConfig) -> None:
    if cfg.output_dir is None:
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    else:
        output_dir = cfg.output_dir
        os.makedirs(output_dir, exist_ok=True)

    system_info = get_system_info()

    logger.info(f"Starting benchmark, output-dir {output_dir}")
    raw_results, human_readable_results = benchmark_models(cfg.nvbenjo.models)

    with open(join(output_dir, "out.yaml"), "w") as f:
        yaml.dump(human_readable_results, f, width=1000.0)
    raw_results.to_csv(join(output_dir, "out.csv"))
    with open(join(output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    plot.print_system_info(system_info)
    plot.visualize_results(raw_results, output_dir=output_dir)
    plot.print_results(raw_results)


if __name__ == "__main__":
    nvbenjo()
