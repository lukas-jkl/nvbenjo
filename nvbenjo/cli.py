import logging
import os
from importlib.resources import files
from os.path import join

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from nvbenjo import plot
from nvbenjo.benchmark import benchmark_models
from nvbenjo.cfg import BenchConfig
from nvbenjo.system_info import get_system_info
from nvbenjo import console
from rich.logging import RichHandler

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="base_config", node=BenchConfig)


@hydra.main(version_base=None, config_path=os.path.join(files("nvbenjo").joinpath("conf")), config_name="default")
def nvbenjo(cfg: BenchConfig):
    run(cfg)


def run(cfg: BenchConfig) -> None:
    logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(console=console)])
    output_dir = cfg.output_dir

    system_info = get_system_info()

    logger.info(f"Starting benchmark, output-dir {output_dir}")
    results = benchmark_models(cfg.nvbenjo.models)

    results.to_csv(join(output_dir, "out.csv"))
    with open(join(output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    plot.print_system_info(system_info)
    plot.visualize_results(results, output_dir=output_dir)
    plot.print_results(results)
    logger.info(f"Benchmark finished, output-dir {output_dir}")


if __name__ == "__main__":
    nvbenjo()
