import logging
import os
from importlib.resources import files
from os.path import join
import typing as ty

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from nvbenjo import plot
from nvbenjo.benchmark import benchmark_models
from nvbenjo.cfg import BenchConfig, instantiate_model_configs
from nvbenjo.system_info import get_system_info
from nvbenjo import console
from rich.logging import RichHandler

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="base_config", node=BenchConfig)


@hydra.main(version_base=None, config_path=os.path.join(files("nvbenjo").joinpath("conf")), config_name="default")
def nvbenjo(cfg: ty.Union[BenchConfig, DictConfig]):
    run(cfg)


def run(cfg: ty.Union[BenchConfig, DictConfig]) -> None:
    logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(console=console)])
    models = instantiate_model_configs(cfg)
    if cfg.output_dir is not None:
        output_dir = os.path.abspath(cfg.output_dir)

    system_info = get_system_info()

    if cfg.output_dir is not None:
        logger.info(f"Starting benchmark, output-dir {output_dir}")

    if len(models) == 0:
        logger.info("No models to benchmark, please specify a configuration or override via the command line.")
        return
    results = benchmark_models(models, measure_memory=cfg.nvbenjo.measure_memory, profile=cfg.nvbenjo.profile)

    if cfg.output_dir is not None:
        results.to_csv(join(output_dir, "out.csv"))
        with open(join(output_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

    if cfg.output_dir is not None:
        logger.info("Generating plots...")
        plot.visualize_results(results, output_dir=output_dir)
    plot.print_system_info(system_info)
    plot.print_results(results)
    logger.info(f"Benchmark finished, outputs in: {output_dir}")


if __name__ == "__main__":
    nvbenjo()
