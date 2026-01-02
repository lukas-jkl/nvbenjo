import logging
import os
import sys
import typing as ty
from importlib.resources import files
from os.path import join

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler

from . import console, plot
from .benchmark import benchmark_models
from .cfg import BenchConfig, instantiate_model_configs
from .system_info import get_system_info

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="base_config", node=BenchConfig)


@hydra.main(version_base=None, config_path=os.path.join(files("nvbenjo").joinpath("conf")), config_name="default")
def _run_nvbenjo(cfg: ty.Union[BenchConfig, DictConfig]):
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
    results = benchmark_models(models, measure_memory=cfg.nvbenjo.measure_memory)

    if cfg.output_dir is not None:
        results.to_csv(join(output_dir, "out.csv"))
        with open(join(output_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

    custom_metric_keys = list(set(sum([list(mcfg.custom_batchmetrics.keys()) for mcfg in models.values()], [])))
    if cfg.output_dir is not None:
        logger.info("Generating plots...")
        plot.visualize_results(
            results,
            keys=[
                "time_cpu_to_device",
                "time_device_to_cpu",
                "time_inference",
                "time_total_batch_normalized",
                "memory_bytes",
            ]
            + custom_metric_keys,
            output_dir=output_dir,
        )
    plot.print_system_info(system_info)
    plot.print_results(results, custom_metric_keys=custom_metric_keys)
    logger.info(f"Benchmark finished, outputs in: {output_dir}")


def _fix_config_path():
    # NOTE: this is a workaround to allow specifying config file with full path
    #       since hydra only allows config name and config dir
    #       so for -cn /path/to/config.yaml we add -cd /path/to and change -cn to config.yaml
    if "-cn" in sys.argv or "--config-name" in sys.argv and "-cd" not in sys.argv and "--config-dir" not in sys.argv:
        arg_index = sys.argv.index("-cn") if "-cn" in sys.argv else sys.argv.index("--config-name")  # type: ignore
        cfg_index = arg_index + 1
        if cfg_index <= len(sys.argv) - 1:
            config_name = sys.argv[cfg_index]
            if os.path.dirname(config_name):
                sys.argv.append("-cd")
                sys.argv.append(os.path.dirname(config_name))
                sys.argv[cfg_index] = os.path.basename(config_name)
                logger.debug("Sys argv: " + str(sys.argv))
                logger.debug(
                    f"Adjusted config path, using -cd {os.path.dirname(config_name)} and -cn {os.path.basename(config_name)}"
                )


def nvbenjo():
    _fix_config_path()
    _run_nvbenjo()


if __name__ == "__main__":
    nvbenjo()
