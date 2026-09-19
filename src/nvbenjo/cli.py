import logging
import os
import sys
from importlib.metadata import version
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


@hydra.main(version_base=None, config_path=os.path.join(str(files("nvbenjo").joinpath("conf"))), config_name="default")
def _run_nvbenjo(cfg: BenchConfig | DictConfig):
    run(cfg)


def run(cfg: BenchConfig | DictConfig) -> None:
    logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(console=console)])
    models = instantiate_model_configs(cfg)
    if cfg.output_dir is not None:
        output_dir = os.path.abspath(cfg.output_dir)
    else:
        output_dir = None

    system_info = get_system_info()

    if output_dir is not None:
        logger.info(f"Starting benchmark, output-dir {output_dir}")

    if len(models) == 0:
        logger.info("No models to benchmark, please specify a configuration or override via the command line.")
        return
    results = benchmark_models(models, measure_memory=cfg.nvbenjo.measure_memory)

    if output_dir is not None:
        results.to_csv(join(output_dir, "out.csv"))
        with open(join(output_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

    custom_metric_keys = _collect_custom_metric_keys(models)
    if output_dir is not None:
        logger.info("Generating plots...")
        plot.visualize_results(
            results,
            keys=[
                "time_cpu_to_device",
                "time_device_to_cpu",
                "time_inference",
                "time_total_batch_normalized",
                "torch_memory_bytes",
                "gpu_memory_bytes",
            ]
            + custom_metric_keys,
            output_dir=output_dir,
        )
    plot.print_system_info(system_info)
    plot.print_results(results, custom_metric_keys=custom_metric_keys)
    logger.info(f"Benchmark finished, outputs in: {output_dir}")


def _collect_custom_metric_keys(models: dict) -> list[str]:
    """Custom batch metric keys of all models, de-duplicated and in config order.

    Order matters: the first key picks the metric for the summary table
    """
    return list(dict.fromkeys(key for mcfg in models.values() for key in mcfg.custom_batchmetrics))


_CONFIG_NAME_FLAGS = ("-cn", "--config-name")
_CONFIG_DIR_FLAGS = ("-cd", "--config-dir")


def _find_flag(argv: list[str], flags: tuple[str, ...]) -> tuple[int, str, str] | None:
    """Locate ``flag <value>`` or ``flag=<value>`` in ``argv``.

    Both spellings have to be handled.
    Returns ``(index, prefix, value)``, where the value is
    rewritten with ``argv[index] = prefix + new_value``.
    """
    for i, arg in enumerate(argv):
        if arg in flags:
            return (i + 1, "", argv[i + 1]) if i + 1 < len(argv) else None
        for flag in flags:
            if arg.startswith(f"{flag}="):
                return i, f"{flag}=", arg[len(flag) + 1 :]
    return None


def _fix_config_path():
    # NOTE: this is a workaround to allow specifying config file with full path
    #       since hydra only allows config name and config dir
    #       so for -cn /path/to/config.yaml we add -cd /path/to and change -cn to config.yaml
    #       an explicit -cd/--config-dir always wins, so we leave argv alone in that case
    if _find_flag(sys.argv, _CONFIG_DIR_FLAGS) is not None:
        return

    found = _find_flag(sys.argv, _CONFIG_NAME_FLAGS)
    if found is None:
        return

    index, prefix, config_name = found
    config_dir = os.path.dirname(config_name)
    if not config_dir:
        return

    config_file = os.path.basename(config_name)
    sys.argv[index] = prefix + config_file
    sys.argv += ["-cd", config_dir]
    logger.debug("Sys argv: " + str(sys.argv))
    logger.debug(f"Adjusted config path, using -cd {config_dir} and -cn {config_file}")


def nvbenjo():
    if "--version" in sys.argv:
        print(f"nvbenjo {version('nvbenjo')}")
        sys.exit(0)
    _fix_config_path()
    _run_nvbenjo()


if __name__ == "__main__":
    nvbenjo()
