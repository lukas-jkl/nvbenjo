import typing as ty
from dataclasses import dataclass, field
from abc import ABC

from omegaconf import DictConfig, OmegaConf, open_dict
from .utils import PrecisionType
from hydra.utils import instantiate
from contextlib import nullcontext


@dataclass
class TorchRuntimeConfig:
    compile: bool = False
    compile_kwargs: dict = field(default_factory=dict)
    precision: PrecisionType = PrecisionType.FP32


@dataclass
class OnnxRuntimeConfig:
    execution_providers: ty.Optional[tuple[str, ...]] = None
    graph_optimization_level: str = (
        "ORT_ENABLE_ALL"  # 99 ORT_ENABLE_ALL, 3 ORT_ENABLE_LAYOUT, 1 ORT_ENABLE_BASIC, 0 ORT_DISABLE_ALL
    )
    intra_op_num_threads: int = 1
    inter_op_num_threads: int = 0
    log_severity_level: int = 2  # Warning
    provider_options: ty.Sequence[dict[ty.Any, ty.Any]] | None = None


@dataclass
class BaseModelConfig(ABC):
    name: str = "resnet"
    type_or_path: str = "torchvision:wide_resnet101_2"
    kwargs: dict = field(default_factory=dict)
    shape: tuple = ("B", 3, 224, 224)
    num_warmup_batches: int = 5
    num_batches: int = 50
    batch_sizes: tuple = (16, 32)
    devices: tuple[str] = ("cpu",)
    runtime_options: dict[str, ty.Any] = field(default_factory=dict)


@dataclass
class TorchModelConfig(BaseModelConfig):
    model_kwargs: dict = field(default_factory=dict)
    runtime_options: dict[str, TorchRuntimeConfig] = field(default_factory=lambda: {"default": TorchRuntimeConfig()})

    def __post_init__(self):
        for i, (key, opt) in enumerate(self.runtime_options.items()):
            if isinstance(opt, DictConfig):
                self.runtime_options[key] = OmegaConf.structured(TorchRuntimeConfig(**OmegaConf.to_container(opt)))  # type: ignore


@dataclass
class OnnxModelConfig(BaseModelConfig):
    runtime_options: dict[str, OnnxRuntimeConfig] = field(default_factory=lambda: {"default": OnnxRuntimeConfig()})

    def __post_init__(self):
        for i, (key, opt) in enumerate(self.runtime_options.items()):
            if isinstance(opt, DictConfig):
                self.runtime_options[key] = OmegaConf.structured(OnnxRuntimeConfig(**OmegaConf.to_container(opt)))  # type: ignore


@dataclass
class NvbenjoConfig:
    measure_memory: bool = True
    profile: bool = False
    models: dict[str, ty.Any] = field(default_factory=lambda: dict())


@dataclass
class BenchConfig:
    nvbenjo: NvbenjoConfig = field(default_factory=NvbenjoConfig)
    output_dir: ty.Optional[str] = None


def instantiate_model_configs(cfg: ty.Union[BenchConfig, DictConfig]) -> dict[str, BaseModelConfig]:
    models = {}
    runtimes = {}
    for model_name, model in cfg.nvbenjo.models.items():
        ctxt = open_dict(model) if isinstance(model, DictConfig) else nullcontext()
        if "_target_" not in model:
            with ctxt:
                if model["type_or_path"].endswith(".onnx"):
                    cfg.nvbenjo.models[model_name]["_target_"] = (
                        f"{OnnxModelConfig.__module__}.{OnnxModelConfig.__qualname__}"
                    )
                    cfg.nvbenjo.models[model_name]["_convert_"] = "all"
                    if "runtime_options" in model:
                        runtimes[model_name] = {}
                        for runtime_name in model["runtime_options"].keys():
                            cfg.nvbenjo.models[model_name]["runtime_options"][runtime_name]["_target_"] = (
                                f"{OnnxRuntimeConfig.__module__}.{OnnxRuntimeConfig.__qualname__}"
                            )
                            runtimes[model_name][runtime_name] = instantiate(
                                cfg.nvbenjo.models[model_name]["runtime_options"][runtime_name]
                            )
                else:
                    cfg.nvbenjo.models[model_name]["_target_"] = (
                        f"{TorchModelConfig.__module__}.{TorchModelConfig.__qualname__}"
                    )
                    cfg.nvbenjo.models[model_name]["_convert_"] = "all"
                    if "runtime_options" in model:
                        runtimes[model_name] = {}
                        for runtime_name in model["runtime_options"].keys():
                            cfg.nvbenjo.models[model_name]["runtime_options"][runtime_name]["_target_"] = (
                                f"{TorchRuntimeConfig.__module__}.{TorchRuntimeConfig.__qualname__}"
                            )
                            runtimes[model_name][runtime_name] = instantiate(
                                cfg.nvbenjo.models[model_name]["runtime_options"][runtime_name]
                            )
                            runtimes[model_name][runtime_name].precision = PrecisionType[
                                cfg.nvbenjo.models[model_name]["runtime_options"][runtime_name]["precision"]
                            ]

        models[model_name] = instantiate(model) if isinstance(model, DictConfig) else model
        if model_name in runtimes:
            models[model_name].runtime_options = runtimes[model_name]

    return models
