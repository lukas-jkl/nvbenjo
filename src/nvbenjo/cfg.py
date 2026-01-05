import os
import typing as ty
from abc import ABC
from contextlib import nullcontext
from dataclasses import dataclass, field

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from .utils import PrecisionType, ProviderType


@dataclass
class BaseModelConfig(ABC):
    """Base model configuration

    Parameters
    ----------
    name : str
        Name of the model.
    type_or_path : str
        Model type or path. Can be a local file path or a model identifier.
    kwargs : dict
        Additional keyword arguments to pass when instantiating the model.
    shape : tuple
        Input shape of the model. Use "B" or "batch_size" to denote the batch size dimension.
    num_warmup_batches : int
        Number of warm-up batches to run before measuring performance.
    num_batches : int
        Number of batches to run for performance measurement.
    batch_sizes : tuple
        Tuple of batch sizes to benchmark.
    devices : tuple of str
        Tuple of device names to benchmark on (e.g., 'cpu', 'cuda:0').
    runtime_options : dict[str, ty.Any]
        Dictionary mapping runtime names to their specific runtime configurations.
    custom_batchmetrics : dict[str, float]
        Dictionary of custom batch metrics to include in the benchmark results.
        These are calculated as `value / time_total_batch_normalized`.
        For example if each batch is a frame you can calculate FPS as:
            custom_batchmetrics:
                fps: 1.0
                batch-per-second: 1.0
        or if each batch is 3s of audio you can calculate the real time factor as:
            custom_batchmetrics:
                real-time-factor: 1/3
    """

    name: str = "resnet"
    type_or_path: str = "torchvision:wide_resnet101_2"
    kwargs: dict = field(default_factory=dict)
    shape: tuple = ("B", 3, 224, 224)
    num_warmup_batches: int = 5
    num_batches: int = 50
    batch_sizes: tuple = (16, 32)
    devices: tuple[str] = ("cpu",)
    runtime_options: dict[str, ty.Any] = field(default_factory=dict)
    custom_batchmetrics: dict[str, float] = field(default_factory=dict)


@dataclass
class NvbenjoConfig:
    """
    Root configuration for nvbenjo benchmarking.

    Parameters
    ----------
    measure_memory : bool
        Whether to measure GPU memory allocation during benchmarking.
    models: dict[str, TorchModelConfig | OnnxModelConfig]
        Dictionary mapping model names to their configurations.
        See :class:`TorchModelConfig` and :class:`OnnxModelConfig` for details.
    """

    measure_memory: bool = True
    models: dict[str, ty.Any] = field(default_factory=lambda: dict())


@dataclass
class BenchConfig:
    """
    Main benchmark configuration container.

    Parameters
    ----------
    nvbenjo : NvbenjoConfig
        Nvbenjo-specific configuration settings.
    output_dir : str or None
        Directory path where benchmark results will be saved.
        If None, uses Hydra's default output directory.
    """

    nvbenjo: NvbenjoConfig = field(default_factory=NvbenjoConfig)
    output_dir: ty.Optional[str] = None


@dataclass
class TorchRuntimeConfig:
    """PyTorch Runtime configuration:

    Parameters
    ----------
    compile : bool
        Whether to compile the model using torch.compile (PyTorch 2.0+).
    compile_kwargs : dict
        Additional keyword arguments for torch.compile.
    precision : PrecisionType
        Precision type for model inference (e.g., fp32, fp16, amp).
    enable_profiling : bool
        Whether to enable PyTorch profiler during inference.
    profiling_prefix : str or None
        Prefix for profiler output files. If None, a default path will be used.
    profiler_kwargs : dict
        Additional keyword arguments for torch.profiler.profile.
    """

    compile: bool = False
    compile_kwargs: dict = field(default_factory=dict)
    precision: PrecisionType = PrecisionType.FP32
    enable_profiling: bool = False
    profiling_prefix: ty.Optional[str] = None
    profiler_kwargs: dict = field(default_factory=dict)


@dataclass
class OnnxRuntimeConfig:
    """ONNX Runtime configuration:

    Parameters
    ----------
    execution_providers : tuple of str or None
        Tuple of execution providers to use (e.g., ('CPUExecutionProvider',
        'CUDAExecutionProvider')). If None, uses the default provider.
    graph_optimization_level : str
        Graph optimization level for ONNX Runtime. Options are 'ORT_ENABLE_ALL', 'ORT_ENABLE_LAYOUT',
        'ORT_ENABLE_BASIC', 'ORT_DISABLE_ALL'.
    intra_op_num_threads : int
        Number of threads used to parallelize the execution within nodes.
    inter_op_num_threads : int
        Number of threads used to parallelize the execution of the graph (between nodes)
    log_severity_level : int
        Logging severity level (0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL)
    enable_profiling : bool
        Whether to enable profiling in ONNX Runtime.
    profiling_prefix : str or None
        Prefix for profiling output files. If None, a default path will be used.
    provider_options : sequence of dict or None
        Additional options for each execution provider.
    """

    execution_providers: ty.Optional[ty.List[ProviderType]] = None
    graph_optimization_level: str = (
        "ORT_ENABLE_ALL"  # 99 ORT_ENABLE_ALL, 3 ORT_ENABLE_LAYOUT, 1 ORT_ENABLE_BASIC, 0 ORT_DISABLE_ALL
    )
    intra_op_num_threads: int = 1
    inter_op_num_threads: int = 0
    log_severity_level: int = 3  # Error
    enable_profiling: bool = False
    profiling_prefix: ty.Optional[str] = None
    provider_options: ty.Sequence[dict[ty.Any, ty.Any]] | None = None


@dataclass
class TorchModelConfig(BaseModelConfig):
    """PyTorch model configuration

    Parameters
    ----------
    name : str
        Name of the model.
    type_or_path : str
        Model type or path. Can be a local file path or a model identifier.
    kwargs : dict
        Additional keyword arguments to pass when instantiating the model.
    shape : tuple
        Input shape of the model. Use "B" to denote the batch size dimension.
        Examples::

            # Single input shape
            ("B", 3, 224, 224)

            # Multiple input shapes
            (("B", 3, 224, 224), ("B", 10))

            # Dictionary with metadata
            ({"name": "input1", "type": "float", "shape": ("B", 3, 224, 224), "min_max": (0, 1)},)

            # Multiple dictionary inputs
            (
                {"name": "input1", "type": "float", "shape": ("B", 3, 224, 224), "min_max": (0, 1)},
                {"name": "input2", "type": "int", "shape": (1, 3)},
                {"name": "input3", "type": "int", "shape": (), "value": 42},
            )
    num_warmup_batches : int
        Number of warm-up batches to run before measuring performance.
    num_batches : int
        Number of batches to run for performance measurement.
    batch_sizes : tuple
        Tuple of batch sizes to benchmark.
    devices : tuple of str
        Tuple of device names to benchmark on (e.g., 'cpu', 'cuda:0').
    runtime_options : dict[str, :class:`~nvbenjo.cfg.TorchRuntimeConfig`]
        Dictionary mapping runtime names to their specific runtime configurations.
    """

    model_kwargs: dict = field(default_factory=dict)
    runtime_options: dict[str, TorchRuntimeConfig] = field(default_factory=lambda: {"default": TorchRuntimeConfig()})

    def __post_init__(self):
        for i, (key, opt) in enumerate(self.runtime_options.items()):
            if isinstance(opt, DictConfig):
                self.runtime_options[key] = OmegaConf.structured(TorchRuntimeConfig(**OmegaConf.to_container(opt)))  # type: ignore


@dataclass
class OnnxModelConfig(BaseModelConfig):
    """ONNX model configuration

    Parameters
    ----------
    name : str
        Name of the model.
    type_or_path : str
        Model type or path. Can be a local file path or a model identifier.
    kwargs : dict
        Additional keyword arguments to pass when instantiating the model.
    shape : tuple
        Input shape of the model. Use "B" to denote the batch size dimension.

        Examples::

            # Single input shape
            ("B", 3, 224, 224)

            # Multiple input shapes
            (("B", 3, 224, 224), ("B", 10))

            # Dictionary with metadata
            ({"name": "input1", "type": "float", "shape": ("B", 3, 224, 224), "min_max": (0, 1)},)

            # Multiple dictionary inputs
            (
                {"name": "input1", "type": "float", "shape": ("B", 3, 224, 224), "min_max": (0, 1)},
                {"name": "input2", "type": "int", "shape": (1, 3)},
                {"name": "input3", "type": "int", "shape": (), "value": 42},
            )
    num_warmup_batches : int
        Number of warm-up batches to run before measuring performance.
    num_batches : int
        Number of batches to run for performance measurement.
    batch_sizes : tuple
        Tuple of batch sizes to benchmark.
    devices : tuple of str
        Tuple of device names to benchmark on (e.g., 'cpu', 'cuda:0').
    runtime_options : dict[str, :class:`~nvbenjo.cfg.OnnxRuntimeConfig`]
        Dictionary mapping runtime names to their specific runtime configurations.
    """

    runtime_options: dict[str, OnnxRuntimeConfig] = field(default_factory=lambda: {"default": OnnxRuntimeConfig()})

    def __post_init__(self):
        for i, (key, opt) in enumerate(self.runtime_options.items()):
            if isinstance(opt, DictConfig):
                self.runtime_options[key] = OmegaConf.structured(OnnxRuntimeConfig(**OmegaConf.to_container(opt)))  # type: ignore


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
                                cfg.nvbenjo.models[model_name]["runtime_options"][runtime_name]["precision"].upper()
                            ]

        models[model_name] = instantiate(model) if isinstance(model, DictConfig) else model
        if model_name in runtimes:
            models[model_name].runtime_options = runtimes[model_name]

    # For onnx profiling we add a valid profiling prefix in the output directory if needed
    for model_name, model in models.items():
        if isinstance(model, (OnnxModelConfig, TorchModelConfig)):
            for runtime_name, runtime in model.runtime_options.items():
                if runtime.enable_profiling:
                    if runtime.profiling_prefix is None:
                        runtime.profiling_prefix = os.path.join(
                            cfg.output_dir, model_name, f"{model_name}_{runtime_name}_profile"
                        )
                    else:
                        # make sure the relative path is inside the output dir
                        if not os.path.abspath(runtime.profiling_prefix) == runtime.profiling_prefix:
                            runtime.profiling_prefix = os.path.abspath(
                                os.path.join(cfg.output_dir, runtime.profiling_prefix)
                            )
                    os.makedirs(os.path.dirname(runtime.profiling_prefix), exist_ok=True)

    return models
