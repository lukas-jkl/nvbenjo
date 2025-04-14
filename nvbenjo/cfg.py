import typing as ty
from dataclasses import dataclass, field
from nvbenjo.utils import PrecisionType


@dataclass
class ModelConfig:
    name: str = "resnet"
    type_or_path: str = "wide_resnet101_2"
    kwargs: dict = field(default_factory=dict)
    shape: tuple = ("B", 3, 224, 224)
    num_warmup_batches: int = 5
    num_batches: int = 50
    batch_sizes: ty.Tuple[int] = (16, 32)
    device_indices: ty.Tuple[int] = (0,)
    precisions: ty.Tuple[PrecisionType] = (PrecisionType.FP32, PrecisionType.FP16, PrecisionType.AMP)


@dataclass
class NvbenjoConfig:
    enable: bool = True
    measure_memory: bool = True
    torch_profile: bool = False
    output_dir_name: ty.Optional[str] = None
    models: ty.List[ModelConfig] = field(default_factory=lambda: [ModelConfig()])


@dataclass
class NvbandwithConfig:
    enable: bool = False


@dataclass
class BenchConfig:
    nvbenjo: NvbenjoConfig = field(default_factory=NvbenjoConfig)
    nvbandiwth: NvbandwithConfig = field(default_factory=NvbandwithConfig)
    output_dir: ty.Optional[str] = None
