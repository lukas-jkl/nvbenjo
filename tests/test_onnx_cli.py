import omegaconf
import pytest
import tempfile

import torch
from hydra import compose, initialize

from tests.test_cli import DummyModelMultiInput, _check_run_files, run_config

try:
    from nvbenjo import onnx_utils  # noqa: F401
except ImportError as e:
    if "onnxruntime" in str(e):
        pytest.skip("onnxruntime is not installed, skipping ONNX utils tests.", allow_module_level=True)
    else:
        raise


def test_onnx():
    model = DummyModelMultiInput()

    with initialize(version_base=None, config_path="conf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmpfile:
            torch.onnx.export(
                model,
                args=(torch.randn(2, 10), torch.randn(2, 20)),
                dynamic_axes={
                    "x": {0: "batch_size"},  # First dimension of input 'x' is dynamic
                    "y": {0: "batch_size"},  # First dimension of input 'y' is dynamic
                    "output": {0: "batch_size"},  # First dimension of output is dynamic
                },
                f=tmpfile.name,
                input_names=["x", "y"],
                output_names=["output"],
                opset_version=17,
            )
            min, max = 0, 5

            with tempfile.TemporaryDirectory() as tmpoutdir:
                config_override = {
                    "nvbenjo": {
                        "models": [
                            {
                                "name": "dummytorchmodel",
                                "type_or_path": tmpfile.name,
                                "num_batches": 2,
                                "batch_sizes": [1, 2],
                                "devices": ["cpu"],
                                "shape": [
                                    {"name": "x", "shape": ["B", 10], "min_max": [min, max]},
                                    {"name": "y", "shape": ["B", 20], "min_max": [min, max]},
                                ],
                                "precisions": ["FP32"],
                            }
                        ]
                    }
                }
                cfg = compose(
                    config_name="default",
                    overrides=[
                        f"output_dir={tmpoutdir}",
                    ],
                )
                cfg = omegaconf.OmegaConf.merge(cfg, omegaconf.OmegaConf.create(config_override))
                run_config(cfg)
                _check_run_files(cfg)


def test_onnx_invalid_amp():
    model = DummyModelMultiInput()

    with initialize(version_base=None, config_path="conf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmpfile:
            torch.onnx.export(
                model,
                args=(torch.randn(2, 10), torch.randn(2, 20)),
                dynamic_axes={
                    "x": {0: "batch_size"},  # First dimension of input 'x' is dynamic
                    "y": {0: "batch_size"},  # First dimension of input 'y' is dynamic
                    "output": {0: "batch_size"},  # First dimension of output is dynamic
                },
                f=tmpfile.name,
                input_names=["x", "y"],
                output_names=["output"],
                opset_version=18,
            )
            min, max = 0, 5

            with tempfile.TemporaryDirectory() as tmpoutdir:
                config_override = {
                    "nvbenjo": {
                        "models": [
                            {
                                "name": "dummytorchmodel",
                                "type_or_path": tmpfile.name,
                                "num_batches": 2,
                                "batch_sizes": [1, 2],
                                "devices": ["cpu"],
                                "shape": [
                                    {"name": "x", "shape": ["B", 10], "min_max": [min, max]},
                                    {"name": "y", "shape": ["B", 20], "min_max": [min, max]},
                                ],
                                "precisions": ["AMP_FP16"],
                            }
                        ]
                    }
                }
                cfg = compose(
                    config_name="default",
                    overrides=[
                        f"output_dir={tmpoutdir}",
                    ],
                )
                cfg = omegaconf.OmegaConf.merge(cfg, omegaconf.OmegaConf.create(config_override))
                with pytest.raises(ValueError, match="ONNX models do not support AMP precision"):
                    run_config(cfg)
