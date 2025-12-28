import os
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


@pytest.mark.parametrize("do_profile", [True, False])
def test_onnx(do_profile: bool):
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
                profile_prefix_dir = "myprefix"
                profile_fileprefix = "testfile"
                profile_prefix = f"{profile_prefix_dir}/{profile_fileprefix}" if do_profile else None
                config_override = {
                    "nvbenjo": {
                        "models": {
                            "dummyonnxmodel": {
                                "type_or_path": tmpfile.name,
                                "num_batches": 2,
                                "batch_sizes": [1, 2],
                                "devices": ["cpu"],
                                "shape": [
                                    {"name": "x", "shape": ["B", 10], "min_max": [min, max]},
                                    {"name": "y", "shape": ["B", 20], "min_max": [min, max]},
                                ],
                                "runtime_options": {
                                    "default": {
                                        "graph_optimization_level": "ORT_ENABLE_ALL",
                                        "intra_op_num_threads": 1,
                                        "inter_op_num_threads": 2,
                                        "enable_profiling": do_profile,
                                        "profiling_prefix": profile_prefix,
                                    }
                                },
                            }
                        }
                    }
                }
                cfg = compose(
                    config_name="default",
                    overrides=[
                        f"output_dir={tmpoutdir}",
                    ],
                )
                # temporary disable struct mode to allow merging additional model
                omegaconf.OmegaConf.set_struct(cfg, False)
                cfg = omegaconf.OmegaConf.merge(cfg, omegaconf.OmegaConf.create(config_override))
                omegaconf.OmegaConf.set_struct(cfg, True)
                run_config(cfg)
                _check_run_files(cfg)
                assert os.path.isdir(os.path.join(tmpoutdir, profile_prefix_dir)) == do_profile
                if do_profile:
                    listprofile = os.listdir(os.path.join(tmpoutdir, profile_prefix_dir))
                    assert len(listprofile) > 0
                    for f in listprofile:
                        assert f.startswith(profile_fileprefix)
