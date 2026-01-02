import csv
import os
import shutil
import subprocess
import pandas as pd
import tempfile
import warnings
from copy import copy
from os.path import isfile, join

import omegaconf
import pytest
import torch
import yaml
from hydra import compose, initialize

from nvbenjo.cli import run

DATA_FILE = "out.csv"
EXPECTED_OUTPUT_FILES = [
    "config.yaml",
    DATA_FILE,
]


def run_config(cfg):
    if isinstance(cfg, omegaconf.DictConfig):
        run(cfg)
    else:
        raise ValueError("Config is not a DictConfig instance")


def _check_files(directory, files):
    for file in files:
        assert isfile(join(directory, file)), f"File {file} not found in {directory}"
        if file.endswith(".yaml"):
            with open(join(directory, file), "r") as f:
                try:
                    yaml.safe_load(f)
                except yaml.YAMLError as e:
                    raise AssertionError(f"YAML file {file} is not valid: {e}")
        elif file.endswith(".csv"):
            with open(join(directory, file), "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                assert len(header) > 0, f"CSV file {file} is empty or has no header"
                for row in reader:
                    assert len(row) == len(header), f"Row in CSV file {file} does not match header length"


def _check_run_files(cfg: omegaconf.DictConfig):
    expected_files = copy(EXPECTED_OUTPUT_FILES)
    for model_name in cfg.nvbenjo.models.keys():
        expected_files.append(join(model_name, "time_inference.png"))
        expected_files.append(join(model_name, "time_total_batch_normalized.png"))
        expected_files.append(join(model_name, "time_device_to_cpu.png"))
        expected_files.append(join(model_name, "time_cpu_to_device.png"))
        expected_files.append(join(model_name, "memory_bytes.png"))
    if len(cfg.nvbenjo.models) > 1:
        expected_files.append(join("summary", "time_inference.png"))
        expected_files.append(join("summary", "time_total_batch_normalized.png"))
        expected_files.append(join("summary", "time_device_to_cpu.png"))
        expected_files.append(join("summary", "time_cpu_to_device.png"))
        expected_files.append(join("summary", "memory_bytes.png"))
    _check_files(cfg.output_dir, expected_files)
    for model_name in cfg.nvbenjo.models.keys():
        for runtime_name in cfg.nvbenjo.models[model_name].get("runtime_options", {}).keys():
            if cfg.nvbenjo.models[model_name]["runtime_options"][runtime_name].get("enable_profiling", False):
                profile_prefix = cfg.nvbenjo.models[model_name]["runtime_options"][runtime_name].get(
                    "profiling_prefix", None
                )
                if profile_prefix is None:
                    profile_prefix = join(
                        cfg.output_dir,
                        model_name,
                        f"{model_name}_{runtime_name}_profile",
                    )
                else:
                    profile_prefix = os.path.abspath(os.path.join(cfg.output_dir, profile_prefix))
                profile_prefix_dir = os.path.dirname(profile_prefix)
                assert os.path.isdir(profile_prefix_dir), f"Profiling directory {profile_prefix_dir} not found"
                profile_files = os.listdir(profile_prefix_dir)
                assert len(profile_files) > 0, f"No profiling files found in {profile_prefix}"
    results = pd.read_csv(join(cfg.output_dir, DATA_FILE))
    assert not results.empty, "Results CSV is empty"
    for model_name, model_cfg in cfg.nvbenjo.models.items():
        assert model_name in results.model.to_numpy(), "Model column not found in results"
        custom_metric_keys = model_cfg.get("custom_batchmetrics", {})
        for key in custom_metric_keys:
            assert key in results.columns, f"Custom batch metric {key} not found in results"


def test_default():
    with initialize(version_base=None, config_path="conf"):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = compose(config_name="default", overrides=[f"output_dir={tmpdir}"])
            run_config(cfg)


def test_small_single():
    with initialize(version_base=None, config_path="conf"):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = compose(config_name="small_single", overrides=[f"output_dir={tmpdir}"])
            run_config(cfg)
            _check_run_files(cfg)


def test_min_max_input_type():
    with initialize(version_base=None, config_path="conf"):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = compose(config_name="input_min_max", overrides=[f"output_dir={tmpdir}"])
            with pytest.raises(ValueError):
                run_config(cfg)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


def test_torch_load():
    model = DummyModel()

    with initialize(version_base=None, config_path="conf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmpfile:
            torch.save(model, tmpfile)
            with tempfile.TemporaryDirectory() as tmpoutdir:
                cfg = compose(
                    config_name="torch_load",
                    overrides=[
                        f"output_dir={tmpoutdir}",
                        f'nvbenjo.models.dummytorchmodel.type_or_path="{tmpfile.name}"',
                    ],
                )
                run_config(cfg)
                _check_run_files(cfg)


class DummyModelMultiInput(torch.nn.Module):
    def __init__(self):
        super(DummyModelMultiInput, self).__init__()
        self.fc1 = torch.nn.Linear(10, 1)
        self.fc2 = torch.nn.Linear(20, 1)

    def forward(self, x, y):
        return self.fc1(x) * self.fc2(y)


def test_torch_load_multiinput():
    model = DummyModelMultiInput()

    with initialize(version_base=None, config_path="conf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmpfile:
            torch.save(model, tmpfile)
            with tempfile.TemporaryDirectory() as tmpoutdir:
                cfg = compose(
                    config_name="torch_load_multiinput",
                    overrides=[
                        f"output_dir={tmpoutdir}",
                        f'nvbenjo.models.dummytorchmodel.type_or_path="{tmpfile.name}"',
                    ],
                )
                run_config(cfg)
                _check_run_files(cfg)


class ComplexDummyModelMultiInput(torch.nn.Module):
    def __init__(self, min, max):
        super(ComplexDummyModelMultiInput, self).__init__()
        self.fc1 = torch.nn.Linear(10, 1)
        self.fc2 = torch.nn.Linear(20, 1)
        self.min = min
        self.max = max

    def forward(self, x, y):
        if torch.any(x < self.min) or torch.any(x > self.max):
            raise ValueError(f"Input x contains values outside the range [{self.min}, {self.max}]")
        return self.fc1(x) * self.fc2(y)


@pytest.mark.parametrize("export_type", ["torchexport", "torchsave", "torchscript"])
def test_torch_load_complex_multiinput(export_type):
    if export_type == "torchexport" and torch.__version__ < "2.1":
        pytest.skip("torch.export is only available in PyTorch 2.1 and later")

    min = 12
    max = 34
    model = ComplexDummyModelMultiInput(min=min, max=max)

    with initialize(version_base=None, config_path="conf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmpfile:
            if export_type == "torchsave":
                torch.save(model, tmpfile)
            elif export_type == "torchexport":
                # for tochexport we need a model with simplified control flow
                model = DummyModelMultiInput()
                batch_size_dim = torch.export.Dim("B", min=1, max=1024)
                program = torch.export.export(
                    model,
                    args=(torch.randn(2, 10), torch.randn(2, 20)),
                    dynamic_shapes={"x": {0: batch_size_dim}, "y": {0: batch_size_dim}},
                )
                torch.export.save(program, tmpfile.name)
                torch.export.load(tmpfile.name)  # verify it can be loaded
            elif export_type == "torchscript":
                m = torch.jit.script(model)
                torch.jit.save(m, tmpfile)
            else:
                raise ValueError(f"Unknown export_type {export_type}")

            with tempfile.TemporaryDirectory() as tmpoutdir:
                config_override = {
                    "nvbenjo": {
                        "models": {
                            "dummytorchmodel": {
                                "type_or_path": tmpfile.name,
                                "num_batches": 2,
                                "batch_sizes": [1, 2],
                                "devices": ["cpu"],
                                "runtime_options": {
                                    "FP32": {"precision": "FP32", "compile": False},
                                    "FP16": {"precision": "FP16", "compile": False},
                                    "AMP_FP16": {"precision": "AMP_FP16", "compile": False},
                                },
                                "shape": [
                                    {"name": "x", "shape": ["B", 10], "min_max": [min, max]},
                                    {"name": "y", "shape": ["B", 20], "min_max": [min, max]},
                                ],
                            }
                        }
                    }
                }
                if torch.__version__ < "2.2":
                    config_override["nvbenjo"]["models"]["dummytorchmodel"]["runtime_options"].pop("AMP_FP16", None)
                    config_override["nvbenjo"]["models"]["dummytorchmodel"]["runtime_options"].pop("FP16", None)

                if export_type == "torchexport":
                    # torchexport does not work with named inputs
                    config_override["nvbenjo"]["models"]["dummytorchmodel"]["shape"] = [
                        ["B", 10],
                        ["B", 20],
                    ]
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
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    run_config(cfg)
                _check_run_files(cfg)


def test_torch_load_complex_invalid_multiinput():
    min = 0
    max = 12
    model = ComplexDummyModelMultiInput(min=min, max=max)

    with initialize(version_base=None, config_path="conf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmpfile:
            torch.save(model, tmpfile)
            with tempfile.TemporaryDirectory() as tmpoutdir:
                config_override = {
                    "nvbenjo": {
                        "models": {
                            "dummytorchmodel": {
                                "type_or_path": tmpfile.name,
                                "num_batches": 2,
                                "batch_sizes": [1, 2],
                                "devices": ["cpu"],
                                "runtime_options": {
                                    "FP32": {
                                        "precision": "FP32",
                                        "compile": False,
                                        "enable_profiling": True,
                                        "profiler_kwargs": {"record_shapes": True, "with_stack": True},
                                    },
                                },
                                "shape": [
                                    {"name": "x", "shape": ["B", 10], "type": "float", "min_max": [max, max * 2]},
                                    {"name": "y", "shape": ["B", 20], "type": "float", "min_max": [max, max * 2]},
                                ],
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
                if isinstance(cfg, omegaconf.DictConfig):
                    with pytest.raises(
                        ValueError, match=f"Input x contains values outside the range \\[{min}, {max}\\]"
                    ):
                        run_config(cfg)
                else:
                    raise ValueError("Config is not a DictConfig instance")


def test_cli_cn_path_arg():
    with initialize(version_base=None, config_path="conf"):
        with tempfile.TemporaryDirectory() as cfg_tmpdir:
            cfg_file = os.path.join(cfg_tmpdir, "smallasdf.yaml")
            shutil.copy2(os.path.join("tests", "conf", "small_single.yaml"), cfg_file)
            with tempfile.TemporaryDirectory() as tmpdir:
                subprocess.run(
                    [
                        "python",
                        "-m",
                        "nvbenjo.cli",
                        "-cn",
                        cfg_file,
                        f"output_dir={tmpdir}",
                    ],
                    check=True,
                )

                cfg = compose(
                    config_name="small_single",
                    overrides=[
                        f"output_dir={tmpdir}",
                    ],
                )
                _check_run_files(cfg)
