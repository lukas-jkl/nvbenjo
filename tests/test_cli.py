import omegaconf
import pytest
import csv
import os
import tempfile

import yaml
import torch
from hydra import compose, initialize

from nvbenjo.cli import run

EXPECTED_OUTPUT_FILES = [
    "config.yaml",
    "out.csv",
]


def run_config(cfg):
    if isinstance(cfg, omegaconf.DictConfig):
        run(cfg)
    else:
        raise ValueError("Config is not a DictConfig instance")


def _check_files(directory, files):
    for file in files:
        assert os.path.exists(os.path.join(directory, file)), f"File {file} not found in {directory}"
        if file.endswith(".yaml"):
            with open(os.path.join(directory, file), "r") as f:
                try:
                    yaml.safe_load(f)
                except yaml.YAMLError as e:
                    raise AssertionError(f"YAML file {file} is not valid: {e}")
        elif file.endswith(".csv"):
            with open(os.path.join(directory, file), "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                assert len(header) > 0, f"CSV file {file} is empty or has no header"
                for row in reader:
                    assert len(row) == len(header), f"Row in CSV file {file} does not match header length"


def test_basic():
    with initialize(version_base=None, config_path="conf"):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = compose(config_name="default", overrides=[f"output_dir={tmpdir}"])
            run_config(cfg)
            _check_files(tmpdir, EXPECTED_OUTPUT_FILES)


def test_small_single():
    with initialize(version_base=None, config_path="conf"):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = compose(config_name="small_single", overrides=[f"output_dir={tmpdir}"])
            run_config(cfg)
            _check_files(tmpdir, EXPECTED_OUTPUT_FILES)


def test_duplicate_model_names():
    with initialize(version_base=None, config_path="conf"):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = compose(config_name="duplicate_model_name", overrides=[f"output_dir={tmpdir}"])
            with pytest.raises(ValueError):
                run_config(cfg)


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
                    overrides=[f"output_dir={tmpoutdir}", f'nvbenjo.models.0.type_or_path="{tmpfile.name}"'],
                )
                run_config(cfg)
                _check_files(tmpoutdir, EXPECTED_OUTPUT_FILES)


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
                        f'nvbenjo.models.0.type_or_path="{tmpfile.name}"',
                    ],
                )
                run_config(cfg)
                _check_files(tmpoutdir, EXPECTED_OUTPUT_FILES)


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


def test_torch_load_complex_multiinput():
    min = 12
    max = 34
    model = ComplexDummyModelMultiInput(min=min, max=max)

    with initialize(version_base=None, config_path="conf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmpfile:
            torch.save(model, tmpfile)
            with tempfile.TemporaryDirectory() as tmpoutdir:
                config_override = {
                    "nvbenjo": {
                        "models": [
                            {
                                "name": "dummytorchmodel",
                                "type_or_path": tmpfile.name,
                                "num_batches": 2,
                                "batch_sizes": [1, 2],
                                "device_indices": [0],
                                "precisions": ["FP32", "FP16", "AMP_FP16"],
                                "shape": [
                                    {"name": "x", "shape": ["B", 10], "min_max": [min, max]},
                                    {"name": "y", "shape": ["B", 20], "min_max": [min, max]},
                                ],
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
                _check_files(tmpoutdir, EXPECTED_OUTPUT_FILES)


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
                        "models": [
                            {
                                "name": "dummytorchmodel",
                                "type_or_path": tmpfile.name,
                                "num_batches": 2,
                                "batch_sizes": [1, 2],
                                "device_indices": [0],
                                "precisions": ["FP32"],
                                "shape": [
                                    {"name": "x", "shape": ["B", 10], "type": "float", "min_max": [max, max * 2]},
                                    {"name": "y", "shape": ["B", 20], "type": "float", "min_max": [max, max * 2]},
                                ],
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
                if isinstance(cfg, omegaconf.DictConfig):
                    with pytest.raises(
                        ValueError, match=f"Input x contains values outside the range \\[{min}, {max}\\]"
                    ):
                        run_config(cfg)
                else:
                    raise ValueError("Config is not a DictConfig instance")
