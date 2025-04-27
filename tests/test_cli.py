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
            run(cfg)
            _check_files(tmpdir, EXPECTED_OUTPUT_FILES)


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
                run(cfg)
                _check_files(tmpoutdir, EXPECTED_OUTPUT_FILES)
