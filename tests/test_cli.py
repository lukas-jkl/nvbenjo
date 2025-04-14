from omegaconf import OmegaConf
from nvbenjo.cli import run
from hydra import compose, initialize
import tempfile

def test_basic():
    with initialize(version_base=None, config_path="conf"):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = compose(config_name="default", overrides=[f"output_dir={tmpdir}"])
            run(cfg)
