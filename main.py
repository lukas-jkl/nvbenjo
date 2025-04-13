import hydra


from nvbenjo.cfg import BenchConfig
from nvbenjo.cli import run


@hydra.main(version_base=None, config_path="./nvbenjo/conf", config_name="default")
def main(cfg: BenchConfig):
    run(cfg)


if __name__ == "__main__":
    main()
