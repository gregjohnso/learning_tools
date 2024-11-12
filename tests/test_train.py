import os

from omegaconf import OmegaConf

from learning_tools.train import train

my_dir = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(my_dir, "../demo/configs/scop_short.yaml")


def test_train():
    # load the config
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg = OmegaConf.to_container(cfg)

    # This is only used from the CLI API, so we remove it here
    cfg.pop("name")

    train(**cfg)
