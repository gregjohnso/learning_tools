"""
Train a model.

Uses Hydra to load the config, and OmegaConf to merge the config with the default config.

Usage:
```
python train.py config=path/to/config.yaml
```

Commands may be overridden by command line arguments:
```
python train.py config=path/to/config.yaml batch_size=16
```
"""

import datetime
import os
from typing import Dict

import lightning
import torch
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from nanogpt_uniprot.loading import load_object

torch.set_float32_matmul_precision("medium")


# TODO: dataloader_test is not used. Remove?
# TODO: Add support for DDP/multi-GPU training. Some good examples at https://github.com/karpathy/nanoGPT/blob/master/train.py


def train(
    save_dir: str,
    model: Dict,
    dataloader_train: Dict,
    dataloader_val: Dict,
    dataloader_test: Dict,
    trainer: Dict,
    compile: bool = False,
    ddp: bool = False,
    ddp_local_rank: int = 0,
):
    os.makedirs(save_dir, exist_ok=True)

    if ddp:
        # TODO
        ...

    model: lightning.LightningModule = load_object(**model)

    if compile:
        # TODO: This doesnt work with lightning.
        model = torch.compile(model)

    if ddp:
        # wrap model in DDP
        model = DDP(model, device_ids=[ddp_local_rank])

    dataloader_train: torch.utils.data.DataLoader = load_object(**dataloader_train)
    dataloader_val: torch.utils.data.DataLoader = load_object(**dataloader_val)

    trainer: lightning.Trainer = load_object(**trainer)

    trainer.fit(
        model=model,
        train_dataloaders=dataloader_train,
        val_dataloaders=dataloader_val,
    )


def _replace_save_dir(cfg: Dict, save_dir: str):
    """
    Recursively replace all ${save_dir} in the config with the actual save_dir.
    """
    for key, value in cfg.items():
        if isinstance(value, dict):
            cfg[key] = _replace_save_dir(value, save_dir)
        elif isinstance(value, list):
            cfg[key] = [_replace_save_dir(v, save_dir) for v in value]
        elif isinstance(value, str) and "${save_dir}" in value:
            cfg[key] = value.replace("${save_dir}", save_dir)
    return cfg


def main():
    """
    CLI entry point for training a model.

    The config file is merged with the command line arguments, with command line arguments taking precedence.

    Usage:
    ```
    python train.py config=path/to/config.yaml
    ```

    Commands may be overridden by command line arguments:
    ```
    python train.py config=path/to/config.yaml batch_size=16
    ```

    """
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    cfg = OmegaConf.merge(file_cfg, cli_args)
    cfg = OmegaConf.to_container(cfg)

    # pop off save_parent from cfg
    save_parent = cfg.pop("save_parent")
    name = cfg.pop("name")
    # save dir is <name>_yymmdd_hhmmss
    save_dir = os.path.join(
        save_parent, f"{name}_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}"
    )

    # recurse through cfg and replace all ${save_dir} with the actual save_dir
    cfg = _replace_save_dir(cfg, save_dir)

    # Save a copy of the config to the save dir using OmegaConf
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), os.path.join(save_dir, "config.yaml"))

    train(**cfg, save_dir=save_dir)


if __name__ == "__main__":
    main()
