import os
import json
import torch
from omegaconf import OmegaConf


def load(dir: str, path: str):
    return torch.load(os.path.join(dir, path))


def checkpoint(cfg):
    checkpoint = load("runs", cfg.paths.checkpoint)
    run_name = cfg.paths.checkpoint.split("/")[0] + "_cont"
    config_path = os.path.join(
        "runs/", cfg.paths.checkpoint.split("/")[0], "config.json"
    )
    cfg = OmegaConf.create(json.load(open(config_path))["cfg"])
    cfg.trainer_args.run_name = run_name

    return cfg, checkpoint


def train_val(cfg, suffix: str, prefix = None):
    prefix = f"{prefix}_" if prefix is not None else ""
    for set in ["train", "val"]:
        yield load(
            cfg.paths.data_dir,
            f"{prefix}{set}_{suffix}.pt",
        )


def encoded(cfg):
    train_encoded, val_encoded = train_val(
        cfg, suffix=cfg.paths.encoded_suffix, prefix=cfg.paths.get("encoded_prefix")
    )
    return train_encoded, val_encoded


def outcomes(cfg):
    train_outcomes, val_outcomes = train_val(cfg, cfg.paths.outcomes_suffix)
    return train_outcomes, val_outcomes


def features(cfg):
    return load(cfg.paths.data_dir, "features.pt")


def vocabulary(cfg):
    return load(cfg.paths.data_dir, cfg.paths.vocabulary)


def tree(cfg):
    return load(cfg.paths.extra_dir, "tree.pt")


def data_config(cfg):
    return json.load(open(os.path.join(cfg.paths.data_dir, "data_config.json")))
