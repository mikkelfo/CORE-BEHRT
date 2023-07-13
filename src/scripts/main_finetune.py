import os
import json
import torch
import pandas as pd
import hydra
from torch.optim import AdamW
from transformers import BertConfig
from torch.utils.data import WeightedRandomSampler
from src.data.dataset import CensorDataset
from src.trainer.trainer import EHRTrainer
from src.model.model import BertForFineTuning


@hydra.main(config_path="../../configs/train", config_name="finetune")
def main_finetune(cfg):
    # Finetune specific
    train_encoded = torch.load(
        os.path.join(cfg.paths.data_dir, f"train_{cfg.paths.encoded_suffix}.pt")
    )
    val_encoded = torch.load(
        os.path.join(cfg.paths.data_dir, f"val_{cfg.paths.encoded_suffix}.pt")
    )
    train_outcomes = torch.load(
        os.path.join(cfg.paths.data_dir, f"train_{cfg.paths.outcomes_suffix}.pt")
    )
    val_outcomes = torch.load(
        os.path.join(cfg.paths.data_dir, f"val_{cfg.paths.outcomes_suffix}.pt")
    )
    n_hours, outcome_type, censor_type = (
        cfg.outcome.n_hours,
        cfg.outcome.type,
        cfg.outcome.censor_type,
    )
    train_dataset = CensorDataset(
        train_encoded,
        n_hours=n_hours,
        outcomes=train_outcomes[outcome_type],
        censor_outcomes=train_outcomes[censor_type],
    )
    val_dataset = CensorDataset(
        val_encoded,
        n_hours=n_hours,
        outcomes=val_outcomes[outcome_type],
        censor_outcomes=val_outcomes[censor_type],
    )

    sampler, pos_weight = None, None
    if cfg.trainer_args["sampler"]:
        labels = pd.Series(train_outcomes[outcome_type]).notna().astype(int)
        label_weight = 1 / labels.value_counts()
        weights = labels.map(label_weight).values
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(train_dataset), replacement=True
        )
    elif cfg.trainer_args["pos_weight"]:
        pos_weight = sum(pd.isna(train_outcomes[outcome_type])) / sum(
            pd.notna(train_outcomes[outcome_type])
        )

    print(
        f'Setting up finetune task on [{outcome_type}] with [{n_hours}] hours censoring at [{censor_type}] using pos_weight [{pos_weight}] and sampler [{cfg.trainer_args["sampler"]}]'
    )

    # Initializing based on pretraining
    pretraining_dir = os.path.join("runs", cfg.paths.pretraining.dir)
    config_path = os.path.join(pretraining_dir, "config.json")
    model_path = os.path.join(pretraining_dir, cfg.paths.pretraining.model_file)
    # Loading the files
    model_config = json.load(open(config_path))["model_config"]
    state_dict = torch.load(model_path)["model_state_dict"]
    # Initialize model
    model = BertForFineTuning(BertConfig(**model_config, pos_weight=pos_weight))
    model.load_state_dict(state_dict, strict=False)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        eps=cfg.optimizer.epsilon,
    )

    trainer = EHRTrainer(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=cfg.trainer_args,
        metrics=cfg.metrics,
        sampler=sampler,
        cfg=cfg,
    )
    trainer.train()


if __name__ == "__main__":
    main_finetune()
