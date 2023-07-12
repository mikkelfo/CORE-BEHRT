import os
import torch
import hydra
from transformers import BertConfig
from torch.optim import AdamW
from src.data.dataset import MLMDataset
from src.trainer.trainer import EHRTrainer
from src.model.model import BertEHRModel


@hydra.main(config_path="../../configs/train", config_name="pretrain")
def main_train(cfg):
    # MLM specific
    train_encoded = torch.load(
        os.path.join(cfg.paths.data_dir, f"train_{cfg.paths.encoded_suffix}.pt")
    )
    val_encoded = torch.load(
        os.path.join(cfg.paths.data_dir, f"val_{cfg.paths.encoded_suffix}.pt")
    )
    vocabulary = torch.load(os.path.join(cfg.paths.data_dir, cfg.paths.vocabulary))
    train_dataset = MLMDataset(
        train_encoded,
        vocabulary=vocabulary,
        ignore_special_tokens=cfg.ignore_special_tokens,
    )
    val_dataset = MLMDataset(
        val_encoded,
        vocabulary=vocabulary,
        ignore_special_tokens=cfg.ignore_special_tokens,
    )

    # Model configuration
    if cfg.model.vocab_size is None:  # Calculates vocab_size if not given
        cfg.model.vocab_size = len(vocabulary)
    model = BertEHRModel(BertConfig(**cfg.model))

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
        cfg=cfg,
    )
    trainer.train()


if __name__ == "__main__":
    main_train()
