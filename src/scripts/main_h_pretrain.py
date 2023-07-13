import os
import torch
import hydra
from torch.optim import AdamW
from transformers import BertConfig
from src.data.dataset import HierarchicalDataset
from src.trainer.trainer import EHRTrainer
from src.model.model import HierarchicalBertForPretraining


@hydra.main(config_path="../../configs/train", config_name="pretrain_h")
def main_train(cfg):
    # MLM specific
    train_encoded = torch.load(
        os.path.join(cfg.paths.data_dir, f"train_{cfg.paths.encoded_suffix}.pt")
    )
    val_encoded = torch.load(
        os.path.join(cfg.paths.data_dir, f"val_{cfg.paths.encoded_suffix}.pt")
    )
    vocabulary = torch.load(os.path.join(cfg.paths.data_dir, cfg.paths.vocabulary))
    tree = torch.load(os.path.join(cfg.paths.extra_dir, "tree.pt"))

    train_dataset = HierarchicalDataset(
        train_encoded,
        tree=tree,
        vocabulary=vocabulary,
        ignore_special_tokens=cfg.ignore_special_tokens,
    )
    val_dataset = HierarchicalDataset(
        val_encoded,
        tree=tree,
        vocabulary=vocabulary,
        ignore_special_tokens=cfg.ignore_special_tokens,
    )

    # Model configuration
    if cfg.model.vocab_size is None:  # Calculates vocab_size if not given
        cfg.model.vocab_size = len(vocabulary)
    if cfg.model.leaf_size is None:  # Calculate leaf_size if not given
        cfg.model.leaf_size = tree.num_children_leaves()
    if cfg.model.type_vocab_size is None:  # Max number of segments if SEP tokens
        cfg.model.type_vocab_size = cfg.model.max_position_embeddings // 2
    model = HierarchicalBertForPretraining(BertConfig(**cfg.model))

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
