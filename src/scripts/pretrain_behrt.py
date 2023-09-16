import hydra
import src.common.loading as loading
import src.common.create as create
from src.trainer.trainer import EHRTrainer
from src.model.model import BehrtModel
from src.data.adapter import DataAdapter


@hydra.main(config_path="../../configs/train", config_name="pretrain")
def main_train(cfg):
    # Load in checkpoint if provided (cont. training)
    if cfg.paths.checkpoint is not None:
        cfg, checkpoint = loading.checkpoint(cfg)

    # Create datasets
    train_dataset, val_dataset = create.mlm_dataset(cfg)
    # Adapt to BEHRT data format
    for dataset in [train_dataset, val_dataset]:
        dataset.features = dataset._to_tensors(
            DataAdapter().adapt_to_behrt(dataset.features), dtypes={}
        )

    # Initialize model and optimizer
    model, optimizer = create.model(
        BehrtModel,
        cfg,
        vocabulary=train_dataset.vocabulary,
        seg_vocab_size=2,
        age_vocab_size=120,
    )

    # Override state_dicts if checkpoint
    if cfg.paths.checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Initialize Trainer
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
