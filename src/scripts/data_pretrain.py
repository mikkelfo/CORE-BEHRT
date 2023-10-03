import os
import torch
import hydra
import src.common.setup as setup
from src.data.split import Splitter
from src.data.tokenizer import EHRTokenizer


@hydra.main(config_path="../../configs/data", config_name="pretrain")
def main_data(cfg):
    # Save config
    setup.save_config(cfg)

    # Get features (load, infer, create, handle, exclude)
    features = setup.get_features(cfg)

    # Split features
    ## Pretrain split
    pretrain_splits = Splitter(
        cfg,
        split_name="pretrain_splits.pt",
    )(features, mode=cfg.split.mode)

    # Tokenize
    tokenizer = EHRTokenizer(cfg.tokenizer)
    _ = tokenizer(pretrain_splits["train"])  # Re-do for ease of use
    tokenizer.freeze_vocabulary(
        vocab_name=os.path.join(cfg.paths.data_dir, cfg.paths.vocabulary)
    )

    # Save features
    setup.save_splits(cfg, pretrain_splits, tokenizer)


if __name__ == "__main__":
    main_data()
