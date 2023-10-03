import os
import torch
import hydra
import src.common.setup as setup
from src.data.tokenizer import EHRTokenizer
from src.data.split import Splitter


@hydra.main(config_path="../../configs/data", config_name="finetune")
def main_data(cfg):
    # Load features
    features = torch.load(os.path.join(cfg.paths.data_dir, "features.pt"))

    # Finetune split
    finetune_splits = Splitter(cfg, split_name="finetune_splits.pt")(
        features, mode="load", file="covid_splits.pt"
    )

    # Tokenize
    tokenizer = EHRTokenizer(
        cfg.tokenizer, vocabulary=os.path.join(cfg.paths.data_dir, cfg.paths.vocabulary)
    )

    # Save features
    setup.save_splits(cfg, finetune_splits, tokenizer)


if __name__ == "__main__":
    main_data()
