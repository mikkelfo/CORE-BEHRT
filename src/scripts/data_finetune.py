import os
import torch
import hydra
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
    for set, data in finetune_splits.items():
        encoded = tokenizer(data)
        torch.save(
            encoded,
            os.path.join(cfg.paths.data_dir, f"{set}_{cfg.paths.encoded_suffix}.pt"),
        )


if __name__ == "__main__":
    main_data()
