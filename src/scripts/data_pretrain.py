import os
import torch
import hydra
import json
from omegaconf import OmegaConf
from src.data.concept_loader import ConceptLoader
from src.data.featuremaker import FeatureMaker
from src.data.tokenizer import EHRTokenizer
from src.data.split import Splitter
from src.data_fixes.infer import Inferrer
from src.data_fixes.handle import Handler
from src.data_fixes.exclude import Excluder


@hydra.main(config_path="../../configs/data", config_name="pretrain")
def main_data(cfg):
    # Save config
    with open(os.path.join(cfg.paths.data_dir, "data_config.json"), "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f)

    # Load concepts
    concepts, patients_info = ConceptLoader()(**cfg.loader)

    # Infer missing values
    concepts = Inferrer()(concepts)

    # Create feature sequences
    features = FeatureMaker(cfg)(concepts, patients_info)

    # Overwrite/drop nans and other incorrect values
    features = Handler()(features)

    # Exclude patients with <k concepts
    features = Excluder()(features, dir=cfg.paths.extra_dir)

    # Save final features
    torch.save(features, os.path.join(cfg.paths.data_dir, "features.pt"))

    # Split features
    ## Pretrain split
    pretrain_splits = Splitter(
        cfg,
        split_name="pretrain_splits.pt",
    )(features, mode="random")

    # Tokenize
    tokenizer = EHRTokenizer(cfg.tokenizer)
    train_encoded = tokenizer(pretrain_splits["train"])  # Re-do for ease of use
    tokenizer.freeze_vocabulary(
        vocab_name=os.path.join(cfg.paths.data_dir, cfg.paths.vocabulary)
    )

    # Save features
    for set, data in pretrain_splits.items():
        encoded = tokenizer(data)
        torch.save(
            encoded,
            os.path.join(cfg.paths.data_dir, f"{set}_{cfg.paths.encoded_suffix}.pt"),
        )


if __name__ == "__main__":
    main_data()
