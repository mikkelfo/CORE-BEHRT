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
from src.downstream_tasks.outcomes import OutcomeMaker


@hydra.main(config_path="configs/data", config_name="data")
def main_data(cfg):
    # Save config
    with open(os.path.join(cfg.paths.data_dir, "data_config.json"), "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f)

    # Load concepts
    concepts, patients_info = ConceptLoader()(**cfg.loader)

    # Infer missing values
    concepts = Inferrer()(concepts)

    # Make outcomes
    patients_info = OutcomeMaker(cfg)(patients_info)

    # Create feature sequences and outcomes
    features, outcomes = FeatureMaker(cfg)(concepts, patients_info)

    # Overwrite nans and other incorrect values
    features = Handler()(features)

    # Exclude patients with <k concepts
    features, outcomes = Excluder()(features, outcomes, dir=cfg.paths.extra_dir)

    # Save final features and outcomes
    torch.save(features, os.path.join(cfg.paths.data_dir, "features.pt"))
    torch.save(outcomes, os.path.join(cfg.paths.data_dir, "outcomes.pt"))

    # Split
    train_features, test_features, val_features = Splitter()(
        features, dir=cfg.paths.extra_dir
    )
    train_outcomes, test_outcomes, val_outcomes = Splitter()(
        outcomes, dir=cfg.paths.extra_dir
    )

    # Tokenize
    tokenizer = EHRTokenizer(cfg.tokenizer)
    train_encoded = tokenizer(train_features)
    tokenizer.freeze_vocabulary(
        vocab_name=os.path.join(cfg.paths.data_dir, cfg.paths.vocabulary)
    )
    test_encoded = tokenizer(test_features)
    val_encoded = tokenizer(val_features)

    # Save features
    for set, encoded in zip(
        ["train", "test", "val"], [train_encoded, test_encoded, val_encoded]
    ):
        path = os.path.join(cfg.paths.data_dir, f"{set}_{cfg.paths.encoded_suffix}.pt")
        torch.save(encoded, path)
    # Save outcomes
    for set, outcomes in zip(
        ["train", "test", "val"], [train_outcomes, test_outcomes, val_outcomes]
    ):
        path = os.path.join(cfg.paths.data_dir, f"{set}_{cfg.paths.outcomes_suffix}.pt")
        torch.save(outcomes, path)


if __name__ == "__main__":
    main_data()
