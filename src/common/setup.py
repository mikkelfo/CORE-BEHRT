import os
import torch
import json
from omegaconf import OmegaConf
from src.data.concept_loader import ConceptLoader
from src.data_fixes.infer import Inferrer
from src.data.featuremaker import FeatureMaker
from src.data_fixes.handle import Handler
from src.data_fixes.exclude import Excluder


def get_features(cfg):
    # Load concepts
    concepts, patients_info = ConceptLoader()(**cfg.loader)

    # Infer missing values
    concepts = Inferrer()(concepts)

    # Create feature sequences
    features = FeatureMaker(cfg)(concepts, patients_info)

    # Overwrite/drop nans and other incorrect values
    features = Handler(cfg)(features)

    # Exclude patients
    features = Excluder(cfg)(features)

    # Save final features
    torch.save(features, os.path.join(cfg.paths.data_dir, "features.pt"))

    return features


def save_splits(cfg, splits: dict, tokenizer=None):
    for set, split_feature in splits.items():
        # Tokenize if tokenizer is provided
        if tokenizer:
            split_feature = tokenizer(split_feature)

        torch.save(
            split_feature,
            os.path.join(cfg.paths.data_dir, f"{set}_{cfg.paths.encoded_suffix}.pt"),
        )


def save_config(cfg):
    # Save config
    with open(os.path.join(cfg.paths.data_dir, "data_config.json"), "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f)


def get_pids_with_exclusion(cfg):
    pids = torch.load(os.path.join(cfg.paths.extra_dir, "PIDs.pt"))
    excluder_kept_indices = torch.load(
        os.path.join(cfg.paths.extra_dir, "excluder_kept_indices.pt")
    )

    return [pids[i] for i in excluder_kept_indices]
