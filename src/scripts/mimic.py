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


@hydra.main(config_path="../../configs/data", config_name="mimic")
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

    # Exclude patients
    features = Excluder(cfg)(features)

    # Save final features
    torch.save(features, os.path.join(cfg.paths.data_dir, "features.pt"))

    # Split features (test is optional)
    train_features, val_features, *test_features = Splitter(
        cfg, split_name="pretrain_splits.pt", mode="random"
    )(features)

    # Tokenize
    tokenizer = EHRTokenizer(cfg.tokenizer)
    train_encoded = tokenizer(train_features)
    tokenizer.freeze_vocabulary(
        vocab_name=os.path.join(cfg.paths.data_dir, cfg.paths.vocabulary)
    )
    val_encoded = tokenizer(val_features)

    feature_set = [("train", train_encoded), ("val", val_encoded)]

    if test_features:
        test_encoded = tokenizer(test_features[0])
        feature_set.append(("test", test_encoded))

    # Save features
    for set, encoded in feature_set:
        torch.save(
            encoded,
            os.path.join(cfg.paths.data_dir, f"{set}_{cfg.paths.encoded_suffix}.pt"),
        )


if __name__ == "__main__":
    main_data()
