import os
import torch
import hydra
import json
from omegaconf import OmegaConf
from src.data.concept_loader import ConceptLoader
from src.data.featuremaker import FeatureMaker
from src.data.tokenizer import EHRTokenizer
from src.data.split import Splitter
from src.data.adapter import DataAdapter
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
    features = Handler(cfg)(features)

    # Exclude patients
    features = Excluder(cfg)(features)

    # Save final features
    torch.save(features, os.path.join(cfg.paths.data_dir, "features.pt"))

    # Split features
    ## First we get the test_split and isolate the test_set from the pretraining set
    _, _, test_split = Splitter().mimic_split(
        dir="data/mimic3", split_name="mimic_splits.pt"
    )
    pretrain_features, _ = Splitter().isolate_holdout(features, test_split)
    ## Pretrain split (single-visit)
    single_visit_features = DataAdapter().adapt_to_single_visit(pretrain_features)
    pretrain_train, pretrain_val = Splitter(
        cfg, dir="data/mimic3", split_name="pretrain_splits.pt"
    )(single_visit_features, mode="random", ratios=[0.8, 0.2])

    ## Finetuning split (multi-visit) - we use the splits found above
    finetune_train, finetune_val, finetune_test = Splitter(
        cfg, dir="data/mimic3", split_name="finetune_splits.pt"
    )(features, file="mimic_splits.pt", mode="load")

    # Tokenize
    tokenizer = EHRTokenizer(cfg.tokenizer)
    ## Not used (we do it again during saving for ease of use)
    pretrain_train_encoded = tokenizer(pretrain_train)
    tokenizer.freeze_vocabulary(
        vocab_name=os.path.join(cfg.paths.data_dir, cfg.paths.vocabulary)
    )

    feature_set = [
        ("pretrain_train", pretrain_train),
        ("val", pretrain_val),
        ("finetune_train", finetune_train),
        ("finetune_val", finetune_val),
        ("finetune_test", finetune_test),
    ]

    # Save features
    for set, split_feature in feature_set:
        encoded = tokenizer(split_feature)
        torch.save(
            encoded,
            os.path.join(cfg.paths.data_dir, f"{set}_{cfg.paths.encoded_suffix}.pt"),
        )


if __name__ == "__main__":
    main_data()
