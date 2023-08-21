import os
import torch
import hydra
from src.data.tokenizer import EHRTokenizer
from src.data.split import Splitter


@hydra.main(config_path="../../configs/data", config_name="finetune")
def main_data(cfg):
    # Load features
    features = torch.load(os.path.join(cfg.paths.data_dir, "features.pt"))

    # Split features (test is optional)
    train_features, val_features, test_features = Splitter(cfg, mode="load")(
        features, file="covid_splits.pt"
    )

    # Tokenize
    tokenizer = EHRTokenizer(
        cfg.tokenizer, vocabulary=os.path.join(cfg.paths.data_dir, cfg.paths.vocabulary)
    )
    train_encoded = tokenizer(train_features)
    val_encoded = tokenizer(val_features)
    test_encoded = tokenizer(test_features)

    feature_set = [
        ("train", train_encoded),
        ("val", val_encoded),
        ("test", test_encoded),
    ]

    # Save features
    for set, encoded in feature_set:
        torch.save(
            encoded,
            os.path.join(cfg.paths.data_dir, f"{set}_{cfg.paths.encoded_suffix}.pt"),
        )


if __name__ == "__main__":
    main_data()
