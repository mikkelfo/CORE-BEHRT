import os
import hydra
import src.common.setup as setup
from src.data.split import Splitter
from src.data.adapter import DataAdapter
from src.data.tokenizer import EHRTokenizer


@hydra.main(config_path="../../configs/data", config_name="mimic")
def main_data(cfg):
    # Save config
    setup.save_config(cfg)

    # Get features (load, infer, create, handle, exclude)
    features = setup.get_features(cfg)

    # Split features
    ## First we get the test_split and isolate the test_set from the pretraining set
    splits = Splitter(
        cfg, dir="data/mimic3", split_name="mimic_splits.pt"
    ).mimic_split()
    pretrain_features, _ = Splitter.isolate_holdout(features, splits["test"])
    ## Pretrain split (single-visit)
    single_visit_features = DataAdapter().adapt_to_single_visit(pretrain_features)
    pretrain_splits = Splitter(cfg, dir="data/mimic3", split_name="pretrain_splits.pt")(
        single_visit_features,
        mode="random",
        ratios={"pretrain_train": 0.8, "pretrain_val": 0.2},
    )

    ## Finetuning split (multi-visit) - we use the splits found above
    finetune_splits = Splitter(cfg, dir="data/mimic3", split_name="finetune_splits.pt")(
        features, file="mimic_splits.pt", mode="load"
    )

    # Tokenize
    tokenizer = EHRTokenizer(cfg.tokenizer)
    ## Not used (we do it again during saving for ease of use)
    _ = tokenizer(pretrain_splits["pretrain_train"])
    tokenizer.freeze_vocabulary(
        vocab_name=os.path.join(cfg.paths.data_dir, cfg.paths.vocabulary)
    )

    feature_set = pretrain_splits | finetune_splits

    # Save features
    setup.save_splits(cfg, feature_set, tokenizer)


if __name__ == "__main__":
    main_data()
