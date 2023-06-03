import torch
import hydra
from omegaconf import OmegaConf
import json
from data.dataset import MLMLargeDataset
from data.batch import batch_tokenize, split_batches
from data.concept_loader import ConceptLoader
from data_fixes.infer import Inferrer
from data.featuremaker import FeatureMaker
from data_fixes.handle import Handler
from data_fixes.exclude import Excluder
from data.tokenizer import EHRTokenizer
from downstream_tasks.outcomes import OutcomeMaker
from os.path import join
import os


@hydra.main(config_path="configs/data", config_name="data")
def main_data(cfg):
    with open('data_config.json', 'w') as f:
        json.dump(
            OmegaConf.to_container(cfg, resolve=True)
        , f)

    """
        Loads data
        Infers nans
        Finds outcomes
        Creates features
        Handles wrong data
        Excludes patients with <k concepts
        Splits data
        Tokenizes
        Saves
    """

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    conceptloader = ConceptLoader(batch_size=50, chunksize=250, **cfg.loader)
    inferrer = Inferrer()
    # outcome_maker = OutcomeMaker(cfg)
    feature_maker = FeatureMaker(cfg)
    handler = Handler()
    excluder = Excluder(cfg)
    num_batches = 11
    print("Loading concepts...")
    for i, (concept_batch, patient_batch) in enumerate(conceptloader()):
        concept_batch = inferrer(concept_batch)
        # patient_batch = OutcomeMaker(cfg)(patient_batch)
        features_batch = feature_maker(concept_batch, patient_batch)
        # print(features_batch['concept'][0])
        features_batch = handler(features_batch)
        features_batch, _ = excluder(features_batch)
        torch.save(features_batch, join(cfg.output_dir, f'features{i}.pt'))
        num_batches += 1
    # split batches into train, test, val
    train_batches, val_batches, test_batches = split_batches(num_batches, cfg.split_ratios)
    
    # Tokenize
    # Loop through train batches before freezing tokenizer
    
    tokenizer = EHRTokenizer(config=cfg.tokenizer)        
    train_files = batch_tokenize(tokenizer, train_batches, mode='train')
    tokenizer.freeze_vocabulary()
    tokenizer.save_vocab(join(cfg.output_dir, 'vocabulary.pt'))
    val_files = batch_tokenize(tokenizer, val_batches, mode='val')
    test_files = batch_tokenize(tokenizer, test_batches, mode='test')
    print("Big data dataset")
    train_dataset = MLMLargeDataset(train_files, vocabulary=tokenizer.vocabulary, **cfg.dataset)
    test_dataset = MLMLargeDataset(test_files, vocabulary=tokenizer.vocabulary, **cfg.dataset)
    val_dataset = MLMLargeDataset(val_files, vocabulary=tokenizer.vocabulary, **cfg.dataset)
    torch.save(train_dataset, join(cfg.output_dir, 'train_dataset.pt'))
    torch.save(test_dataset, join(cfg.output_dir,'test_dataset.pt'))
    torch.save(val_dataset, join(cfg.output_dir,'val_dataset.pt'))

if __name__ == '__main__':
    main_data()

