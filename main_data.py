import torch
import hydra
from omegaconf import OmegaConf
import json
from data.dataset import MLMLargeDataset
from data.concept_loader import ConceptLoader
from data_fixes.infer import Inferrer
from data.featuremaker import FeatureMaker
from data_fixes.handle import Handler
from data_fixes.exclude import Excluder
from data.tokenizer import EHRTokenizer
from data.split import Splitter
from downstream_tasks.outcomes import OutcomeMaker
import numpy as np
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

    def split_batches(num_batches, split_ratios):
        batches = np.arange(num_batches)
        np.random.shuffle(batches)
        # calculate the number of batches for each set
        train_end = int(split_ratios['train'] * len(batches))
        val_end = train_end + int(split_ratios['val'] * len(batches))
        # split the batches into train, validation and test
        train_batches = batches[:train_end]
        val_batches = batches[train_end:val_end]
        test_batches = batches[val_end:]
        return train_batches, val_batches, test_batches

    def tokenize(tokenzier, batches, mode='train'):
        for batch in batches:
            features = torch.load(join(cfg.out_dir, f'features{batch}.pt'))
            train_encoded = tokenizer(features)
            torch.save(train_encoded, join(cfg.out_dir, f'encoded_{mode}{batch}.pt'))


    num_batches = 0
    for i, concept_batch, patient_batch in enumerate(ConceptLoader(**cfg.loader)):
        print("Loading concepts...")
        concepts, patients_info = ConceptLoader()(**cfg.loader)
        concepts = Inferrer()(concepts)
        patients_info = OutcomeMaker(cfg)(patients_info)
        print("Creating feature sequences")
        features = FeatureMaker(cfg)(concepts, patients_info)
        features = Handler()(features)

        print("Exclude patients with <k concepts")
        features = Excluder()(features, k=cfg.min_concepts)
        torch.save(features, join(cfg.out_dir, f'features{i}.pt'))
        num_batches += 1
    # split batches into train, test, val
    train_batches, val_batches, test_batches = split_batches(num_batches, cfg.split_ratios)
    
    # Tokenize
    # Loop through train batches before freezing tokenizer
    if not os.path.exists(cfg.out_dir):
        os.makedirs(cfg.out_dir)
    tokenizer = EHRTokenizer(config=cfg.tokenizer)        
    train_files = tokenize(tokenizer, train_batches, mode='train')
    tokenizer.freeze_vocabulary()
    tokenizer.save_vocab(join(cfg.out_dir, 'vocabulary.pt'))
    val_files = tokenize(tokenizer, val_batches, mode='val')
    test_files = tokenize(tokenizer, test_batches, mode='test')
    
    print("Dataset with MLM and prolonged length of stay")
    train_dataset = MLMLargeDataset(train_files, vocabulary=tokenizer.vocabulary, **cfg.dataset)
    test_dataset = MLMLargeDataset(test_files, vocabulary=tokenizer.vocabulary, **cfg.dataset)
    val_dataset = MLMLargeDataset(val_files, vocabulary=tokenizer.vocabulary, **cfg.dataset)
    torch.save(train_dataset, 'train_dataset.pt')
    torch.save(test_dataset, 'test_dataset.pt')
    torch.save(val_dataset, 'val_dataset.pt')

if __name__ == '__main__':
    main_data()

