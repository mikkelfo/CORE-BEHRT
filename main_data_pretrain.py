import os
from os.path import join
from shutil import copyfile

import torch
from tqdm import tqdm

from data import utils
from data.batch import batch_tokenize, split_batches
from data.concept_loader import ConceptLoader
from data.dataset import MLMLargeDataset
from data.featuremaker import FeatureMaker
from data.tokenizer import EHRTokenizer
from data_fixes.exclude import Excluder
from data_fixes.handle import Handler
from data_fixes.infer import Inferrer

config_path = join("configs", "data.yaml")
cfg = utils.load_config(config_path)
def main_data(cfg):
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    copyfile(config_path, join(cfg.output_dir, 'data_config.yaml'))
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


    conceptloader = ConceptLoader(**cfg.loader)
    inferrer = Inferrer()
    # outcome_maker = OutcomeMaker(cfg)
    feature_maker = FeatureMaker(cfg)
    handler = Handler()
    excluder = Excluder(cfg)
    num_batches = 0
    num_patients = 0
    for i, (concept_batch, patient_batch) in enumerate(tqdm(conceptloader(), desc='Process')):
        num_patients += len(patient_batch)
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
    print('Tokenizing')
    tokenizer = EHRTokenizer(config=cfg.tokenizer)        
    train_files = batch_tokenize(cfg, tokenizer, train_batches, mode='train')
    tokenizer.freeze_vocabulary()
    tokenizer.save_vocab(join(cfg.output_dir, 'vocabulary.pt'))
    val_files = batch_tokenize(cfg, tokenizer, val_batches, mode='val')
    test_files = batch_tokenize(cfg, tokenizer, test_batches, mode='test')
    print('Saving datasets')
    cfg.dataset.vocabulary = tokenizer.vocab
    cfg.dataset.num_patients = num_patients
    train_dataset = MLMLargeDataset(train_files, **cfg.dataset)
    test_dataset = MLMLargeDataset(test_files, **cfg.dataset)
    val_dataset = MLMLargeDataset(val_files, **cfg.dataset)
    torch.save(train_dataset, join(cfg.output_dir, 'train_dataset.pt'))
    torch.save(test_dataset, join(cfg.output_dir,'test_dataset.pt'))
    torch.save(val_dataset, join(cfg.output_dir,'val_dataset.pt'))

if __name__ == '__main__':
    main_data(cfg)

