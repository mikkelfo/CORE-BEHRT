from os.path import join

import torch
from tqdm import tqdm

from common.config import load_config, prepare_directory
from data.batch import Batches, BatchTokenize
from data.concept_loader import ConceptLoader
from data.dataset import MLMLargeDataset
from data.featuremaker import FeatureMaker
from data.tokenizer import EHRTokenizer
from data_fixes.exclude import Excluder
from data_fixes.handle import Handler

config_path = join("configs", "data.yaml")
cfg = load_config(config_path)


def main_data(cfg):
    """
        Loads data
        Finds outcomes
        Creates features
        Handles wrong data
        Excludes patients with <k concepts
        Splits data
        Tokenizes
        Saves
    """
    logger = prepare_directory(config_path, cfg)
    
    conceptloader = ConceptLoader(**cfg.loader)
    feature_maker = FeatureMaker(cfg)
    handler = Handler(**cfg.handler)
    excluder = Excluder(cfg)
    logger.info('Starting data processing')
    pids = []
    for i, (concept_batch, patient_batch) in enumerate(tqdm(conceptloader(), desc='Process')):
        pids.append(patient_batch['PID'].tolist())
        features_batch = feature_maker(concept_batch, patient_batch)
        features_batch = handler(features_batch)
        features_batch, _ = excluder(features_batch)
        torch.save(features_batch, join(cfg.output_dir, f'features_{i}.pt'))
    logger.info('Finished data processing')
    logger.info('Splitting batches')
    batches = Batches(cfg, pids)
    batches.split_and_save()

    logger.info('Tokenizing')
    tokenizer = EHRTokenizer(config=cfg.tokenizer)
    batch_tokenize = BatchTokenize(tokenizer, cfg)
    train_files, val_files, test_files = batch_tokenize.tokenize(batches)

    logger.info('Creating datasets')
    cfg.dataset.vocabulary = tokenizer.vocabulary
    train_dataset = MLMLargeDataset(train_files, **{**cfg.dataset, 'pids': batches.train.pids})
    test_dataset = MLMLargeDataset(test_files, **{**cfg.dataset, 'pids': batches.test.pids})
    val_dataset = MLMLargeDataset(val_files, **{**cfg.dataset, 'pids': batches.val.pids})
    logger.info('Saving datasets')
    torch.save(train_dataset, join(cfg.output_dir, 'train_dataset.pt'))
    torch.save(test_dataset, join(cfg.output_dir,'test_dataset.pt'))
    torch.save(val_dataset, join(cfg.output_dir,'val_dataset.pt'))

if __name__ == '__main__':
    main_data(cfg)

