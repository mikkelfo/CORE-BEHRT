"""Create tokenized features from formatted data. config template: data.yaml"""
import os
from os.path import join

import torch
from common.azure import AzurePathContext, save_to_blobstore
from common.config import load_config
from common.setup import DirectoryPreparer, get_args
from common.utils import (check_directory_for_features, check_existing_splits,
                          check_patient_counts)
from data.concept_loader import ConceptLoader
from data.featuremaker import FeatureMaker
from data.split import Splitter
from data.tokenizer import EHRTokenizer
from data_fixes.exclude import Excluder
from data_fixes.handle import Handler

BLOBSTORE = 'PHAIR'

args = get_args('data_pretrain.yaml', 'data_pretrain')
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)


def main_data(config_path):
    """
        Loads data
        Creates features
        Handles wrong data
        Excludes patients with <k concepts
        Splits data
        Tokenizes
        Saves
    """
    cfg = load_config(config_path)
    cfg, _, mount_context = AzurePathContext(cfg, dataset_name=BLOBSTORE).azure_data_pretrain_setup()

    logger = DirectoryPreparer(config_path).prepare_directory(cfg)  
    logger.info('Mount Dataset')
    logger.info('Starting data processing')
    if not check_directory_for_features(cfg.loader.data_dir):
        features, pids = process_data(ConceptLoader(**cfg.loader), Handler(**cfg.handler), Excluder(**cfg.excluder), cfg, logger)
        torch.save(features, join(cfg.output_dir, 'features', f'features.pt'))
        torch.save(pids, join(cfg.output_dir, 'features', 'pids_features.pt'))
    else:
        pids = torch.load(join(cfg.loader.data_dir, 'features', 'pids_features.pt'))
    
    logger.info("Splitting data")
    if not check_existing_splits(cfg.loader.data_dir):
        splitter = Splitter(ratios=cfg.split_ratios)
        features_split, pids_split = splitter(features, pids)
        for idx, mode in enumerate(['train', 'val', 'test']):
            torch.save(pids_split[mode], join(cfg.output_dir, 'features', f'{mode}_pids.pt'))
            torch.save(features_split[mode], join(cfg.output_dir, 'features' , f'{mode}_features.pt'))
    else:
        features_split = {}
        pids_split = {}
        for mode in ['train', 'val', 'test']:
            features_split[mode] = torch.load(join(cfg.loader.data_dir, 'features', f'{mode}_features.pt'))
            pids_split[mode] = torch.load(join(cfg.loader.data_dir, 'features', f'{mode}_pids.pt'))

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    splitter.save(cfg.output_dir)
    
    logger.info("Tokenizing")
    tokenizer = EHRTokenizer(config=cfg.tokenizer)
    encoded_data = {}
    encoded_data['train'] = tokenizer(features_split['train'])
    tokenizer.freeze_vocabulary()
    tokenizer.save_vocab(join(cfg.output_dir, cfg.tokenized_dir_name, 'vocabulary.pt'))
    encoded_data['val'] = tokenizer(features_split['val'])
    encoded_data['test'] = tokenizer(features_split['test'])


    logger.info("Saving tokenized data")
    for idx, mode in enumerate(['train', 'val', 'test']):
        torch.save(encoded_data[mode], join(cfg.output_dir, cfg.tokenized_dir_name, f'tokenized_{mode}.pt')) 
        torch.save(pids_split[mode], join(cfg.output_dir, cfg.tokenized_dir_name, f'{mode}_pids.pt'))
    # Ensure compatibility with large dataset
    logger.info('Finished data processing')

    if cfg.env=='azure':
        save_to_blobstore(local_path=cfg.run_name, 
                          remote_path=join(BLOBSTORE, 'features', cfg.run_name))
        mount_context.stop()

def process_data(loader, handler, excluder, cfg, logger):
    concepts, patients_info = loader()
    check_patient_counts(concepts, patients_info, logger)
    features, pids = FeatureMaker(cfg.features)(concepts, patients_info)
    features = handler(features)
    features, _, kept_indices = excluder(features)
    kept_pids = [pids[idx] for idx in kept_indices]
    return features, kept_pids

if __name__ == '__main__':
    main_data(config_path)

