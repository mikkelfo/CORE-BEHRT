"""Create tokenized features from formatted data. config template: data.yaml"""
import os
from os.path import join

import torch

from common import azure
from common.config import load_config
from common.setup import prepare_directory
from data.concept_loader import ConceptLoader
from data.featuremaker import FeatureMaker
from data.split import Splitter
from data.tokenizer import EHRTokenizer
from data_fixes.exclude import Excluder
from data_fixes.handle import Handler

config_path = join('configs', 'data_pretrain.yaml')#
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

def check_patient_counts(concepts, patients_info, logger):
    if concepts.PID.nunique() != patients_info.PID.nunique():
            logger.warning(f"patients info contains {patients_info.PID.nunique()} patients != \
                        {concepts.PID.nunique()} unique patients in concepts")
            
def process_data(loader, handler, excluder, cfg, logger):
    concepts, patients_info = loader()
    check_patient_counts(concepts, patients_info, logger)
    features, pids = FeatureMaker(cfg.features)(concepts, patients_info)
    features = handler(features)
    features, _, kept_indices = excluder(features)
    kept_pids = [pids[idx] for idx in kept_indices]
    return features, kept_pids

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
    if cfg.env=='azure':
        _, mount_context = azure.setup_azure(cfg.run_name)
        mount_dir = mount_context.mount_point
        cfg.loader.data_dir = join(mount_dir, cfg.loader.data_dir) # specify paths here

    logger = prepare_directory(config_path, cfg)  
    logger.info('Mount Dataset')
    logger.info('Starting data processing')
    features, kept_pids = process_data(ConceptLoader(**cfg.loader), Handler(**cfg.handler), Excluder(**cfg.excluder), cfg, logger)
    
    torch.save(features, join(cfg.output_dir, 'features', f'features.pt'))
    torch.save(kept_pids, join(cfg.output_dir, 'features', f'pids_features.pt'))
    
    logger.info("Splitting data")
    splitter = Splitter(ratios=cfg.split_ratios)
    features_split, pids_split = splitter(features, kept_pids)

    logger.info("Saving split pids")
    for idx, mode in enumerate(['train', 'val', 'test']):
        torch.save(pids_split[mode], join(cfg.output_dir, f'{mode}_pids.pt'))
        torch.save(pids_split[mode], join(cfg.output_dir, 'features' , f'pids_features_{idx}.pt'))

    torch.save(kept_pids, join(cfg.output_dir, 'features', f'pids_features.pt'))

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    splitter.save(cfg.output_dir)
    

    logger.info("Saving split data")
    for mode in ['train', 'val', 'test']:
        torch.save(features_split[mode], join(cfg.output_dir, 'features', f'{mode}.pt'))
    
    logger.info("Tokenizing")
    tokenizer = EHRTokenizer(config=cfg.tokenizer)
    encoded_data = {}
    encoded_data['train'] = tokenizer(features_split['train'])
    tokenizer.freeze_vocabulary()
    tokenizer.save_vocab(join(cfg.output_dir, 'vocabulary.pt'))
    encoded_data['val'] = tokenizer(features_split['val'])
    encoded_data['test'] = tokenizer(features_split['test'])


    logger.info("Saving tokenized data")
    for idx, mode in enumerate(['train', 'val', 'test']):
        torch.save(encoded_data[mode], join(cfg.output_dir,'tokenized',f'tokenized_{mode}_{idx}.pt')) 
        torch.save([idx],join(cfg.output_dir, f'{mode}_file_ids.pt'))
    torch.save(tokenizer.vocabulary, join(cfg.output_dir, 'vocabulary.pt'))
    
    # Ensure compatibility with large dataset
    logger.info('Finished data processing')

    if cfg.env=='azure':
        mount_context.stop()

if __name__ == '__main__':
    main_data(config_path)

