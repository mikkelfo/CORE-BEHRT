"""Create tokenized features from formatted data. config template: data.yaml"""
import os
from os.path import join

import torch

from common import azure
from common.utils import check_patient_counts
from common.config import load_config
from common.setup import prepare_directory
from data.concept_loader import ConceptLoader
from downstream_tasks.outcomes import OutcomeMaker 
from data.split import Splitter
from data.tokenizer import EHRTokenizer
from data_fixes.exclude import Excluder
from data_fixes.handle import Handler

config_path = join('configs', 'data_finetune.yaml')#
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)


            
def process_data(loader, cfg, logger):
    concepts, patients_info = loader()
    check_patient_counts(concepts, patients_info, logger)
    pids = concepts.PID.unique()
    outcomes, pids = OutcomeMaker(cfg.outcomes)(concepts, patients_info, pids)
    return outcomes, pids

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
    outcomes, pids = process_data(ConceptLoader(**cfg.loader), cfg, logger)
    
    torch.save(outcomes, join(cfg.output_dir, 'outcomes', f'outcomes.pt'))
    torch.save(pids, join(cfg.output_dir, 'outcomes', f'pids_outcomes.pt'))
    
    logger.info("Splitting data")
    splitter = Splitter(ratios=cfg.split_ratios)
    features_split, pids_split = splitter(outcomes, pids)

    logger.info("Saving split pids")
    for idx, mode in enumerate(['train', 'val', 'test']):
        torch.save(pids_split[mode], join(cfg.output_dir, f'{mode}_pids.pt'))
        torch.save(pids_split[mode], join(cfg.output_dir, 'outcomes' , f'pids_outcomes_{idx}.pt'))

    torch.save(pids, join(cfg.output_dir, 'outcomes', f'pids_outcomes.pt'))

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

