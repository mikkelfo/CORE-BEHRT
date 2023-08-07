"""Create tokenized features from formatted data. config template: data.yaml"""
import os
from collections import defaultdict
from os.path import join

import torch
from common import azure
from common.config import load_config
from common.logger import TqdmToLogger
from common.setup import prepare_directory_outcomes
from common.utils import check_patient_counts
from data.concept_loader import ConceptLoaderLarge
from downstream_tasks.outcomes import OutcomeMaker
from tqdm import tqdm

config_path = join('configs', 'data_finetune_test.yaml')
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

            
def process_data(loader, cfg, features_cfg, logger):
    all_outcomes = defaultdict(list)
    for (concept_batch, patient_batch) in tqdm(loader(), desc='Batch Process Data', file=TqdmToLogger(logger)):
        check_patient_counts(concept_batch, patient_batch, logger)
        pids = concept_batch.PID.unique()
        outcomes = OutcomeMaker(cfg, features_cfg)(concept_batch, patient_batch, pids)
        for key, value in outcomes.items():
            all_outcomes[key].extend(value)
    return all_outcomes

def main_data(config_path):
    cfg = load_config(config_path)
    outcome_dir = join(cfg.features_dir, 'outcomes')
    features_cfg = load_config(join(cfg.features_dir, 'data_config.yaml'))
    if cfg.env=='azure':
        _, mount_context = azure.setup_azure(cfg.run_name)
        mount_dir = mount_context.mount_point
        cfg.loader.data_dir = join(mount_dir, cfg.loader.data_dir) # specify paths here
        cfg.features_path = join(mount_dir, cfg.features_path)
        outcome_dir = 'output/outcomes'
    
    logger = prepare_directory_outcomes(config_path, outcome_dir, cfg.outcomes_name)
    logger.info('Mount Dataset')
    logger.info('Starting outcomes creation')
    outcomes = process_data(ConceptLoaderLarge(**cfg.loader), cfg, features_cfg, logger)
    
    torch.save(outcomes, join(outcome_dir, f'{cfg.outcomes_name}.pt'))
    
    logger.info('Finish outcomes creation')

    if cfg.env=='azure':
        mount_context.stop()

if __name__ == '__main__':
    main_data(config_path)

