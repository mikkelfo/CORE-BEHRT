"""Create tokenized features from formatted data. config template: data.yaml"""
import os
from os.path import join

import torch
from common.azure import setup_azure
from common.config import load_config
from common.setup import prepare_directory_outcomes, get_args
from common.utils import check_patient_counts
from data.concept_loader import ConceptLoader
from downstream_tasks.outcomes import OutcomeMaker

args = get_args('data_finetune.yaml')
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)

            
def process_data(loader, cfg, features_cfg, logger):
    concepts, patients_info = loader()
    check_patient_counts(concepts, patients_info, logger)
    pids = concepts.PID.unique()
    outcomes = OutcomeMaker(cfg, features_cfg)(concepts, patients_info, pids)
    return outcomes

def main_data(config_path):
    cfg = load_config(config_path)
    outcome_dir = join(cfg.features_dir, 'outcomes')
    features_cfg = load_config(join(cfg.features_dir, 'data_config.yaml'))
    if cfg.env=='azure':
        _, mount_context = setup_azure(cfg.run_name)
        mount_dir = mount_context.mount_point
        cfg.loader.data_dir = join(mount_dir, cfg.loader.data_dir) # specify paths here
        cfg.features_path = join(mount_dir, cfg.features_path)
        outcome_dir = 'output/outcomes'
    
    logger = prepare_directory_outcomes(config_path, outcome_dir, cfg.outcomes_name)
    logger.info('Mount Dataset')
    logger.info('Starting outcomes creation')
    outcomes = process_data(ConceptLoader(**cfg.loader), cfg, features_cfg, logger)
    
    torch.save(outcomes, join(outcome_dir, f'{cfg.outcomes_name}.pt'))
    
    logger.info('Finish outcomes creation')

    if cfg.env=='azure':
        mount_context.stop()

if __name__ == '__main__':
    main_data(config_path)

