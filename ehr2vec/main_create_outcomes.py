"""Create tokenized features from formatted data. config template: data.yaml"""
import os
from os.path import join

import torch
from ehr2vec.common.azure import save_to_blobstore, setup_azure
from ehr2vec.common.config import load_config
from ehr2vec.common.setup import DirectoryPreparer, get_args
from ehr2vec.common.utils import check_patient_counts
from ehr2vec.data.concept_loader import ConceptLoader
from ehr2vec.downstream_tasks.outcomes import OutcomeMaker

args = get_args('outcomes_test.yaml')
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)
BLOBSTORE = 'PHAIR'
            
def process_data(loader, cfg, features_cfg, logger):
    concepts, patients_info = loader()
    check_patient_counts(concepts, patients_info, logger)
    pids = concepts.PID.unique()
    outcomes = OutcomeMaker(cfg, features_cfg)(concepts, patients_info, pids)
    return outcomes

def main_data(config_path):
    cfg = load_config(config_path)
    outcome_dir = join(cfg.features_dir, 'outcomes', cfg.outcomes_name)
    
    if cfg.env=='azure':
        _, mount_context = setup_azure(cfg.run_name, dataset_name=BLOBSTORE)
        mount_dir = mount_context.mount_point
        cfg.loader.data_dir = join(mount_dir, cfg.loader.data_dir) # specify paths here
        cfg.features_dir = join(mount_dir, cfg.features_dir)
        outcome_dir = 'output/outcomes'
    features_cfg = load_config(join(cfg.features_dir, 'data_config.yaml'))
    logger = DirectoryPreparer(config_path).prepare_directory_outcomes(outcome_dir, cfg.outcomes_name)
    logger.info('Mount Dataset')
    logger.info('Starting outcomes creation')
    outcomes = process_data(ConceptLoader(**cfg.loader), cfg, features_cfg, logger)
    
    torch.save(outcomes, join(outcome_dir, f'{cfg.outcomes_name}.pt'))
    
    logger.info('Finish outcomes creation')

    if cfg.env=='azure':
        save_to_blobstore(local_path='outcomes' , 
                          remote_path=join(BLOBSTORE, 'outcomes', cfg.paths.run_name))
        mount_context.stop()

if __name__ == '__main__':
    main_data(config_path)

