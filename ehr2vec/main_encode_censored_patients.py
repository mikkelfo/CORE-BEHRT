"""
!Currently, this script is not used in the paper. It is a first attempt to evaluate the RF model.

"""

import os
import shutil
from os.path import join

import torch
from ehr2vec.common.azure import AzurePathContext, save_to_blobstore
from ehr2vec.common.config import load_config
from ehr2vec.common.io import PatientHDF5Writer
from ehr2vec.common.loader import ModelLoader
from ehr2vec.common.logger import close_handlers
from ehr2vec.common.setup import DirectoryPreparer, get_args, setup_logger
from ehr2vec.data.prepare_data import DatasetPreparer
from ehr2vec.evaluation.encodings import Forwarder
from ehr2vec.evaluation.utils import validate_outcomes
from ehr2vec.model.model import BertEHREncoder

BLOBSTORE = 'PHAIR'
CONFIG_PATH = 'encode_censored.yaml'

args = get_args(CONFIG_PATH, "encode_censored")
config_path = args.config_path
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

def _get_output_path_name(dataset, cfg):
    num_patients = str(int((len(dataset))/1000))+'k'
    if cfg.outcome.censor_type:
        days = True if abs(cfg.outcome.n_hours)>48 else False
        window = int(abs(cfg.outcome.n_hours/24)) if days else abs(cfg.outcome.n_hours)
        days_hours = 'days' if days else 'hours'
        pre_post = 'pre' if cfg.outcome.n_hours<0 else 'post'
        return f"{cfg.outcome.type}_Patients_{num_patients}_Censor_{window}{days_hours}_{pre_post}_{cfg.outcome.censor_type}"
    else:
        if cfg.outcome.type:
            return f"{cfg.outcome.type}_Patients_{num_patients}_Uncensored"
        else:
            return f"Patients_{num_patients}_Uncensored"

def main_encode():
    os.makedirs(join('outputs','tmp'), exist_ok=True)
    logger = setup_logger(join('outputs', 'tmp'))
    encodings_file_name = 'encodings.h5'
    logger.info('Loading config')
    cfg = load_config(config_path)
    run = None
    model_path = cfg.paths.model_path
    censored_patients_path = join(model_path, 'encodings', 'censored_patients')
    logger.info(f"Access data from {cfg.paths.data_path}")
    logger.info(f"Access outcomes from {cfg.paths.outcomes_path}")
    
    cfg.output_dir = censored_patients_path
    cfg, run, mount_context = AzurePathContext(cfg, dataset_name=BLOBSTORE).azure_encode_setup()    

    
    output_dir = cfg.output_dir # we will modify cfg. output_dir
    all_outcomes = torch.load(cfg.paths.outcomes_path)
    validate_outcomes(all_outcomes, cfg)
    for i, outcome in enumerate(cfg.outcomes):
        cfg.outcome = cfg.outcomes[outcome]

        logger.info(f'Outcome name: {cfg.outcome.type}')
        logger.info(f"Censoring {cfg.outcome.n_hours} hours after {cfg.outcome.censor_type}")
        logger.info("Creating datasets")
        dataset_preparer = DatasetPreparer(cfg)
        dataset = dataset_preparer.prepare_finetune_data() 
        
        output_path_name = _get_output_path_name(dataset, cfg)
        cfg.output_dir = join(output_dir, output_path_name)
        
        if i==0:
            close_handlers()
            logger = DirectoryPreparer(config_path).prepare_encodings_directory(cfg)
            shutil.copy(join('outputs','tmp', 'info.log'), join(cfg.output_dir, 'info.log'))
            logger.info('Deleting tmp directory')
            shutil.rmtree(join('outputs', 'tmp'))
        else:
            shutil.copyfile(config_path, join(cfg.output_dir, 'encodings_config.yaml'))
            
        logger.info(f"Store in directory with name: {_get_output_path_name(dataset, cfg)}")
        logger.info('Initializing model')
        model = ModelLoader(cfg).load_model(BertEHREncoder)

        forwarder = Forwarder( 
            model=model, 
            dataset=dataset, 
            run=run,
            logger=logger,
            writer=PatientHDF5Writer(join(cfg.output_dir, encodings_file_name)),
            **cfg.forwarder_args,
        )
        forwarder.forward_patients()

        if cfg.env=='azure':
            save_to_blobstore(cfg.paths.output_path, join(BLOBSTORE, censored_patients_path))
    if cfg.env=='azure':
        mount_context.stop()
    logger.info('Done')
        
if __name__ == '__main__':
    main_encode()