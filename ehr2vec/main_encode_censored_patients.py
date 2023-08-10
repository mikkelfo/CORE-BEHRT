import os
from os.path import join

from common.config import load_config
from common import azure
from common.setup import prepare_encodings_directory
from common.loader import create_binary_outcome_datasets, load_model

from model.model import BertEHRModel
from evaluation.encodings import Forwarder
from common.utils import ConcatIterableDataset

config_path = join("configs", "encode_censored_test.yaml")
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

run_name = "encode_censored_patients"

def main_encode():
    cfg = load_config(config_path)
    run = None

    if cfg.env=='azure':
        run, mount_context = azure.setup_azure(cfg.paths.run_name)
        cfg.paths.data_path = join(mount_context.mount_point, cfg.paths.data_path)
        cfg.paths.model_path = join(mount_context.mount_point, cfg.paths.model_path)
        cfg.paths.outcomes_path = join(mount_context.mount_point, cfg.paths.outcomes_path)
    # Finetune specific
    logger = prepare_encodings_directory(config_path, cfg)
    logger.info(f"Access data from {cfg.paths.data_path}")
    logger.info(f"Access outcomes from {cfg.paths.outcomes_path}")
    logger.info(f'Outcome name: {cfg.outcome.type}')
    logger.info(f'Censor name: {cfg.outcome.censor_type}')
    logger.info(f"Censoring {cfg.outcome.n_hours} hours after censor_outcome")
    logger.info("Create Datasets")
    train_dataset, val_dataset, _ = create_binary_outcome_datasets(cfg)
    
    #binary_outcomes = pd.notna(outcome)
    dataset = ConcatIterableDataset([train_dataset, val_dataset])

    logger.info('Initializing model')
    model = load_model(BertEHRModel, cfg)

    forwarder = Forwarder( 
        model=model, 
        dataset=dataset, 
        run=run,
        logger=logger,
        output_path=join(cfg.output_dir, cfg.file_name),
        **cfg.forwarder_args,
    )
    forwarder.forward_patients()

    if cfg.env=='azure':
        from azure_run import file_dataset_save
        from pathlib import Path
        path = Path(cfg.paths.model_path)
        model_path = Path(*path.parts[2:]) # first two paths are mount point 
        file_dataset_save(local_path=cfg.output_dir, datastore_name = "workspaceblobstore",
                    remote_path = join("PHAIR", model_path, 'encodings', 'censored_patients'))
        mount_context.stop()

        
if __name__ == '__main__':
    main_encode()