import os
from os.path import join

from common.config import load_config
from common import azure
from common.setup import setup_run_folder
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
    # Finetune specific
    logger = setup_run_folder(cfg)
    logger.info(f"Access data from {cfg.paths.data_path}")
    logger.info(f'Outcome file: {cfg.paths.outcome}, Outcome name: {cfg.outcome.type}')
    logger.info(f'Censor file: {cfg.paths.censor}, Censor name: {cfg.outcome.censor_type}')
    logger.info(f"Censoring {cfg.outcome.n_hours} hours after censor_outcome")
    train_dataset, val_dataset, outcomes = create_binary_outcome_datasets(cfg)

    dataset = ConcatIterableDataset([train_dataset, val_dataset])

    logger.info('Initializing model')
    model = load_model(BertEHRModel, cfg)


    forwarder = Forwarder( 
        model=model, 
        dataset=dataset, 
        run=run,
        **cfg.forwarder_args,
    )
    output_ = forwarder.forward_patients()
    print(output_.shape)

if __name__ == '__main__':
    main_encode()