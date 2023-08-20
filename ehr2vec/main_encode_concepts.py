"""
This script is used to embed concepts into a vector space.
"""
import os
from os.path import join

import torch
from common.azure import setup_azure
from common.config import load_config
from common.loader import load_model, retrieve_outcomes, select_positives
from common.setup import get_args, prepare_encodings_directory
from common.utils import ConcatIterableDataset
from data.dataset import BaseEHRDataset
from evaluation.encodings import Forwarder
from evaluation.utils import validate_outcomes
from model.model import BertEHREncoder

args = get_args("encode_concepts.yaml")
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)

def main(config_path):
    """Produce patient trajectories. Cut off at increasing number of codes."""
    encodings_file_name = 'encodings.h5'
    cfg = load_config(config_path)
    run = None
    model_path = cfg.paths.model_path
    concepts_path = join(model_path, 'encodings', 'concepts')
    logger.info(f"Access data from {cfg.paths.data_path}")
    logger.info(f"Access outcomes from {cfg.paths.outcomes_path}")
    cfg.output_dir = concepts_path
    if cfg.env=='azure':
        run, mount_context = setup_azure(cfg.paths.run_name)
        cfg.paths.data_path = join(mount_context.mount_point, cfg.paths.data_path)
        cfg.paths.model_path = join(mount_context.mount_point, cfg.paths.model_path)
        cfg.paths.outcomes_path = join(mount_context.mount_point, cfg.paths.outcomes_path)
        cfg.output_dir = 'outputs'
    logger = prepare_encodings_directory(config_path, cfg)
    
    logger.info('Initializing model')
    model = load_model(BertEHREncoder, cfg)
    dataset = get_dataset(cfg)
    
    logger.info('Produce embeddings')
    forwarder = Forwarder(model=model, 
            dataset=dataset, 
            run=run,
            logger=logger,
            output_path=join(cfg.output_dir, encodings_file_name),
            **cfg.forwarder_args,)
    forwarder.encode_concepts()
    
    if cfg.env=='azure':
        logger.info('Saving to blob storage')
        from azure_run import file_dataset_save
        file_dataset_save(local_path='outputs', datastore_name = "workspaceblobstore",
                    remote_path = join("PHAIR", concepts_path))
        mount_context.stop()
    logger.info('Done')


def get_dataset(cfg):
    all_outcomes = torch.load(cfg.paths.outcomes_path)
    validate_outcomes(all_outcomes, cfg)
    pids = []
    for i, outcome in enumerate(cfg.outcomes):
        cfg.outcome = cfg.outcomes[outcome]
        outcomes, censor_outcomes, pids_outcome = retrieve_outcomes(all_outcomes, cfg)
        _, _, pids_outcome = select_positives(outcomes, censor_outcomes, pids)
        pids.extend(pids_outcome)
    pids = list(set(pids))
    train_dataset = BaseEHRDataset(cfg.paths.data_path, 'train', pids)
    val_dataset = BaseEHRDataset(cfg.paths.data_path, 'val',  pids=pids)
    if isinstance(train_dataset, type(None)) or isinstance(val_dataset, type(None)):
        return train_dataset if train_dataset is not None else val_dataset
    else:
        return ConcatIterableDataset([train_dataset, val_dataset])
        

if __name__ == "__main__":
    main(config_path)