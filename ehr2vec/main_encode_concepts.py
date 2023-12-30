"""
This script is used to embed concepts into a vector space.
!Currently, this script is not used in the paper. It is a first attempt to evaluate the RF model.
"""
import os
import pandas as pd
from os.path import join

import torch
from common.azure import AzurePathContext, save_to_blobstore
from common.config import load_config
from common.loader import ModelLoader
from common.setup import get_args, DirectoryPreparer 
from common.utils import ConcatIterableDataset
from common.io import ConceptHDF5Writer
from data.dataset import BaseEHRDataset
from evaluation.encodings import Forwarder
from evaluation.utils import validate_outcomes
from model.model import BertEHREncoder

BLOBSTORE = 'PHAIR'
CONFIG_PATH = 'encode_concepts.yaml'

args = get_args(CONFIG_PATH)
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)

def main(config_path):
    """Encode selected concepts associated with patients with certain outcomes."""
    encodings_file_name = 'encodings.h5'
    cfg = load_config(config_path)
    run = None
    model_path = cfg.paths.model_path
    concepts_path = join(model_path, 'encodings', 'concepts')
    cfg.output_dir = join(concepts_path, cfg.paths.run_name)
    cfg, run, mount_context = AzurePathContext(cfg, dataset_name=BLOBSTORE).azure_encode_setup()
    
    logger = DirectoryPreparer(config_path).prepare_encodings_directory(cfg)
    logger.info('Initializing model')
    model = ModelLoader(cfg).load_model(BertEHREncoder)
    logger.info(f"Access data from {cfg.paths.data_path}")
    logger.info(f"Access outcomes from {cfg.paths.outcomes_path}")
    dataset = get_dataset(cfg)
    logger.info('Produce embeddings')
    forwarder = Forwarder(model=model, 
            dataset=dataset, 
            run=run,
            logger=logger,
            writer=ConceptHDF5Writer(join(cfg.output_dir, encodings_file_name)),
            **cfg.forwarder_args,)
    forwarder.encode_concepts(cfg)
    
    if cfg.env=='azure':
        save_to_blobstore(cfg.paths.run_name,
                          join(BLOBSTORE, concepts_path, cfg.paths.run_name))
        mount_context.stop()
    logger.info('Done')

def get_dataset(cfg):
    all_outcomes = torch.load(cfg.paths.outcomes_path)
    validate_outcomes(all_outcomes, cfg)
    pids = retrieve_relevant_pids(all_outcomes, cfg)
    train_dataset = BaseEHRDataset(cfg.paths.data_path, 'train', pids)
    val_dataset = BaseEHRDataset(cfg.paths.data_path, 'val',  pids=pids)
    if isinstance(train_dataset, type(None)) or isinstance(val_dataset, type(None)):
        return train_dataset if train_dataset is not None else val_dataset
    else:
        return ConcatIterableDataset([train_dataset, val_dataset])
        
def retrieve_relevant_pids(all_outcomes, cfg):
    pids = []
    all_pids = all_outcomes['PID']
    for i, outcome in enumerate(cfg.outcomes):
        cfg.outcome = cfg.outcomes[outcome]
        outcomes = all_outcomes.get(cfg.outcome.type, [None]*len(all_outcomes['PID']))
        select_indices = [i for i, outcome in enumerate(outcomes) if pd.notna(outcome)]
        pos_pids = [all_pids[i] for i in select_indices]
        pids.extend(pos_pids)
    return list(set(pids))


    
if __name__ == "__main__":
    main(config_path)