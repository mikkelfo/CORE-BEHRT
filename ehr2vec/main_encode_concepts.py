"""
This script is used to embed concepts into a vector space.
"""

import json
import os
from os.path import join

import torch

from common.azure import setup_azure
from common.io import save_hdf5

from common.config import load_config
from common.setup import prepare_embedding_directory, get_args
from evaluation.encodings import Forwarder
from evaluation.visualization import project_embeddings
from model.model import BertEHRModel
from transformers import BertConfig

from data.dataset import BaseEHRDataset 

args = get_args("encode_concepts.yaml")
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)

def main(config_path):
    """Produce patient trajectories. Cut off at increasing number of codes."""
    cfg = load_config(config_path)
    if cfg.env=='azure':
        _, mount_context = setup_azure(cfg.run_name)
        mount_dir = mount_context.mount_dir
        cfg.loader.data_dir = join(mount_dir, cfg.loader.data_dir) # specify paths here
        cfg.output_dir = join(mount_dir, cfg.output_dir)
    
    
    logger = prepare_embedding_directory(config_path, cfg)  
    
    with open(join(cfg.loader.model_dir, 'config.json'), 'r') as f:
        model_cfg = json.load(f)
    logger.info('Initialize Model')
    
    model = BertEHRModel(BertConfig(**{**model_cfg, **{'output_hidden_states':True}}))
    model_path = join(cfg.loader.model_dir, 'checkpoints', f'checkpoint_epoch{cfg.loader.checkpoint_num}_end.pt')
    state_dict = torch.load(model_path)['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    
    ds = BaseEHRDataset(cfg.loader.data_dir, mode='val')
    
    logger.info('Produce embeddings')
    forwarder = Forwarder(model, ds, **cfg.embeddings )
    data = forwarder.produce_concept_embeddings()
    
    emb_save_path = join(cfg.loader, 'embeddings', f'concept_emb_{cfg.batch_size*cfg.stop_iter}pats.hdf5')
    logger.info(f"Save embeddings to {emb_save_path}")
    save_hdf5(data, emb_save_path)
    if cfg.project:
        logger.info('Project')
        data = project_embeddings(data, cfg)
        del data['concept_enc']
        df = forwarder.store_to_df(data, cfg.output_dir)
        save_path = join(cfg.output_dir, f'concept_proj_{cfg.batch_size*cfg.stop_iter}pats.csv') 
        df.to_csv(save_path, index=False)
        logger.info(f'Embeddings stored to {save_path}')

        

if __name__ == "__main__":
    main(config_path)