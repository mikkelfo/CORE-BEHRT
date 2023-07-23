"""
This script is used to embed concepts into a vector space.
"""

import json
import os
from os.path import join, split

import torch
import typer
from common import azure, io
from common.config import load_config
from common.setup import prepare_embedding_directory
from evaluation import embeddings, visualization
from model.model import BertEHRModel
from transformers import BertConfig

from data import dataset

config_path = join('configs', 'embed_concepts.yaml')
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

def main(config_path):
    """Produce patient trajectories. Cut off at increasing number of codes."""
    cfg = load_config(config_path)
    if cfg.env=='azure':
        _, mount_context = azure.setup_azure(cfg.run_name)
        mount_dir = mount_context.mount_dir
        cfg.loader.data_dir = join(mount_dir, cfg.loader.data_dir) # specify paths here
    
    model_path = cfg.paths.model
    data_path = cfg.paths.data
    
    logger = prepare_embedding_directory(config_path, cfg)  
    
    model_dir = split(model_path)[0]
    with open(join(model_dir, 'config.json'), 'r') as f:
        cfg = json.load(f)['model_config']
    logger.info('Initialize Model')
    model = BertEHRModel(BertConfig(**{**cfg, **{'output_hidden_states':True}}))
    state_dict = torch.load(model_path)['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    
    # TODO: allow big datasets
    ds = dataset.BaseSmallDataset(torch.load(data_path))
    
    logger.info('Produce embeddings')
    data = embeddings.produce_concept_embeddings(model, ds, **cfg.embeddings)
    
    emb_save_path = join(model_dir, 'embeddings', f'concept_emb_{cfg.batch_size*cfg.stop_iter}pats.hdf5')
    logger.info(f"Save embeddings to {emb_save_path}")
    io.save_hdf5(data, emb_save_path)
    if cfg.project:
        logger.info('Project')
        data = visualization.project_embeddings(data, cfg)
        del data['concept_enc']
        df = pe.store_to_df(data, data_path)
        save_path = join(model_dir, f'concept_proj_{cfg.batch_size*cfg.stop_iter}pats.csv') 
        df.to_csv(save_path, index=False)
        logger.info(f'Embeddings stored to {save_path}')

        

if __name__ == "__main__":
    typer.run(main)