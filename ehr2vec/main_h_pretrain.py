"""Pretrain hierarchical model on EHR data. Use config_template h_pretrain.yaml. Run setup_hierarchical.py first to create the vocabulary and tree."""
import os
from os.path import join

import torch
from common.azure import setup_azure
from common.config import instantiate, load_config
from common.loader import DatasetPreparer
from common.setup import get_args, setup_run_folder
from model.config import adjust_cfg_for_behrt
from model.model import HierarchicalBertForPretraining
from torch.optim import AdamW
from trainer.trainer import EHRTrainer
from transformers import BertConfig

args = get_args("h_pretrain.yaml")
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main_train(config_path):
    cfg = load_config(config_path)
    run = None
    if cfg.env=='azure':
        run, mount_context = setup_azure(cfg.paths.run_name)
        cfg.paths.data_path = join(mount_context.mount_point, cfg.paths.data_path)
    
    logger = setup_run_folder(cfg)
    
    logger.info(f"Loading data from {cfg.paths.data_path}")
    train_dataset, val_dataset = DatasetPreparer(cfg).prepare_hmlm_dataset(
        original_behrt = cfg.model.get('behrt_embeddings', False)
    )
    torch.save(train_dataset.target_mapping, join(cfg.paths.output_path, cfg.paths.run_name, 'target_mapping.pt'))
    logger.info("Setup model")
    if cfg.model.get('behrt_embeddings', False):
        cfg = adjust_cfg_for_behrt(cfg)
    bertconfig = BertConfig(leaf_size=len(train_dataset.leaf_counts), 
                            vocab_size=len(train_dataset.vocabulary),
                            levels=train_dataset.levels,
                            **cfg.model)
    model = HierarchicalBertForPretraining(
        bertconfig, tree_matrix=train_dataset.tree_matrix)

    try:
        logger.warning('Compilation currently leads to torchdynamo error during training. Skip it')
        #model = torch.compile(model)
        #logger.info('Model compiled')
    except:
        logger.info('Model not compiled')
        
    logger.info("Setup optimizer")
    optimizer = AdamW(
        model.parameters(),
        **cfg.optimizer
    )
    if cfg.scheduler:
        scheduler = instantiate(cfg.scheduler, **{'optimizer': optimizer})
        
    logger.info("Setup trainer")
    trainer = EHRTrainer( 
        model=model, 
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        args=cfg.trainer_args,
        metrics=cfg.metrics,
        cfg=cfg,
        logger=logger,
        run=run
    )
    logger.info("Start training")
    trainer.train()
    if cfg.env == 'azure':
        from azure_run import file_dataset_save
        file_dataset_save(local_path=join('outputs', cfg.paths.run_name), datastore_name = "workspaceblobstore",
                    remote_path = join("PHAIR", 'models', cfg.paths.type, cfg.paths.run_name))
        logger.info("Saved model to Azure")
        mount_context.stop()
    logger.info("Done")

if __name__ == '__main__':
    main_train(config_path)