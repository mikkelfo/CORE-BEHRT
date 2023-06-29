from os.path import join

from common.config import load_config
from common.loader import create_hierarchical_dataset
from common.setup import setup_run_folder
from model.model import HierarchicalBertForPretraining
from torch.optim import AdamW
from trainer.trainer import EHRTrainer
from transformers import BertConfig
import torch
# from azure_run.run import Run
# from azure_run import datastore
# from azureml.core import Dataset


# run = Run
run = None
# run.name(f"Pretrain hierarchical diagnosis medication")

# ds = datastore("workspaceblobstore")
# dataset = Dataset.File.from_files(path=(ds, 'PHAIR'))
config_path = 'configs/h_pretrain.yaml'

def load_hierarchical_data(cfg):
    """Load hierarchical data from disk"""
    vocab = torch.load(join(cfg.paths.data_path, 'vocabulary.pt'))
    tree = torch.load(join(cfg.paths.data_path, 'hierarchical', 'tree.pt'))
    train_dataset, val_dataset = create_hierarchical_dataset(cfg)

    return vocab, tree, train_dataset, val_dataset


def main_train(config_path):
    cfg = load_config(config_path)
    
    # mount_context = dataset.mount()
    # mount_context.start()  # this will mount the file streams
    
    # cfg.paths.data_path = join(mount_context.mount_point, cfg.paths.data_path)
    cfg.paths.output_path = join('outputs', cfg.paths.output_path)
    
    logger = setup_run_folder(cfg)
    
    logger.info(f'Loading data from {cfg.paths.data_path}')
    vocab, tree, train_dataset, val_dataset = load_hierarchical_data(cfg)
  
    logger.info("Setup model")
    bertconfig = BertConfig(leaf_size=len(train_dataset.leaf_counts), vocab_size=len(vocab), **cfg.model)
    model = HierarchicalBertForPretraining(bertconfig, tree=tree)

    logger.info("Setup optimizer")
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        eps=cfg.optimizer.epsilon,
    )

    logger.info("Setup trainer")
    trainer = EHRTrainer( 
        model=model, 
        optimizer=optimizer,
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        args=cfg.trainer_args,
        metrics=cfg.metrics,
        cfg=cfg,
        logger=logger,
    )
    breakpoint()
    trainer.train()


if __name__ == '__main__':
    main_train(config_path)