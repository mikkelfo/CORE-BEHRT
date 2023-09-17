import os
from os.path import join

import torch
from common.config import load_config
from common.loader import DatasetPreparer, load_model
from common.setup import (add_pretrain_info_to_cfg, adjust_paths_for_finetune,
                          azure_finetune_setup, get_args, setup_run_folder)
from evaluation.utils import get_pos_weight, get_sampler
from model.model import BertForFineTuning
from torch.optim import AdamW
from trainer.trainer import EHRTrainer

args = get_args('finetune.yaml')
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)


def main_finetune():
    cfg = load_config(config_path)

    cfg = adjust_paths_for_finetune(cfg)
    cfg, run, mount_context = azure_finetune_setup(cfg)
    cfg = add_pretrain_info_to_cfg(cfg)
    logger = setup_run_folder(cfg)
    cfg.save_to_yaml(join(cfg.paths.output_path, cfg.paths.run_name, 'finetune_config.yaml'))
    
    train_dataset, val_dataset = DatasetPreparer(cfg).prepare_finetune_dataset()
   
    logger.info('Initializing model')
    model = load_model(BertForFineTuning, cfg, 
                       {'pos_weight':get_pos_weight(cfg, train_dataset.outcomes),
                        'pool_type': cfg.model.get('pool_type', 'mean')})
    try:
        model = torch.compile(model)
        logger.info('Model compiled')
    except:
        logger.info('Model not compiled')    
    optimizer = AdamW(
        model.parameters(),
        **cfg.optimizer
    )

    trainer = EHRTrainer( 
        model=model, 
        optimizer=optimizer,
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        args=cfg.trainer_args,
        metrics=cfg.metrics,
        sampler=get_sampler(cfg, train_dataset, train_dataset.outcomes, logger),
        cfg=cfg,
        run=run,
        logger=logger,
    )
    trainer.train()
    if cfg.env=='azure':
        from azure_run import file_dataset_save
        file_dataset_save(local_path=join('outputs', cfg.paths.run_name), datastore_name = "workspaceblobstore",
                    remote_path = join("PHAIR", cfg.paths.model_path, cfg.paths.run_name))
        mount_context.stop()
    logger.info('Done')


if __name__ == '__main__':
    main_finetune()