import os
from os.path import join, split

import torch
from common.azure import setup_azure
from common.config import load_config
from common.loader import DatasetPreparer, load_model
from common.setup import get_args, setup_run_folder
from evaluation.utils import get_pos_weight, get_sampler
from model.model import BertForFineTuning
from torch.optim import AdamW
from trainer.trainer import EHRTrainer

args = get_args('finetune.yaml')
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)


def main_finetune():
    cfg = load_config(config_path)
    run = None

    if cfg.env=='azure':
        run, mount_context = setup_azure(cfg.paths.run_name)
        cfg.paths.data_path = join(mount_context.mount_point, cfg.paths.data_path)
        cfg.paths.model_path = join(mount_context.mount_point, cfg.paths.model_path)
        cfg.paths.output_path = join("outputs")
    logger = setup_run_folder(cfg)
    
    # Get settings from pretraining
    pretrain_cfg = load_config(join(cfg.paths.model_path, 'pretrain_config.yaml'))
    cfg.data.remove_background = pretrain_cfg.data.remove_background
    cfg.paths.tokenized_dir = pretrain_cfg.paths.tokenized_dir
    
    train_dataset, val_dataset = DatasetPreparer(cfg).prepare_finetune_dataset_for_behrt()
   
    logger.info('Initializing model')
    model = load_model(BertForFineTuning, cfg, add_config = 
                       {'pos_weight':get_pos_weight(cfg, train_dataset.outcomes),
                        'embedding':'original_behrt',
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
                    remote_path = join("PHAIR", 'models', split(cfg.paths.model_path)[-1], "finetune_"+cfg.run_name))
        mount_context.stop()
    logger.info('Done')






if __name__ == '__main__':
    main_finetune()