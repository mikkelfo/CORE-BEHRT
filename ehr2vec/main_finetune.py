import os
from os.path import join

import torch
from common.azure import save_to_blobstore
from common.initialize import Initializer, ModelManager
from common.setup import (DirectoryPreparer, copy_data_config,
                          copy_pretrain_config, get_args)
from data.dataset import BinaryOutcomeDataset
from data.prepare_data import DatasetPreparer
from trainer.trainer import EHRTrainer

CONFIG_NAME = 'finetune.yaml'
BLOBSTORE='PHAIR'

args = get_args(CONFIG_NAME)
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main_finetune():
    cfg, run, mount_context, pretrain_model_path = Initializer.initialize_configuration_finetune(config_path, 
                                                                                                 dataset_name=BLOBSTORE)
    
    logger, run_folder = DirectoryPreparer.setup_run_folder(cfg)

    copy_data_config(cfg, run_folder)
    copy_pretrain_config(cfg, run_folder)
    cfg.save_to_yaml(join(run_folder, 'finetune_config.yaml'))
    dataset_preparer = DatasetPreparer(cfg)
    data = dataset_preparer.prepare_finetune_features()
    
    train_data, val_data = data.split(cfg.data.get('val_split', 0.2))
    torch.save(train_data.pids, join(run_folder, 'train_pids.pt'))
    torch.save(val_data.pids, join(run_folder, 'val_pids.pt'))
    
    dataset_preparer.saver.save_patient_nums(train_data, val_data)
    train_dataset = BinaryOutcomeDataset(train_data.features, train_data.outcomes)
    val_dataset = BinaryOutcomeDataset(val_data.features, val_data.outcomes)

    modelmanager = ModelManager(cfg)
    checkpoint = modelmanager.load_checkpoint() 
    modelmanager.load_model_config() # check whether cfg is changed after that.
    model = modelmanager.initialize_finetune_model(checkpoint, train_dataset)
    
    optimizer, sampler, scheduler, cfg = modelmanager.initialize_training_components(model, train_dataset)
    epoch = modelmanager.get_epoch()


    trainer = EHRTrainer( 
        model=model, 
        optimizer=optimizer,
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        args=cfg.trainer_args,
        metrics=cfg.metrics,
        sampler=sampler,
        scheduler=scheduler,
        cfg=cfg,
        run=run,
        logger=logger,
        accumulate_logits=True,
        last_epoch=epoch
    )
    trainer.train()
    if cfg.env=='azure':
        save_path = pretrain_model_path if cfg.paths.get("save_folder_path", None) is None else cfg.paths.save_folder_path
        save_to_blobstore(cfg.paths.run_name,
                          join(BLOBSTORE, save_path, cfg.paths.run_name))
        mount_context.stop()
    logger.info('Done')

if __name__ == '__main__':
    main_finetune()