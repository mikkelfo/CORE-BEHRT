import os
from os.path import join

import torch
from common.config import instantiate, load_config
from common.loader import DatasetPreparer, load_model
from common.setup import (add_pretrain_info_to_cfg, adjust_paths_for_finetune,
                          azure_finetune_setup, get_args, setup_run_folder)
from data.dataset import BinaryOutcomeDataset
from evaluation.utils import get_pos_weight, get_sampler
from model.model import BertForFineTuning
from torch.optim import AdamW
from trainer.trainer import EHRTrainer

args = get_args('finetune.yaml')
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main_finetune():
    cfg = load_config(config_path)
    model_path = cfg.paths.model_path
    cfg = adjust_paths_for_finetune(cfg)
    cfg, run, mount_context = azure_finetune_setup(cfg)
    cfg = add_pretrain_info_to_cfg(cfg)
    logger = setup_run_folder(cfg)
    cfg.save_to_yaml(join(cfg.paths.output_path, cfg.paths.run_name, 'finetune_config.yaml'))
    dataset_preparer = DatasetPreparer(cfg)
    data = dataset_preparer.prepare_finetune_features(
        original_behrt = cfg.model.get('behrt_embeddings', False)
    )
    
    train_data, val_data = data.split(cfg.data.get('val_split', 0.2))
    torch.save(train_data.pids, join(cfg.paths.output_path, cfg.paths.run_name, 'train_pids.pt'))
    torch.save(val_data.pids, join(cfg.paths.output_path, cfg.paths.run_name, 'val_pids.pt'))
    
    dataset_preparer.save_patient_nums(train_data, val_data)
    train_dataset = BinaryOutcomeDataset(train_data.features, train_data.outcomes)
    val_dataset = BinaryOutcomeDataset(val_data.features, val_data.outcomes)

    logger.info('Initializing model')
    model = load_model(BertForFineTuning, cfg, 
                       {'pos_weight':get_pos_weight(cfg, train_dataset.outcomes),
                        'embedding':'original_behrt' if cfg.model.get('behrt_embeddings', False) else None,
                        'pool_type': cfg.model.get('pool_type', 'mean')})

    try:
        logger.warning('Compilation currently leads to torchdynamo error during training. Skip it')
        #model = torch.compile(model)
        #logger.info('Model compiled')
    except:
        logger.info('Model not compiled')    
    optimizer = AdamW(
        model.parameters(),
        **cfg.optimizer
    )

    sampler = get_sampler(cfg, train_dataset, train_dataset.outcomes)
    if sampler:
        cfg.trainer_args.shuffle = False

    if cfg.scheduler:
        cfg.scheduler.optimizer = optimizer
        scheduler = instantiate(cfg.scheduler)

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
        accumulate_logits=True
    )
    trainer.train()
    if cfg.env=='azure':
        try:
            from azure_run import file_dataset_save
            file_dataset_save(local_path=join('outputs', cfg.paths.run_name), datastore_name = "workspaceblobstore",
                        remote_path = join("PHAIR", model_path, cfg.paths.run_name))
            logger.info("Saved to Azure Blob Storage")
        except:
            logger.warning('Could not save to Azure Blob Storage')
        mount_context.stop()
    logger.info('Done')

if __name__ == '__main__':
    main_finetune()