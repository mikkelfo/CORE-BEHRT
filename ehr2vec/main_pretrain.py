"""Pretrain BERT model on EHR data. Use config_template pretrain.yaml. Run main_data_pretrain.py first to create the dataset and vocabulary."""
import os
from os.path import join

from common.azure import setup_azure
from common.config import instantiate, load_config
from common.loader import DatasetPreparer
from common.setup import get_args, setup_run_folder, copy_data_config
from model.config import adjust_cfg_for_behrt
from model.model import BertEHRModel
from torch.optim import AdamW
from trainer.trainer import EHRTrainer
from transformers import BertConfig

CONFIG_NAME = 'pretrain.yaml'

args = get_args(CONFIG_NAME)
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main_train(config_path):
    cfg = load_config(config_path)
    run = None
    if cfg.env=='azure':
        run, mount_context = setup_azure(cfg.paths.run_name)
        cfg.paths.data_path = join(mount_context.mount_point, cfg.paths.data_path)

    logger, run_folder = setup_run_folder(cfg)
    copy_data_config(cfg, run_folder)

    train_dataset, val_dataset = DatasetPreparer(cfg).prepare_mlm_dataset()
    
    if cfg.model.get('behrt_embeddings', False):
        cfg = adjust_cfg_for_behrt(cfg)

    logger.info('Initializing model')
    model = BertEHRModel(
        BertConfig(
            **cfg.model,
            vocab_size=len(train_dataset.vocabulary),
        )
    )
    try:
        logger.warning('Compilation currently leads to torchdynamo error during training. Skip it')
        #model = torch.compile(model)
        #logger.info('Model compiled')
    except:
        logger.info('Model not compiled')

    logger.info('Initializing optimizer')
    optimizer = AdamW(
        model.parameters(),
        **cfg.optimizer
    )

    if cfg.scheduler:
        scheduler = instantiate(cfg.scheduler, **{'optimizer': optimizer})
        
    logger.info('Initialize trainer')
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
        run=run,
    )
    logger.info('Start training')
    trainer.train()
    if cfg.env == 'azure':
        try:
            from azure_run import file_dataset_save
            output_path = 'outputs' if not os.path.exists(join('outputs', 'retry_001')) else join('outputs', 'retry_001')
            file_dataset_save(local_path=join(output_path, cfg.paths.run_name), datastore_name = "workspaceblobstore",
                        remote_path = join("PHAIR", 'models', cfg.paths.type, cfg.paths.run_name))
            logger.info('Save model to blob')
        except:
            logger.warning('Could not save model to blob')
        mount_context.stop()
    logger.info("Done")

if __name__ == '__main__':
    main_train(config_path)