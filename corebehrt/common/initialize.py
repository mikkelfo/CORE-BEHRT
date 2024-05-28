import logging
import os
from os.path import join
from typing import Optional, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import Sampler
from transformers import BertConfig

from corebehrt.common.azure import AzurePathContext
from corebehrt.common.config import Config, instantiate, load_config
from corebehrt.common.loader import ModelLoader, load_model_cfg_from_checkpoint
from corebehrt.common.setup import DirectoryPreparer
from corebehrt.data.utils import Utilities
from corebehrt.evaluation.utils import get_sampler
from corebehrt.model.model import BertEHRModel, BertForFineTuning

logger = logging.getLogger(__name__)  # Get the logger for this module
CHECKPOINTS_DIR = "checkpoints"

class Initializer:
    """Initialize model, optimizer and scheduler."""
    def __init__(self, cfg:Config, checkpoint:dict=None, model_path:str=None):
        self.cfg = cfg
        self.checkpoint = checkpoint
        if checkpoint:
            self.loader = ModelLoader(cfg, model_path)
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    def initialize_pretrain_model(self, train_dataset):
        """Initialize model from checkpoint or from scratch."""
        if self.checkpoint:
            logger.info('Loading model from checkpoint')
            model = self.loader.load_model(BertEHRModel, checkpoint=self.checkpoint)
            model.to(self.device)
        else:
            logger.info('Initializing new model')
            vocab_size = len(train_dataset.vocabulary)
            model = BertEHRModel(BertConfig( **self.cfg.model, vocab_size=vocab_size,))
        return model
        
    def initialize_finetune_model(self, train_dataset):
        if self.checkpoint:
            logger.info('Loading model from checkpoint')
            add_config = {**self.cfg.model}
            model = self.loader.load_model(
                BertForFineTuning, 
                checkpoint=self.checkpoint, 
                add_config=add_config,
                )
            model.to(self.device) 
            return model
        else:
            raise NotImplementedError('Fine-tuning from scratch is not supported.')

    def initialize_optimizer(self, model):
        """Initialize optimizer from checkpoint or from scratch."""
        if self.checkpoint:
            logger.info('Loading AdamW optimizer from checkpoint')
            optimizer = AdamW(model.parameters(),)
            self.optimizer_state_dic_to_device(self.checkpoint['optimizer_state_dict'])
            optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            return optimizer
        else:
            logger.info('Initializing new AdamW optimizer')
            return AdamW(
                model.parameters(),
                **self.cfg.optimizer
            )

    def initialize_scheduler(self, optimizer):
        """Initialize scheduler from checkpoint or from scratch."""
        if not self.cfg.get('scheduler', None):
            return None
        logger.info('Initializing new scheduler')
        scheduler = instantiate(self.cfg.scheduler, **{'optimizer': optimizer})

        if not self.checkpoint:
            return scheduler
        
        logger.info('Loading scheduler_state_dict from checkpoint')
        scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])
        return scheduler
    
    def initialize_sampler(self, train_dataset)->Tuple[Optional[Sampler], Config]:
        """Initialize sampler and modify cfg."""
        sampler = get_sampler(self.cfg, train_dataset, train_dataset.outcomes)
        if sampler:
            self.cfg.trainer_args.shuffle = False
        return sampler, self.cfg

    def optimizer_state_dic_to_device(self, optimizer_state_dic):
        """Move optimizer state dict to device."""
        for state in optimizer_state_dic['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
    @staticmethod
    def initialize_configuration_finetune(config_path:str, dataset_name:str='PHAIR')->Tuple[Config, str, str, str]:
        """Load and adjust the configuration."""
        cfg = load_config(config_path)
        pretrain_model_path = cfg.paths.get('pretrain_model_path', None)
        cfg = DirectoryPreparer.adjust_paths_for_finetune(cfg)
        azure_context = AzurePathContext(cfg, dataset_name=dataset_name)
        cfg, run, mount_context = azure_context.azure_finetune_setup()
        cfg = azure_context.add_pretrain_info_to_cfg()
        return cfg, run, mount_context, pretrain_model_path
    

class ModelManager:
    """Manager for initializing model, optimizer and scheduler."""
    def __init__(self, cfg, fold:int=None, model_path:str=None):
        self.cfg = cfg
        self.fold = fold
        self.model_path = model_path if model_path is not None else cfg.paths.get('model_path', None)
        self.pretrain_model_path = cfg.paths.get('pretrain_model_path', None)
        self.check_arguments()
        if self.fold is not None:
            if self.model_path is not None:
                self.model_path = join(self.model_path, f'fold_{fold}')
                if not os.path.exists(self.model_path):
                    logger.warning(f'Could not find model path {self.model_path}. Start from scratch')
                    self.model_path = None
        self.checkpoint_model_path = self.model_path if self.model_path is not None else self.pretrain_model_path
        logger.info(f'Checkpoint model path: {self.checkpoint_model_path}')
        self.initializer = None

    def check_arguments(self):
        if isinstance(self.pretrain_model_path, type(None)) and isinstance(self.model_path, type(None)):
            raise ValueError('Either pretrain_model_path or model_path must be provided.')
    
    def load_checkpoint(self):
        return ModelLoader(self.cfg, self.checkpoint_model_path).load_checkpoint()
    
    def load_model_config(self):
        if self.model_path:
            load_model_cfg_from_checkpoint(self.cfg, 'finetune_config.yaml')
    
    def initialize_finetune_model(self,  checkpoint, train_dataset):
        logger.info('Initializing model')
        self.initializer = Initializer(self.cfg, checkpoint=checkpoint, model_path=self.checkpoint_model_path)
        model = self.initializer.initialize_finetune_model(train_dataset) 
        return model

    def initialize_training_components(self, model, train_dataset):
        """Initialize training components. If no model_path provided, optimizer and scheduler are initialized from scratch."""
        if self.model_path is None:
            logger.info('Initializing optimizer and scheduler from scratch')
            self.initializer.checkpoint = None
        optimizer = self.initializer.initialize_optimizer(model)
        sampler, cfg = self.initializer.initialize_sampler(train_dataset)
        scheduler = self.initializer.initialize_scheduler(optimizer)
        return optimizer, sampler, scheduler, cfg
    
    def get_epoch(self):
        """Get epoch from model_path."""
        if self.model_path is None:
            return 0
        else:
            return Utilities.get_last_checkpoint_epoch(join(self.model_path, CHECKPOINTS_DIR))
    
    