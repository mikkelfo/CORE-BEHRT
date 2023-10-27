import logging
from typing import Optional, Tuple

import torch
from common.config import Config, instantiate
from common.loader import ModelLoader
from evaluation.utils import get_pos_weight, get_sampler
from model.model import (BertEHRModel, BertForFineTuning,
                         HierarchicalBertForPretraining)
from torch.optim import AdamW
from torch.utils.data import Sampler
from transformers import BertConfig

logger = logging.getLogger(__name__)  # Get the logger for this module


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
            return model
        else:
            logger.info('Initializing new model')
            vocab_size = len(train_dataset.vocabulary)
            return BertEHRModel(
                    BertConfig(
                        **self.cfg.model,
                        vocab_size=vocab_size,
                    )
                )
    
    def initialize_finetune_model(self, train_dataset):
        if self.checkpoint:
            logger.info('Loading model from checkpoint')
            return self.loader.load_model(
                BertForFineTuning, 
                checkpoint=self.checkpoint, 
                add_config={'pos_weight':get_pos_weight(self.cfg, train_dataset.outcomes),
                            'embedding':'original_behrt' if self.cfg.model.get('behrt_embeddings', False) else None,
                            'pool_type': self.cfg.model.get('pool_type', 'mean')})
        else:
            raise NotImplementedError('Fine-tuning from scratch is not supported.')
            logger.info('Initializing new model')
            return BertForFineTuning(
                BertConfig(
                    **self.cfg.model,
                    pos_weight=get_pos_weight(self.cfg, train_dataset.outcomes),
                    embedding='original_behrt' if self.cfg.model.get('behrt_embeddings', False) else None,
                    pool_type=self.cfg.model.get('pool_type', 'mean')),)
        
    def initialize_hierachical_pretrain_model(self, train_dataset):
        if self.checkpoint:
            logger.info('Loading model from checkpoint')
            model = self.loader.load_model(HierarchicalBertForPretraining,
                                                 checkpoint=self.checkpoint,
                                                 kwargs={'tree_matrix':train_dataset.tree_matrix})
        else:
            bertconfig = BertConfig(leaf_size=len(train_dataset.leaf_counts), 
                                vocab_size=len(train_dataset.vocabulary),
                                levels=train_dataset.levels,
                                **self.cfg.model)
            model = HierarchicalBertForPretraining(
                bertconfig, tree_matrix=train_dataset.tree_matrix)
        return model

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
        for state in optimizer_state_dic['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)