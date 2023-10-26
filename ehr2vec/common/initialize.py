import logging

from common.config import instantiate
from common.loader import Loader
from model.model import BertEHRModel
from torch.optim import AdamW
from transformers import BertConfig

logger = logging.getLogger(__name__)  # Get the logger for this module


class Initializer:
    """Initialize model, optimizer and scheduler."""
    def __init__(self, cfg, checkpoint=None):
        self.cfg = cfg
        self.checkpoint = checkpoint

    def initialize_pretrain_model(self, vocab_size:int=None):
        """Initialize model from checkpoint or from scratch."""
        if self.checkpoint:
            logger.info('Loading model from checkpoint')
            return Loader(self.cfg).load_model(BertEHRModel, checkpoint=self.checkpoint)
        else:
            logger.info('Initializing new model')
            assert vocab_size is not None, 'vocab_size must be provided when initializing from scratch'
            return BertEHRModel(
                    BertConfig(
                        **self.cfg.model,
                        vocab_size=vocab_size,
                    )
                )

    def initialize_optimizer(self, model):
        """Initialize optimizer from checkpoint or from scratch."""
        if self.checkpoint:
            logger.info('Loading AdamW optimizer from checkpoint')
            optimizer = AdamW(model.parameters(),)
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
        
        scheduler = instantiate(self.cfg.scheduler, **{'optimizer': optimizer})
        if not self.checkpoint:
            return scheduler
        
        logger.info('Loading scheduler from checkpoint')
        scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])
        return scheduler