import logging
import os
from os.path import join
from typing import Dict, List, Tuple, Union

import torch
from transformers import BertConfig

from corebehrt.common.config import Config, load_config
from corebehrt.common.utils import Data
from corebehrt.data.utils import Utilities

logger = logging.getLogger(__name__)  # Get the logger for this module

VOCABULARY_FILE = 'vocabulary.pt'
CHECKPOINT_FOLDER = 'checkpoints'
VAL_RATIO = 0.2


def load_checkpoint_and_epoch(cfg: Config)->Tuple:
    """Load checkpoint and epoch from config."""
    model_path = cfg.paths.get('model_path', None)
    checkpoint = ModelLoader(cfg).load_checkpoint() if model_path is not None else None
    if checkpoint is not None:
        epoch = checkpoint['epoch']
    else:
        epoch = Utilities.get_last_checkpoint_epoch(join(model_path, CHECKPOINT_FOLDER)) if model_path is not None else None
    return checkpoint, epoch

def load_model_cfg_from_checkpoint(cfg: Config, config_name: str)->bool:
    """If training from checkpoint, we need to get the old config"""
    model_path = cfg.paths.get('model_path', None)
    if model_path is not None: # if we are training from checkpoint, we need to load the old config
        old_cfg = load_config(join(cfg.paths.model_path, config_name))
        cfg.model = old_cfg.model
        return True
    return False

class FeaturesLoader:
    def __init__(self, cfg):
        self.path_cfg = cfg.paths
        self.cfg = cfg

    def load_tokenized_data(self, mode: str=None)->Data:
        """Load features for finetuning"""
        tokenized_dir = self.path_cfg.get('tokenized_dir', 'tokenized')
        tokenized_files = self.path_cfg.get('tokenized_file', f"tokenized_{mode}.pt")
        tokenized_pids_files = self.path_cfg.get('tokenized_pids', f"pids_{mode}.pt")
        
        # Ensure the files are in a list. We might want to load multiple files.
        tokenized_files = [tokenized_files] if isinstance(tokenized_files, str) else tokenized_files
        tokenized_pids_files = [tokenized_pids_files] if isinstance(tokenized_pids_files, str) else tokenized_pids_files
        assert len(tokenized_files) == len(tokenized_pids_files), "Number of tokenized files and pids files must be equal."

        tokenized_data_path = join(self.path_cfg.data_path, tokenized_dir)
        
        logger.info(f"Loading tokenized data from {tokenized_data_path}")
        features, pids = self.load_features_and_pids(tokenized_data_path, tokenized_files, tokenized_pids_files)
        
        logger.info("Loading vocabulary")
        vocabulary = self.load_vocabulary(tokenized_data_path)
        
        return Data(features, pids, vocabulary=vocabulary, mode=mode)
    
    @staticmethod
    def load_features_and_pids(tokenized_data_path: str, tokenized_files: list, tokenized_pids_files: list):
        """Load features and pids from multiple files."""
        features = {}
        pids = []
        for tokenized_file, tokenized_pids_file in zip(tokenized_files, tokenized_pids_files):
            features_temp = torch.load(join(tokenized_data_path, tokenized_file))
            pids_temp = torch.load(join(tokenized_data_path, tokenized_pids_file))
            # Concatenate features
            for key in features_temp.keys():
                features.setdefault(key, []).extend(features_temp[key])
            
            # Concatenate pids
            pids.extend(pids_temp)
        
        return features, pids

    def load_vocabulary(self, tokenized_data_path: str):
        """Load vocabulary from file."""
        vocabulary_file_path = join(tokenized_data_path, VOCABULARY_FILE)
        if not os.path.exists(vocabulary_file_path):
            vocabulary_file_path = join(self.path_cfg.data_path, VOCABULARY_FILE)
        
        return torch.load(vocabulary_file_path)

    def load_outcomes(self)->Tuple[dict, dict]:
        """Load outcomes and censoring timestamps from file. If no censoring timestamps provided, use outcomes as censoring timestamps."""
        logger.info(f'Load outcomes from {self.path_cfg.outcome}')
        censoring_timestamps_path = self.path_cfg.censor if self.path_cfg.get("censor", False) else self.path_cfg.outcome
        logger.info(f'Load censoring_timestamps from {censoring_timestamps_path}')
        outcomes = torch.load(self.path_cfg.outcome)
        censor_outcomes = torch.load(self.path_cfg.censor) if self.path_cfg.get('censor', False) else outcomes   
        return outcomes, censor_outcomes
    
    def load_finetune_data(self, path: str=None, mode: str=None)->Data:
        """Load features for finetuning"""
        path = self.path_cfg.finetune_features_path if path is None else path
        features = torch.load(join(path, f'features.pt'))
        outcomes = torch.load(join(path, f'outcomes.pt'))
        pids = torch.load(join(path, f'pids.pt'))
        vocabulary = torch.load(join(path, 'vocabulary.pt'))
        return Data(features, pids, outcomes, vocabulary=vocabulary, mode=mode)

class ModelLoader:
    def __init__(self, cfg: Config, model_path: str=None):
        """Load model from config and checkpoint."""
        self.cfg = cfg
        if model_path is not None:
            self.model_path = model_path
        elif self.cfg.paths.get('model_path', None) is not None:
            self.model_path = self.cfg.paths.model_path
        else:
            self.model_path = None
    
    def load_model(self, model_class, add_config:dict={}, checkpoint: dict=None, kwargs={}):
        """Load model from config and checkpoint. model_class is the class of the model to be loaded."""
        checkpoint = self.load_checkpoint() if checkpoint is None else checkpoint
        # Load the config from file
        config = BertConfig.from_pretrained(self.model_path) 
        config.update(add_config)
        model = model_class(config, **kwargs)
        
        return self.load_state_dict_into_model(model, checkpoint)
    
    def load_state_dict_into_model(self, model: torch.nn.Module, checkpoint: dict)->torch.nn.Module:
        """Load state dict into model. If embeddings are not loaded, raise an error."""
        load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        missing_keys = load_result.missing_keys

        if len([k for k in missing_keys if k.startswith('embeddings')])>0:
            pretrained_model_embeddings = model.embeddings.__class__.__name__
            raise ValueError(f"Embeddings not loaded. Ensure that model.behrt_embeddings is compatible with pretrained model embeddings {pretrained_model_embeddings}.")
        logger.warning("missing state dict keys: %s", missing_keys)
        return model

    def load_checkpoint(self)->dict:
        """Load checkpoint, if checkpoint epoch provided. Else load last checkpoint."""
        checkpoints_dir = join(self.model_path, CHECKPOINT_FOLDER)
        checkpoint_epoch = self.get_checkpoint_epoch()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint_path = join(checkpoints_dir,f'checkpoint_epoch{checkpoint_epoch}_end.pt')
        logger.info("Loading checkpoint from %s", checkpoint_path)
        return torch.load(checkpoint_path, map_location=device)
    
    def get_checkpoint_epoch(self)->int:
        """Get checkpoint epoch from config or return the last checkpoint_epoch for this model."""
        checkpoint_epoch = self.cfg.paths.get('checkpoint_epoch', None)
        if checkpoint_epoch is None:
            logger.info("No checkpoint provided. Loading last checkpoint.")
            checkpoint_epoch = Utilities.get_last_checkpoint_epoch(join(
                self.model_path, CHECKPOINT_FOLDER))
        return checkpoint_epoch

def load_exclude_pids(cfg)->List:
    """
    Loads pids from file
    Excluded pids
    """
    if cfg.get('exclude_pids', None) is None:
        return []
    return _load_pids(cfg.exclude_pids)

def load_assigned_pids(cfg)->Dict:
    """ Loads pids which should be assigned to certain splits."""
    if cfg.get('assigned_pids', None) is None:
        return {}
    assigned_pids = {}
    for split, files in cfg.assigned_pids.items():
        assigned_pids[split] = _load_pids(files)
    return assigned_pids

def _load_pids(files: Union[List, str])->List:
    """Loads pids from multiple files or one file. Doesn't preserve order."""
    if isinstance(files, str):
        return set(torch.load(files))    
    pids = set()
    for file in files:
        pids.update(set(torch.load(file)))
    return pids

def get_pids_file(split_dir: str, mode: str)->str:
    """Get pids file from predefined splits. 
    The file can be named pids_{mode}.pt or {mode}_pids.pt."""
    if os.path.exists(join(split_dir, f'{mode}_pids.pt')):
        return join(split_dir, f'{mode}_pids.pt')
    elif os.path.exists(join(split_dir, f'pids_{mode}.pt')):
        return join(split_dir, f'pids_{mode}.pt')
    else:
        raise ValueError(f'No pids file found for mode {mode} in {split_dir}')

def load_and_select_splits(split_dir: str, data: Data)->Tuple[Data, Data]:
    """Load and select pids from predefined splits"""
    logger.info("Load and select pids")
    train_pids = torch.load(get_pids_file(split_dir, 'train'))
    val_pids = torch.load(get_pids_file(split_dir, 'val'))
    train_data = data.select_data_subset_by_pids(train_pids, mode='train')
    val_data = data.select_data_subset_by_pids(val_pids, mode='val')
    return train_data, val_data