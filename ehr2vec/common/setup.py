import argparse
import logging
import os
import uuid
from os.path import join
from shutil import copyfile
from typing import Tuple

from common.azure import setup_azure
from common.config import Config, load_config
from common.loader import ModelLoader, Utilities

OUTPUTS_DIR = "outputs"
CHECKPOINTS_DIR = "checkpoints"

def get_args(default_config_name, default_run_name=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=join('configs', default_config_name))
    parser.add_argument('--run_name', type=str, default=default_run_name if default_run_name else default_config_name.split('.')[0])
    return parser.parse_args()

def setup_logger(dir: str, log_file: str = 'info.log'):
    """Sets up the logger."""
    logging.basicConfig(filename=join(dir, log_file), level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def create_directory_and_copy_config(config_path: str, output_dir: str, new_config_name: str)->logging.Logger:
    """Creates output directory and copies config file"""
    os.makedirs(output_dir, exist_ok=True)
    destination = join(output_dir, new_config_name)
    copyfile(config_path, destination)
    return setup_logger(output_dir)

def prepare_directory(config_path: str, cfg: Config):
    """Creates output directory and copies config file"""
    logger = create_directory_and_copy_config(config_path, cfg.output_dir, 'data_config.yaml')
    os.makedirs(join(cfg.output_dir, 'features'), exist_ok=True)
    os.makedirs(join(cfg.output_dir, cfg.tokenized_dir_name), exist_ok=True)
    copyfile(config_path, join(cfg.output_dir, cfg.tokenized_dir_name, 'data_config.yaml'))
    return logger

def prepare_directory_outcomes(config_path: str, outcome_dir: str, outcomes_name: str):
    """Creates output directory for outcomes and copies config file"""
    return create_directory_and_copy_config(config_path, outcome_dir, f'outcome_{outcomes_name}_config.yaml')

def prepare_directory_hierarchical(config_path: str, out_dir: str, hierarchical_name: str = 'hierarchical'):
    """Creates hierarchical directory and copies config file"""
    hierarchical_dir = join(out_dir, hierarchical_name)
    return create_directory_and_copy_config(config_path, hierarchical_dir, 'h_setup.yaml')

def prepare_embedding_directory(config_path: str, cfg: Config):
    """Creates output directory and copies config file"""
    return create_directory_and_copy_config(config_path, cfg.output_dir, 'emb_config.yaml')

def prepare_encodings_directory(config_path: str, cfg: Config):
    """Creates output directory and copies config file"""
    return create_directory_and_copy_config(config_path, cfg.output_dir, 'encodings_config.yaml')

def setup_run_folder(cfg:Config, run_folder: str=None)->Tuple[logging.Logger, str]:
    """Creates a run folder and checkpoints folder inside it. Returns logger and run folder path."""
    # Generate unique run_name if not provided
    run_name = cfg.paths.run_name if hasattr(cfg.paths, 'run_name') else uuid.uuid4().hex
    if run_folder is None:
        run_folder = join(cfg.paths.output_path, run_name)

    os.makedirs(run_folder, exist_ok=True)
    os.makedirs(join(run_folder, CHECKPOINTS_DIR), exist_ok=True)
    logger = setup_logger(run_folder)
    logger.info(f'Run folder: {run_folder}')
    return logger, run_folder

def adjust_paths_for_finetune(cfg: Config)->Config:
    """
    Adjusts the following paths in the configuration for the finetune environment:
    - output_path: set to pretrain_model_path
    - run_name: constructed according to setting
    """
    cfg.paths.output_path = cfg.paths.pretrain_model_path
    cfg.paths.run_name = construct_finetune_model_dir_name(cfg)
    return cfg

def construct_finetune_model_dir_name(cfg: Config)->str:
    """Constructs the name of the finetune model directory. Based on the outcome type, the censor type, and the number of hours pre- or post- outcome."""
    days = True if abs(cfg.outcome.n_hours)>48 else False
    window = int(abs(cfg.outcome.n_hours/24)) if days else abs(cfg.outcome.n_hours)
    days_hours = 'days' if days else 'hours'
    pre_post = 'pre' if cfg.outcome.n_hours<0 else 'post'
    return f"finetune_{cfg.outcome.type}_censored_{window}_{days_hours}_{pre_post}_{cfg.outcome.censor_type}_{cfg.paths.run_name}"

def split_path(path_str: str) -> list:
    """Split path into its components."""
    directories = []
    while path_str:
        path_str, directory = os.path.split(path_str)
        # If we've reached the root directory
        if directory:
            directories.append(directory)
        elif path_str:
            break
    return directories[::-1]  # Reverse the list to get original order

def copy_data_config(cfg: Config, run_folder: str)->None:
    """
    Copy data_config.yaml to run folder.
    By default copy from tokenized folder, if not available, copy from data folder.
    """
    tokenized_dir_name = cfg.paths.get('tokenized_dir', 'tokenized')
    try:
        copyfile(join(cfg.paths.data_path, tokenized_dir_name, 'data_config.yaml'), join(run_folder, 'data_config.yaml'))
    except:
        copyfile(join(cfg.paths.data_path, 'data_config.yaml'), join(run_folder, 'data_config.yaml'))

def load_model_cfg_from_checkpoint(cfg: Config, config_name: str)->None:
    """If training from checkpoint, we need to get the old config"""
    model_path = cfg.paths.get('model_path', None)
    if model_path is not None: # if we are training from checkpoint, we need to load the old config
        old_cfg = load_config(join(cfg.paths.model_path, config_name))
        cfg.model = old_cfg.model

def load_checkpoint_and_epoch(cfg: Config)->Tuple:
    model_path = cfg.paths.get('model_path', None)
    checkpoint = ModelLoader(cfg).load_checkpoint() if model_path is not None else None
    epoch = Utilities.get_last_checkpoint_epoch(join(model_path, CHECKPOINTS_DIR)) if model_path is not None else None
    return checkpoint, epoch


class AzurePathContext:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.azure_env = cfg.env=='azure'
        self.run, self.mount_context = self.setup_run_and_mount_context()
        if self.azure_env:
            self.mount_point = self.mount_context.mount_point
    
    def setup_run_and_mount_context(self)->Tuple:
        run, mount_context = None, None
        if self.azure_env:
            run, mount_context = setup_azure(self.cfg.paths.run_name)
        return run, mount_context

    def adjust_paths_for_azure_pretrain(self)->Tuple:
        """
        Adjusts the following paths in the configuration for the Azure environment:
        - data_path
        - model_path
        - output_path
        """
        if self.azure_env:
            self.cfg.paths.data_path = self.prepend_mount_point(self.cfg.paths.data_path)
            if self.cfg.paths.get('model_path', None) is not None:
                self.cfg.paths.model_path = self.prepend_mount_point(self.cfg.paths.model_path)
            if not self.cfg.paths.output_path.startswith('outputs'):
                self.cfg.paths.output_path = join('outputs', self.cfg.paths.output_path)
        return self.cfg, self.run, self.mount_context
    
    def azure_onehot_setup(self)->Tuple:
        """Azure setup for onehot encoding. Prepend mount folder."""
        if self.azure_env:
            self.cfg.paths.finetune_features_path = self.prepend_mount_point(self.cfg.paths.finetune_features_path)
            self.cfg.paths.output_path = OUTPUTS_DIR
            
        return self.cfg, self.run, self.mount_context
    
    def azure_finetune_setup(self)->Tuple:
        """Azure setup for finetuning. Prepend mount folder."""
        if self.azure_env:
            self.cfg.paths.pretrain_model_path = self.prepend_mount_point(self.cfg.paths.pretrain_model_path)
            self.cfg.paths.outcome = self.prepend_mount_point(self.cfg.paths.outcome)
            if self.cfg.paths.get('censor', None) is not None:
                self.cfg.paths.censor = self.prepend_mount_point(self.cfg.paths.censor)
            self.cfg.paths.output_path = OUTPUTS_DIR
        return self.cfg, self.run, self.mount_context
    
    def add_pretrain_info_to_cfg(self)->Config:
        """Add information about pretraining to the config. Used in finetuning.
        We need first to get the pretrain information, before we can prepend the mount folder to the data path."""
        pretrain_cfg = load_config(join(self.cfg.paths.pretrain_model_path, 'pretrain_config.yaml'))
        pretrain_data_path = self.remove_mount_folder(pretrain_cfg.paths.data_path)
        
        self.cfg.data.remove_background = pretrain_cfg.data.remove_background
        self.cfg.paths.tokenized_dir = pretrain_cfg.paths.tokenized_dir

        self.cfg.paths.data_path = pretrain_data_path 
        if self.azure_env: # if we are in azure, we need to prepend the mount folder
            self.cfg.paths.data_path = self.prepend_mount_point(self.cfg.paths.data_path)
        return self.cfg
    
    def prepend_mount_point(self, path: str)->str:
        """Prepend mount point to path."""
        if self.azure_env:
            path = join(self.mount_point, path)
        return path

    @staticmethod
    def remove_mount_folder(path_str: str) -> str:
        """Remove mount folder from path."""
        path_parts = split_path(path_str)
        return os.path.join(*[part for part in path_parts if not part.startswith('tmp')])
