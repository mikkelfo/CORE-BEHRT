import argparse
import logging
import os
import uuid
from os.path import join, split
from shutil import copyfile
from typing import Tuple

from corebehrt.common.config import Config

logger = logging.getLogger(__name__)  # Get the logger for this module

CHECKPOINTS_DIR = "checkpoints"

def get_args(default_config_name, default_run_name=None):
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=join('configs', default_config_name))
    parser.add_argument('--run_name', type=str, default=default_run_name if default_run_name else default_config_name.split('.')[0])
    return parser.parse_args()

def setup_logger(dir: str, log_file: str = 'info.log'):
    """Sets up the logger."""
    logging.basicConfig(filename=join(dir, log_file), level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def copy_data_config(cfg: Config, run_folder: str)->None:
    """
    Copy data_config.yaml to run folder.
    By default copy from tokenized folder, if not available, copy from data folder.
    """
    tokenized_dir_name = cfg.paths.get('tokenized_dir', 'tokenized')
    
    try:
        copyfile(join(cfg.paths.data_path, tokenized_dir_name, 'data_cfg.yaml'), join(run_folder, 'data_config.yaml'))
    except:
        copyfile(join(cfg.paths.data_path, 'data_config.yaml'), join(run_folder, 'data_config.yaml'))

def copy_pretrain_config(cfg: Config, run_folder: str)->None:
    """
    Copy pretrain_config.yaml to run folder.
    """
    pretrain_model_path = cfg.paths.get('pretrain_model_path')
    model_path = cfg.paths.get('model_path')
    pt_cfg_name = 'pretrain_config.yaml'
    
    pretrain_cfg_path = pretrain_model_path if pretrain_model_path is not None else model_path
    if pretrain_cfg_path is None:
        raise ValueError("Either pretrain_model_path or model_path must be specified in the configuration.")
    
    if os.path.exists(join(pretrain_cfg_path, pt_cfg_name)):
        pretrain_cfg_path = join(pretrain_cfg_path, pt_cfg_name)
    elif os.path.exists(join(pretrain_cfg_path, 'fold_1', pt_cfg_name)):
        pretrain_cfg_path = join(pretrain_cfg_path, 'fold_1', pt_cfg_name)
    else:
        raise FileNotFoundError(f"Could not find pretrain config in {pretrain_cfg_path}")
    try:
        copyfile(pretrain_cfg_path, join(run_folder, pt_cfg_name))
    except:
        logger.warning(f"Could not copy pretrain config from {pretrain_cfg_path} to {run_folder}")
        

class DirectoryPreparer:
    """Prepares directories for training and evaluation."""
    def __init__(self, config_path) -> None:
        self.config_path = config_path
        
    def create_directory_and_copy_config(self, output_dir: str, new_config_name: str)->logging.Logger:
        """Creates output directory and copies config file"""
        os.makedirs(output_dir, exist_ok=True)
        destination = join(output_dir, new_config_name)
        copyfile(self.config_path, destination)
        return setup_logger(output_dir)

    def prepare_directory(self, cfg: Config):
        """Creates output directory and copies config file"""
        logger = self.create_directory_and_copy_config(cfg.output_dir, 'data_config.yaml')
        os.makedirs(join(cfg.output_dir, 'features'), exist_ok=True)
        os.makedirs(join(cfg.output_dir, cfg.tokenized_dir_name), exist_ok=True)
        copyfile(self.config_path, join(cfg.output_dir, cfg.tokenized_dir_name, 'data_config.yaml'))
        return logger

    def prepare_directory_outcomes(self, outcome_dir: str, outcomes_name: str):
        """Creates output directory for outcomes and copies config file"""
        return self.create_directory_and_copy_config(outcome_dir, f'outcome_{outcomes_name}_config.yaml')

    def prepare_embedding_directory(self, cfg: Config):
        """Creates output directory and copies config file"""
        return self.create_directory_and_copy_config(cfg.output_dir, 'emb_config.yaml')

    def prepare_encodings_directory(self, cfg: Config):
        """Creates output directory and copies config file"""
        return self.create_directory_and_copy_config(cfg.output_dir, 'encodings_config.yaml')
    
    @staticmethod
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
    
    @staticmethod
    def adjust_paths_for_finetune(cfg: Config)->Config:
        """
        Adjusts the following paths in the configuration for the finetune environment:
        - output_path: set to pretrain_model_path
        - run_name: constructed according to setting
        """
        pretrain_model_path = cfg.paths.get('pretrain_model_path')
        model_path = cfg.paths.get('model_path') 
        if model_path is not None:
            model_path = split(model_path)[0] # Use directory of model path (the model path will be constructed in the finetune script)
        save_folder_path = cfg.paths.get('save_folder_path')
        
       # Determine the output path with a priority order
        output_path = pretrain_model_path or model_path or save_folder_path
        if output_path is None:
            raise ValueError("Either pretrain_model_path, model_path, or save_folder_path must be provided.")
        cfg.paths.output_path = output_path
        cfg.paths.run_name = DirectoryPreparer.construct_finetune_model_dir_name(cfg)
        return cfg
    
    @staticmethod
    def construct_finetune_model_dir_name(cfg: Config)->str:
        """Constructs the name of the finetune model directory. Based on the outcome type, the censor type, and the number of hours pre- or post- outcome."""
        days = True if abs(cfg.outcome.n_hours)>48 else False
        window = int(abs(cfg.outcome.n_hours/24)) if days else abs(cfg.outcome.n_hours)
        days_hours = 'days' if days else 'hours'
        pre_post = 'pre' if cfg.outcome.n_hours<0 else 'post'
        if isinstance(cfg.outcome.censor_type, str):
            censor_event = cfg.outcome.censor_type
        elif isinstance(cfg.outcome.censor_type, dict):
            censor_event = [f"{k}{v}" for k, v in cfg.outcome.censor_type.items() if v is not None]
            censor_event = '_'.join(censor_event)
        else:
            raise ValueError(f"Unknown censor type {cfg.outcome.censor_type}")
        return f"finetune_{cfg.outcome.type}_censored_{window}_{days_hours}_{pre_post}_{censor_event}_{cfg.paths.run_name}"


