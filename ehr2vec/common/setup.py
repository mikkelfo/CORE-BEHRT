import argparse
import logging
import os
import uuid
from os.path import join
from shutil import copyfile

from common.azure import setup_azure
from common.config import Config, load_config

OUTPUTS_DIR = "outputs"
CHECKPOINTS_DIR = "checkpoints"

def get_args(default_config_name, default_run_name=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=join('configs', default_config_name))
    parser.add_argument('--run_name', type=str, default=default_run_name if default_run_name else default_config_name.split('.')[0])
    return parser.parse_args()

def create_directory_and_copy_config(config_path: str, output_dir: str, new_config_name: str):
    os.makedirs(output_dir, exist_ok=True)
    destination = join(output_dir, new_config_name)
    copyfile(config_path, destination)
    return setup_logger(output_dir)

def setup_logger(dir: str, log_file: str = 'info.log'):
    """Sets up the logger."""
    logging.basicConfig(filename=join(dir, log_file), level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

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

def setup_run_folder(cfg, run_folder=None):
    """Creates a run folder"""
    # Generate unique run_name if not provided
    run_name = cfg.paths.run_name if hasattr(cfg.paths, 'run_name') else uuid.uuid4().hex
    if run_folder is None:
        run_folder = join(cfg.paths.output_path, run_name)

    os.makedirs(run_folder, exist_ok=True)
    os.makedirs(join(run_folder, CHECKPOINTS_DIR), exist_ok=True)
    
    tokenized_dir_name = cfg.paths.get('tokenized_dir', 'tokenized')
    try:
        copyfile(join(cfg.paths.data_path, tokenized_dir_name, 'data_config.yaml'), join(run_folder, 'data_config.yaml'))
    except:
        copyfile(join(cfg.paths.data_path, 'data_config.yaml'), join(run_folder, 'data_config.yaml'))
    logger = setup_logger(run_folder)
    logger.info(f'Run folder: {run_folder}')
    return logger

def azure_finetune_setup(cfg):
    if cfg.env=='azure':
        run, mount_context = setup_azure(cfg.paths.run_name)
        cfg.paths.data_path = join(mount_context.mount_point, cfg.paths.data_path)
        cfg.paths.model_path = join(mount_context.mount_point, cfg.paths.model_path)
        cfg.paths.outcome = join(mount_context.mount_point, cfg.paths.outcome)
        if cfg.paths.get('censor', None) is not None:
            cfg.paths.censor = join(mount_context.mount_point, cfg.paths.censor)
        cfg.paths.output_path = OUTPUTS_DIR
        return cfg, run, mount_context
    return cfg, None, None

def azure_onehot_setup(cfg):
    if cfg.env=='azure':
        run, mount_context = setup_azure(cfg.paths.run_name)
        cfg.finetune_features_path = join(mount_context.mount_point, cfg.finetune_features_path)
        cfg.paths.output_path = OUTPUTS_DIR
        return cfg, run, mount_context
    return cfg, None, None

def adjust_paths_for_finetune(cfg):
    cfg.paths.output_path = cfg.paths.model_path
    cfg.paths.run_name = construct_finetune_model_dir_name(cfg)
    return cfg

def add_pretrain_info_to_cfg(cfg):
    pretrain_cfg = load_config(join(cfg.paths.model_path, 'pretrain_config.yaml'))
    cfg.data.remove_background = pretrain_cfg.data.remove_background
    cfg.paths.tokenized_dir = pretrain_cfg.paths.tokenized_dir
    return cfg

def construct_finetune_model_dir_name(cfg):
    days = True if abs(cfg.outcome.n_hours)>48 else False
    window = int(abs(cfg.outcome.n_hours/24)) if days else abs(cfg.outcome.n_hours)
    days_hours = 'days' if days else 'hours'
    pre_post = 'pre' if cfg.outcome.n_hours<0 else 'post'
    return f"finetune_{cfg.outcome.type}_censored_{window}_{days_hours}_{pre_post}_{cfg.outcome.censor_type}_{cfg.paths.run_name}"
