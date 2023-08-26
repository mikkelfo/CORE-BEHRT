import argparse
import logging
import os
import uuid
from os.path import join
from shutil import copyfile

from common.config import Config


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

def prepare_directory(config_path: str, cfg: Config):
    """Creates output directory and copies config file"""
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(join(cfg.output_dir, 'features'), exist_ok=True)
    os.makedirs(join(cfg.output_dir, 'tokenized'), exist_ok=True)
    copyfile(config_path, join(cfg.output_dir, 'data_config.yaml'))
    
    return setup_logger(cfg.output_dir)

def prepare_directory_outcomes(config_path: str, outcome_dir: str, outcomes_name: str):
    os.makedirs(outcome_dir, exist_ok=True)
    copyfile(config_path, join(outcome_dir, f'outcome_{outcomes_name}_config.yaml'))
    
    return setup_logger(outcome_dir)  

def prepare_directory_hierarchical(config_path: str, out_dir: str, hierarchical_name: str = 'hierarchical'):
    """Creates hierarchical directory and copies config file"""
    hierarchical_dir = join(out_dir, hierarchical_name)
    os.makedirs(hierarchical_dir, exist_ok=True)
    copyfile(config_path, join(hierarchical_dir, 'h_setup.yaml'))
    return setup_logger(hierarchical_dir)

def prepare_embedding_directory(config_path: str, cfg: Config):
    """Creates output directory and copies config file"""
    os.makedirs(cfg.output_dir, exist_ok=True)
    copyfile(config_path, join(cfg.output_dir, 'emb_config.yaml'))
    return setup_logger(cfg.output_dir)

def prepare_encodings_directory(config_path: str, cfg: Config):
    """Creates output directory and copies config file"""
    os.makedirs(cfg.output_dir, exist_ok=True)
    copyfile(config_path, join(cfg.output_dir, 'encodings_config.yaml'))
    return setup_logger(cfg.output_dir)

def setup_run_folder(cfg):
    """Creates a run folder"""
    # Generate unique run_name if not provided
    if hasattr(cfg.paths, 'run_name'):
        run_name = cfg.paths.run_name
    else:
        run_name = uuid.uuid4().hex
       
    run_folder = join(cfg.paths.output_path, run_name)

    if not os.path.exists(run_folder):
        os.makedirs(run_folder) 
    if not os.path.exists(join(run_folder,'checkpoints')):
        os.makedirs(join(run_folder,'checkpoints')) 
    copyfile(join(cfg.paths.data_path, 'data_config.yaml'), join(run_folder, 'data_config.yaml'))
    logger = setup_logger(run_folder)
    logger.info(f'Run folder: {run_folder}')
    return logger