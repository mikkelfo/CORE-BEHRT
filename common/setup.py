import logging
import os
import uuid
from os.path import join
from shutil import copyfile

from common.config import Config


def prepare_directory(config_path: str, cfg: Config):
    """Creates output directory and copies config file"""
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    copyfile(config_path, join(cfg.output_dir, 'data_config.yaml'))
    if not os.path.exists(join(cfg.output_dir, 'features')):
        os.makedirs(join(cfg.output_dir, 'features'))
    if not os.path.exists(join(cfg.output_dir, 'tokenized')):
        os.makedirs(join(cfg.output_dir, 'tokenized'))
    logging.basicConfig(filename=join(cfg.output_dir, 'info.log'), level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger

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
  
    logging.basicConfig(filename=join(run_folder, 'info.log'), level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f'Run folder: {run_folder}')
    return logger