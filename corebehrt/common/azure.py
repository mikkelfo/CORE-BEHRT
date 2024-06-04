import logging
import os
import re
from os.path import join
from typing import Dict, List, Tuple, Union

from corebehrt.common.config import Config, load_config
from corebehrt.common.utils import split_path


logger = logging.getLogger(__name__)  # Get the logger for this module

OUTPUTS_DIR = "outputs"

def get_run_info():
    from azureml.core import Run
    """Get experiment name and run_id of the current run."""
    # Get the current run context
    run_context = Run.get_context()
    run_id = run_context.id
    experiment_name = run_context.experiment.name
    return experiment_name, run_id

def get_workspace():
    from corebehrt.azure_run import workspace
    """Initializes workspase and gets datastore and dump_path"""    
    return workspace()

def setup_azure(run_name, datastore_name='workspaceblobstore', dataset_name='PHAIR'):
    """Sets up azure run and mounts data on PHAIR blobstore"""
    from corebehrt.azure_run import datastore
    from corebehrt.azure_run.run import Run
    from azureml.core import Dataset
    
    run = Run
    run.name(run_name)
    ds = datastore(datastore_name)
    dataset = Dataset.File.from_files(path=(ds, dataset_name))
    mount_context = dataset.mount()
    mount_context.start()  # this will mount the file streams
    return run, mount_context

def get_max_retry_folder(folders):
    """Returns the folder with the maximum retry number"""
    # Regular expression to match the pattern retry_XXX
    pattern = r'^retry_(\d{3})$'
    
    # Extract all matching folder numbers
    retry_numbers = [int(re.search(pattern, folder).group(1)) for folder in folders if re.match(pattern, folder)]
    
    # Return the folder with the maximum retry number
    if retry_numbers:
        max_retry = max(retry_numbers)
        return f"retry_{max_retry:03}"
    else:
        return None
    
def save_to_blobstore(local_path: str, remote_path: str, overwrite: bool = False):
    """
    Saves a file to the blobstore. 
    local_path: The path to the file to be saved (inside outputs or the last retry folder)
    remote_path: The path inside workspaceblobstore to save the files to
    """
    try:
        from corebehrt.azure_run import file_dataset_save
        retry_folder = get_max_retry_folder(os.listdir('outputs'))
        output_path = 'outputs' if retry_folder is None else join('outputs', retry_folder)
        src_dir = join(output_path, local_path)
        logger.info(f"Try copying {src_dir} to {remote_path}")
        file_dataset_save(local_path=src_dir, datastore_name = "workspaceblobstore",
                    remote_path = remote_path, overwrite=overwrite)
        logger.info('Saved model to blob')
    except Exception as e:
        logger.warning(f'Could not save to blob. Exception: {e}')


class AzurePathContext:
    """Context manager for azure paths. Adjusts paths for azure environment."""
    def __init__(self, cfg: Config, dataset_name:str='PHAIR'):
        self.cfg = cfg
        self.azure_env = cfg.env=='azure'
        self.dataset_name = dataset_name
        self.run, self.mount_context = self.setup_run_and_mount_context()
        if self.azure_env:
            self.mount_point = self.mount_context.mount_point
    
    def setup_run_and_mount_context(self)->Tuple:
        """Setup run and mount context for azure environment."""
        run, mount_context = None, None
        if self.azure_env:
            run, mount_context = setup_azure(self.cfg.paths.run_name, dataset_name=self.dataset_name)
        return run, mount_context

    def adjust_paths_for_azure_pretrain(self)->Tuple:
        """
        Adjusts the following paths in the configuration for the Azure environment:
        - data_path
        - model_path
        - output_path
        """
        if self.azure_env:
            self.cfg.paths.data_path = self._prepend_mount_point(self.cfg.paths.data_path)
            if 'predefined_splits' in self.cfg.paths:
                self.cfg.paths.predefined_splits = self._prepend_mount_point(self.cfg.paths.predefined_splits)
            if self.cfg.paths.get('model_path', None) is not None:
                self.cfg.paths.model_path = self._prepend_mount_point(self.cfg.paths.model_path)
            self._handle_outputs_path()
        return self.cfg, self.run, self.mount_context
    
    def azure_onehot_setup(self)->Tuple:
        """Azure setup for onehot encoding. Prepend mount folder."""
        if self.azure_env:
            self.cfg.paths.finetune_features_path = self._prepend_mount_point(self.cfg.paths.finetune_features_path)
            self.cfg.paths.output_path = OUTPUTS_DIR
            
        return self.cfg, self.run, self.mount_context

    def azure_finetune_setup(self)->Tuple:
        """Azure setup for finetuning. Prepend mount folder."""
        if self.azure_env:
            for entry in self.cfg.paths:
                if isinstance(self.cfg.paths[entry], list):
                    new_list = []
                    for path in self.cfg.paths[entry]:
                        new_list.append(self._prepend_mount_point(path))
                    self.cfg.paths[entry] = new_list
                    continue
                if entry not in ['output_path', 'run_name', 'save_folder_path', 'tokenized_file', 'tokenized_pids', 'tokenized_dir']:
                    self.cfg.paths[entry] = self._prepend_mount_point(self.cfg.paths[entry])
            self.cfg.paths.output_path = OUTPUTS_DIR
        return self.cfg, self.run, self.mount_context
    
    def azure_data_pretrain_setup(self)->Tuple:
        """Azure setup for pretraining. Prepend mount folder."""
        if self.azure_env:
            self.cfg.loader.data_dir = self._prepend_mount_point(self.cfg.loader.data_dir)
            if 'vocabulary' in self.cfg.paths:
                self.cfg.paths.vocabulary = self._prepend_mount_point(self.cfg.paths.vocabulary)
            if 'predefined_splits_dir' in self.cfg:
                self.cfg.predefined_splits_dir = self._prepend_mount_point(self.cfg.predefined_splits_dir)
            if 'exclude_pids' in self.cfg and self.cfg['exclude_pids']:
                self.cfg.exclude_pids = self.prepend_mount_point_to_paths(self.cfg['exclude_pids'])
            if 'assigned_pids' in self.cfg and self.cfg['assigned_pids']:
                self.cfg.assigned_pids = self.prepend_mount_point_to_assigned_pids(self.cfg['assigned_pids'])
            self._handle_outputs_path()
        return self.cfg, self.run, self.mount_context

    def azure_outcomes_setup(self)->Tuple:
        """Azure setup for outcomes. Prepend mount folder."""
        if self.azure_env:
            self.cfg.loader.data_dir = self._prepend_mount_point(self.cfg.loader.data_dir)
            self.cfg.features_dir = self._prepend_mount_point(self.cfg.features_dir)
            self.cfg.paths.outcome_dir = 'outputs/outcomes'
            self._handle_outputs_path()
        return self.cfg, self.run, self.mount_context

    def azure_encode_setup(self)->Tuple:
        """Azure setup for encoding. Prepend mount folder."""
        if self.azure_env:
            self.cfg.paths.data_path = self._prepend_mount_point(self.cfg.paths.data_path)
            self.cfg.paths.model_path = self._prepend_mount_point(self.cfg.paths.model_path)
            self.cfg.paths.outcomes_path = self._prepend_mount_point(self.cfg.paths.outcomes_path)
            self._handle_outputs_path()
        return self.cfg, self.run, self.mount_context

    def azure_hierarchical_setup(self)->Tuple:
        """Azure setup for hierarchical encoding. Prepend mount folder."""
        if self.azure_env:
            self.cfg.paths.features = self._prepend_mount_point(self.cfg.paths.features)
            self._handle_outputs_path()
        return self.cfg, self.run, self.mount_context

    def add_pretrain_info_to_cfg(self)->Config:
        """Add information about pretraining to the config. Used in finetuning.
        We need first to get the pretrain information, before we can prepend the mount folder to the data path."""
        pretrain_model_path = self.cfg.paths.get('pretrain_model_path')
        model_path = self.cfg.paths.get('model_path')
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
        
        logger.info(f"Loading pretrain config from {pretrain_cfg_path}")
        pretrain_cfg = load_config(pretrain_cfg_path)
 
        pretrain_data_path = self._remove_mount_folder(pretrain_cfg.paths.data_path)
        
        if 'tokenized_dir' not in self.cfg.paths:
            logger.info("Tokenized dir not in config. Adding from pretrain config.")
            self.cfg.paths.tokenized_dir = pretrain_cfg.paths.tokenized_dir

        self.cfg.paths.data_path = pretrain_data_path 
        if self.azure_env: # if we are in azure, we need to prepend the mount folder
            self.cfg.paths.data_path = self._prepend_mount_point(self.cfg.paths.data_path)
        return self.cfg
    
    def _handle_outputs_path(self)->None:
        """Handle the output path in the configuration."""
        if self.cfg.paths.get('output_path', None) is None:
            self.cfg.paths.output_path = OUTPUTS_DIR
        else:
            if not self.cfg.paths.output_path.startswith(OUTPUTS_DIR):
                self.cfg.paths.output_path = join(OUTPUTS_DIR, self.cfg.paths.output_path)

    def _prepend_mount_point(self, path: str)->str:
        """Prepend mount point to path."""
        if self.azure_env:
            path = join(self.mount_point, path)
        return path

    @staticmethod
    def _remove_mount_folder(path_str: str) -> str:
        """Remove mount folder from path."""
        path_parts = split_path(path_str)
        return os.path.join(*[part for part in path_parts if not part.startswith('tmp')])
    
    def prepend_mount_point_to_assigned_pids(self, assigned_pids: Dict[str, Union[str, List[str]]])->Dict[str, Union[str, List[str]]]:
        """Prepend mount_path to paths in the configuration."""
        for split, paths in assigned_pids.items():
            assigned_pids[split] = self.prepend_mount_point_to_paths(paths)
        return assigned_pids
    
    def prepend_mount_point_to_paths(self, paths: Union[str, List[str]])->Union[str, List[str]]:
        """Prepends mount_path to each path in the provided list or single path."""
        if isinstance(paths, list):
            return [self._prepend_mount_point(path) for path in paths]
        return self._prepend_mount_point(paths)
