import logging
import os
import re
from os.path import join

logger = logging.getLogger(__name__)  # Get the logger for this module

def get_workspace():
    from azureml.core import Workspace
    """Initializes workspase and gets datastor and dump_path"""
    subscription_id = 'f8c5aac3-29fc-4387-858a-1f61722fb57a'
    resource_group = 'forskerpl-n0ybkr-rg'
    workspace_name = 'forskerpl-n0ybkr-mlw'
    
    workspace = Workspace(subscription_id, resource_group, workspace_name)
    return workspace

def setup_azure(run_name, datastore_name='workspaceblobstore', dataset_name='PHAIR'):
    """Sets up azure run and mounts data on PHAIR blobstore"""
    from azure_run import datastore
    from azure_run.run import Run
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
    
def save_to_blobstore(local_path: str, remote_path: str):
    """
    Saves a file to the blobstore. 
    local_path: The path to the file to be saved (inside outputs or the last retry folder)
    remote_path: The path inside workspaceblobstore to save the files to
    """
    try:
        from azure_run import file_dataset_save
        retry_folder = get_max_retry_folder(os.listdir('outputs'))
        output_path = 'outputs' if retry_folder is None else join('outputs', retry_folder)
        src_dir = join(output_path, local_path)
        logger.info(f"Try copying {src_dir} to {remote_path}")
        file_dataset_save(local_path=src_dir, datastore_name = "workspaceblobstore",
                    remote_path = remote_path)
        logger.info('Saved model to blob')
    except:
        logger.warning('Could not save model to blob')


