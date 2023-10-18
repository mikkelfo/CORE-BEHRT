from os.path import join

def get_workspace():
    from azureml.core import Workspace, Datastore
    """Initializes workspase and gets datastor and dump_path"""
    subscription_id = 'f8c5aac3-29fc-4387-858a-1f61722fb57a'
    resource_group = 'forskerpl-n0ybkr-rg'
    workspace_name = 'forskerpl-n0ybkr-mlw'
    
    workspace = Workspace(subscription_id, resource_group, workspace_name)
    return workspace

def setup_azure(run_name, datastore_name='workspaceblobstore', dataset_name='PHAIR'):
    """Sets up azure run and mounts data on PHAIR blobstore"""
    from azure_run.run import Run
    from azure_run import datastore
    from azureml.core import Dataset
    
    run = Run
    run.name(run_name)
    ds = datastore(datastore_name)
    dataset = Dataset.File.from_files(path=(ds, dataset_name))
    mount_context = dataset.mount()
    mount_context.start()  # this will mount the file streams
    return run, mount_context

