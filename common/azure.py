from os.path import join

def get_workspace():
    from azureml.core import Workspace, Datastore
    """Initializes workspase and gets datastor and dump_path"""
    subscription_id = 'f8c5aac3-29fc-4387-858a-1f61722fb57a'
    resource_group = 'forskerpl-n0ybkr-rg'
    workspace_name = 'forskerpl-n0ybkr-mlw'
    
    workspace = Workspace(subscription_id, resource_group, workspace_name)
    return workspace

def get_datastore(name):
    from azureml.core import Datastore
    workspace = get_workspace()
    datastore = Datastore.get(workspace, name)
    return datastore

def setup_azure(cfg):
    from azure_run.run import Run
    from azure_run import datastore
    from azureml.core import Dataset

    run = Run
    run.name(f"Pretrain hierarchical diagnosis medication")
    ds = datastore("workspaceblobstore")
    dataset = Dataset.File.from_files(path=(ds, 'PHAIR'))
    mount_context = dataset.mount()
    mount_context.start()  # this will mount the file streams
    cfg.paths.data_path = join(mount_context.mount_point, cfg.paths.data_path)
    setup = {'run': run, 'mount_context': mount_context, 'cfg': cfg}
    return setup

