from azureml.core import Workspace, Datastore

def get_workspace():
    """Initializes workspase and gets datastor and dump_path"""
    subscription_id = 'f8c5aac3-29fc-4387-858a-1f61722fb57a'
    resource_group = 'forskerpl-n0ybkr-rg'
    workspace_name = 'forskerpl-n0ybkr-mlw'
    
    workspace = Workspace(subscription_id, resource_group, workspace_name)
    return workspace

def get_datastore(name):
    workspace = get_workspace()
    datastore = Datastore.get(workspace, name)
    return datastore

