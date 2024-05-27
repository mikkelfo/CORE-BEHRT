"""aiomic.core provides core functionality.

Core functionality includes e.g. loading and saving of datasets, manipulating runs,
making backups to a separate datastore, and more...
"""

from azureml.core import Dataset, Datastore, Workspace, Model
import os
from os.path import split
import pandas as pd

def log():
    return logger.log(name=__name__)


_DATASTORES = {
    "workspaceblobstore",
    "sp_data",
    "researcher_data",
    "workspaceartifactstore"
}
# FILL IN YOUR WORKSPACE CONFIG HERE
_WS_CONFIG = {
    "subscription_id": "",
    "resource_group": "",
    "workspace_name": ""
}
_WS = None

def workspace() -> Workspace:
    """Load workspace with authentication

    Returns
    -------
    Workspace
        An authenticated AzureML workspace object
    """
    global _WS_CONFIG
    global _WS
    if _WS is None:
        if Run.is_remote():
            _WS = Run.init().remote.experiment.workspace
        else:
            _WS = Workspace(_WS_CONFIG["subscription_id"], _WS_CONFIG["resource_group"], _WS_CONFIG["workspace_name"])
    return _WS

def datastore(name: str = "workspaceblobstore") -> Datastore:
    """Load requested datastore.

    Parameters
    ----------
    name : str
        The name of the datastore to load (default='workspaceblobstore').

    Returns
    -------
    Datastore
        An AzureML datastore object.
    """
    global _DATASTORES
    if name not in _DATASTORES:
        raise Exception(f"Unknown datastore: {name}")
    ws = workspace()
    return Datastore.get(ws, name)

def dataset(name: str, version: int = None) -> Dataset:
    """Load requested dataset

    Parameters
    ----------
    name : str
        The name of the dataset to load.
    version : int
        The version of the dataset to load (default is newest).

    Returns
    -------
    Dataset
        An AzureML dataset object.
    """
    return Dataset.get_by_name(workspace(), name, version=version)
    
_DS_LIST_CACHE = None

def dataset_save(df: pd.DataFrame, name: str, tags: dict = None, description: str = None):
    """Save given dataset.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame to save.
    name : str
        The name of the dataset to save.
    tags : dict(str: any)
        Dictionary of tags (values will be converted to str).
    description : str
        Description for the dataset.

    Returns
    -------
    Dataset
        The AzureML dataset object created for the dataset.
    """
    global _DS_LIST_CACHE
    _DS_LIST_CACHE = None # Invalidate cache
    return Dataset.Tabular.register_pandas_dataframe(df, datastore(), name, show_progress=False, tags=tags, description=description)

def file_dataset_save(local_path: str, tags: dict = None, description: str = None, datastore_name = "workspaceblobstore", remote_path = "PHAIR"):
    """Save given file dataset (given as txt files in a local directory).

    Parameters
    ----------
    local_path : str
        Path to local directory containing files.
    name : str
        The name of the dataset to save.
    tags : dict(str: any)
        Dictionary of tags (values will be converted to str).
    description : str
        Description for the dataset.

    Returns
    -------
    Dataset
        The AzureML dataset object created for the dataset.
    """
    global _DS_LIST_CACHE
    _DS_LIST_CACHE = None # Invalidate cache

    dtst = datastore(name=datastore_name)

    ds = Dataset.File.upload_directory(local_path, (dtst, remote_path))
    
    # Register
    return ds.register(workspace=workspace(), name=split(remote_path)[1], tags=tags, description=description, create_new_version=True)

def dataset_list(tags=None):
    """List datasets registered with the default workspace.

    Parameters
    ----------
    tags : set of str or list of tuples (str, str)
        Set of tags or list of tuples (tag, value). Datasets matching at
        least one tag/tag pair will be returned.

    Returns
    -------
    list
        A list of dicts with keys: name, tags, version, description.
    """
    global _DS_LIST_CACHE
    if _DS_LIST_CACHE is None:
        # Update cache
        log().debug(f"Fetching dataset list cache...")
        _DS_LIST_CACHE = Dataset.get_all(workspace())
    
    # Prepare filter
    filt = lambda ts: True
    if type(tags)==set:
        filt = lambda ts: len(ts.keys()&tags)>0
    elif type(tags)==list:
        filt = lambda ts: any([ts.get(t)==v for t,v in tags])
    res = []
    for ds_name, ds in _DS_LIST_CACHE.items():
        if filt(ds.tags):
            res.append({"name":ds_name,"tags":ds.tags,"version":ds.version,"description":ds.description})
    return res
    

from . import backup
from . import logger
from .run import Run
from .model import Model
from .validation import validate

