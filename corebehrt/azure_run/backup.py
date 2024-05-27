import os
import json
from azureml.core import Dataset

_BACKUP_DATASTORE = "researcher_data"
_BACKUP_ROOT_DIR  = "data-backup"

def dataset(name, remote_path, version=None, overwrite_ok=False, post_validate_compare=True, post_validate_func=None):
    global _BACKUP_DATASTORE, _BACKUP_ROOT_DIR
    from . import log, datastore, dataset as load_dataset

    # Validate
    if type(name)!=str:        raise Exception(f"Invalid parameter 'name', expected type str.")
    if type(remote_path)!=str: raise Exception(f"Invalid parameter 'remote_name', expected type str.")
    # Clean path
    if remote_path[0] == "/": remote_path = remote_path[1:]
    elif remote_path[:2] == "./": remote_path = remote_path[2:]
    if remote_path[-1] == "/": remote_path = remote_path[:-1]

    # Get datastore and full remote path
    dast = datastore(name=_BACKUP_DATASTORE)
    full_remote_path = f"{_BACKUP_ROOT_DIR}/{remote_path}/{name}"

    tmp_dir = "./._backup_"
    local_dir = f"{tmp_dir}/{name}/"
    
    # Get dataset
    log().info(f"Making backup of dataset {name}.")
    ds = load_dataset(name)
    log().debug(f"Fetching dataset...")
    df = ds.to_pandas_dataframe()
    log().info(f"Loaded {name}:{ds.version}, got {len(df)} rows with {len(df.columns)} columns!")

    if not overwrite_ok:
        # Check file does not exist already
        found = False
        try:
            ds = Dataset.Tabular.from_parquet_files((dast,full_remote_path+"/data.parquet"))
            found = True
        except Exception as e:
            # Should give an exception
            if not e.error_code == "ScriptExecution.StreamAccess.NotFound":
                # Unexpected exception?
                raise Exception(f"Looking for dataset @ {(_BACKUP_DATASTORE,full_remote_path)} caused an unexpected exception: {e}.")
        
        if found: raise Exception(f"Cannot make backup of {name}, backup {(_BACKUP_DATASTORE,full_remote_path)} already exists. Set ovewrite_ok=True to ignore!")

    try:
        # Store locally
        log().info(f"Storing files locally!")
        # Create temporary directory
        os.makedirs(local_dir)
        # Store ds as local file
        df.to_parquet(local_dir+"data.parquet")
        # Store meta
        with open(local_dir+"meta.json", "w") as fm:
            fm.write(json.dumps({
                "name":ds.name,
                "version":ds.version,
                "tags":ds.tags,
            }))
        
        # Upload to datastore
        log().info(f"Uploading to datastore {_BACKUP_DATASTORE}:{full_remote_path}...")
        Dataset.File.upload_directory(local_dir, (dast,full_remote_path))
        log().info("File upload successful!")
    except Exception as e:
        # Clean up and report failure
        log().error(f"Failed to backup dataset: {name}:{ds.version} to {(dast,full_remote_path)}!")
        raise e
    finally:
        # Clean-up
        log().debug("Cleaning up temporary data...")
        try: os.remove(local_dir+"meta.json")
        except OSError: pass
        try: os.remove(local_dir+"data.parquet")
        except OSError: pass
        try: os.rmdir(local_dir)
        except OSError: pass
        try: os.rmdir(tmp_dir)
        except OSError: pass
    
    # Post validation
    if post_validate_compare or post_validate_func is not None:
        ds = Dataset.Tabular.from_parquet_files((dast,full_remote_path+"/data.parquet"))
        df_backup = ds.to_pandas_dataframe()
            
        if post_validate_compare:
            log().info("Performing post-validation (comparison)...")
            if (df.columns!=df_backup.columns).any() or len(df) != len(df_backup):
                log().error("Post-validation (comparison) failed!")
                raise Exception("Post-validation (comparison) failed!")
            log().info("Validation (comparison) passed!")
        if post_validate_func is not None:
            log().info("Performing post-validation (custom)...")
            if post_validate_func(df_backup):
                log().error("Post-validation  (custom) failed!")
                raise Exception("Post-validation  (custom) failed!")
            log().info("Validation (custom) passed!")

def file(path, remote_location, overwrite_ok=False):
    from . import log
    raise Exception("Not implemented!")





