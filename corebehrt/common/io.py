import os
from os.path import split
import torch
import h5py
from typing import Tuple, List


class BaseHDF5Writer:
    """Base class to write encodings to HDF5 file in batches."""
    def __init__(self, output_path:str, encodings_dataset_name:str="X",
                 targets_dataset_name:str="y"):
        self.output_path = output_path
        self.encodings_dataset_name = encodings_dataset_name
        self.targets_dataset_name = targets_dataset_name
        self.hidden_dim = None
        self.initialized = False

    def initialize(self, hidden_dim:int)->None:
        if os.path.exists(self.output_path):
            raise ValueError(f"File {self.output_path} already exists!")
        os.makedirs(split(self.output_path)[0], exist_ok=True)
        with h5py.File(self.output_path, 'a') as f:
            if self.encodings_dataset_name not in f:
                f.create_dataset(self.encodings_dataset_name, shape=(0, hidden_dim), maxshape=(None, hidden_dim), dtype=float)
            if self.targets_dataset_name not in f:
                f.create_dataset(self.targets_dataset_name, shape=(0,), maxshape=(None,), dtype=int)
        self.initialized = True
        self.hidden_dim = hidden_dim

    def write_tensor(self, tensor:torch.Tensor, tensor_dset:h5py.Dataset)->None:
        """Write tensor to HDF5 dataset."""
        tensor_dset, start, stop = self.resize_tensor_dataset(tensor_dset, tensor)
        tensor_dset[start:stop, :] = tensor.cpu().numpy()
        
    def write_targets(self, targets:torch.Tensor, targets_dset:h5py.Dataset)->None:
        """Write targets to HDF5 dataset."""
        targets_dset, start, stop = self.resize_target_dataset(targets_dset, targets)
        targets_dset[start:stop] = targets.cpu().numpy()

    def resize_tensor_dataset(self, tensor_dset:h5py.Dataset, tensor:torch.Tensor
                              )->Tuple[h5py.Dataset, int, int]:
        """Resize the tensor dataset to accomodate the new tensor."""
        current_tensor_len = tensor_dset.shape[0]
        new_tensor_len = current_tensor_len + tensor.shape[0]
        tensor_dset.resize((new_tensor_len, self.hidden_dim))
        return tensor_dset, current_tensor_len, new_tensor_len

    def resize_target_dataset(self, targets_dset:h5py.Dataset, targets:torch.Tensor
                              )->Tuple[h5py.Dataset, int, int]:
        """Resize the target dataset to accomodate the new targets."""
        current_targets_len = targets_dset.shape[0]
        new_targets_len = current_targets_len + len(targets)
        targets_dset.resize((new_targets_len,)) 
        return targets_dset, current_targets_len, new_targets_len

class ConceptHDF5Writer(BaseHDF5Writer):
    """Class to write concept encodings (n_conceptsxhidden_dim) to HDF5 file in batches."""
    def write(self, tensor:torch.Tensor, labels:torch.Tensor=None):
        """Save tensor and pids to HDF5."""
        if not self.initialized or self.hidden_dim != tensor.shape[1]:
            self.initialize(tensor.shape[1])

        with h5py.File(self.output_path, 'a') as f:
            tensor_dset = f[self.encodings_dataset_name]
            self.write_tensor(tensor, tensor_dset)
            # Append the pids to the HDF5 pid dataset

            if labels is not None:
                labels_dset = f[self.targets_dataset_name]
                self.write_targets(labels, labels_dset)

class PatientHDF5Writer(BaseHDF5Writer):
    """Class to write patient encodings (bsxhidden_dim) to HDF5 file in batches."""
    def __init__(self, output_path:str, encodings_dataset_name:str="X", 
                 pid_dataset_name:str="pids",
                 targets_dataset_name:str="y"):
        super().__init__(output_path, encodings_dataset_name, targets_dataset_name)
        self.pid_dataset_name = pid_dataset_name
        
    def write(self, tensor:torch.Tensor, pids:List[str], targets:torch.Tensor=None):
        """Save tensor and pids to HDF5."""
        if not self.initialized or self.hidden_dim != tensor.shape[1]:
            self.initialize(tensor.shape[1])

        with h5py.File(self.output_path, 'a') as f:
            tensor_dset = f[self.encodings_dataset_name]
            pid_dset = f[self.pid_dataset_name]
           
            self.write_tensor(tensor, tensor_dset)
            # Append the pids to the HDF5 pid dataset
            self.write_pids(pids, pid_dset)

            if targets is not None:
                targets_dset = f[self.targets_dataset_name]
                self.write_targets(targets, targets_dset)

    def write_pids(self, pids:List[str], pid_dset:h5py.Dataset)->None:
        """Write pids to HDF5 dataset."""
        pid_dset, start, stop = self.resize_pid_dataset(pid_dset, pids)
        pid_dset[start:stop] = pids

    def resize_pid_dataset(self, pid_dset:h5py.Dataset, pids:List[str]
                           )->Tuple[h5py.Dataset, int, int]:
        """Resize the pid dataset to accomodate the new pids."""
        current_pid_len = pid_dset.shape[0]
        new_pid_len = current_pid_len + len(pids)
        pid_dset.resize((new_pid_len,))
        return pid_dset, current_pid_len, new_pid_len
    
    def initialize(self, hidden_dim:int)->None:
        """Initialize datasets within the HDF5 file."""
        super().initialize(hidden_dim)
        
        with h5py.File(self.output_path, 'a') as f:
            if self.pid_dataset_name not in f:
                dt = h5py.special_dtype(vlen=str)
                f.create_dataset(self.pid_dataset_name, shape=(0,), maxshape=(None,), dtype=dt)
class BaseHDF5Reader:
    def __init__(self, input_path:str, encodings_ds_name:str="X", 
                 target_ds_name:str="y"):
        self.input_path = input_path
        self.encodings_dataset_name = encodings_ds_name
        self.target_dataset_name = target_ds_name
    def read_arrays(self, start_idx:int=None, end_idx:int=None
                    )->Tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(self.input_path, 'r') as f:
            Xds = f[self.encodings_dataset_name]
            yds = f[self.target_dataset_name]                
            return Xds[start_idx:end_idx], yds[start_idx:end_idx] 

class PatientHDF5Reader(BaseHDF5Reader):
    """Reads a encoded patients (pids and tensor of bsxhiddden_dim) from HDF5 file."""
    def __init__(self, input_path:str, encodings_ds_name:str="X", 
                 pid_ds_name:str="pids", target_ds_name:str="y"):
        super().__init__(input_path, encodings_ds_name, target_ds_name)        
        self.pid_dataset_name = pid_ds_name

    def read_pids(self, start_idx:int=None, end_idx:int=None)->List[str]:
        """Read a slice of the patient IDs dataset."""
        with h5py.File(self.input_path, 'r') as f:
            pid_dset = f[self.pid_dataset_name]
            return [pid.decode('utf-8') for pid in pid_dset[start_idx:end_idx]]

def save_hdf5(dic:dict, path:str)->None:
    """Simple method to save a dictionary to hdf5"""
    with h5py.File(path, 'w') as f:
        for key, value in dic.items():
            f.create_dataset(key, data=value)

def load_hdf5(path:str)->dict:
    """Simple method to load a dictionary from hdf5"""
    data_loaded = {}
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            data_loaded[key] = f[key][()]
    return data_loaded

