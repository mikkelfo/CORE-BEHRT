import os
from os.path import join, split

import h5py
import torch


class BaseHDF5Writer:
    def __init__(self, output_path, encodings_dataset_name="X",
                 targets_dataset_name="y"):
        self.output_path = output_path
        self.encodings_dataset_name = encodings_dataset_name
        self.targets_dataset_name = targets_dataset_name
        self.hidden_dim = None
        self.initialized = False

    def initialize(self, hidden_dim):
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

    def write_tensor(self, tensor, tensor_dset):
        tensor_dset, start, stop = self.resize_tensor_dataset(tensor_dset, tensor)
        tensor_dset[start:stop, :] = tensor.cpu().numpy()
        
    def write_targets(self, targets, targets_dset):
        targets_dset, start, stop = self.resize_target_dataset(targets_dset, targets)
        targets_dset[start:stop] = targets.cpu().numpy()

    def resize_tensor_dataset(self, tensor_dset, tensor):
        current_tensor_len = tensor_dset.shape[0]
        new_tensor_len = current_tensor_len + tensor.shape[0]
        tensor_dset.resize((new_tensor_len, self.hidden_dim))
        return tensor_dset, current_tensor_len, new_tensor_len

    def resize_target_dataset(self, targets_dset, targets):
        current_targets_len = targets_dset.shape[0]
        new_targets_len = current_targets_len + len(targets)
        targets_dset.resize((new_targets_len,)) 
        return targets_dset, current_targets_len, new_targets_len

class ConceptHDF5Writer(BaseHDF5Writer):
    def write(self, tensor, labels=None):
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
    def __init__(self, output_path, encodings_dataset_name="X", pid_dataset_name="pids",
                 targets_dataset_name="y"):
        super().__init__(output_path, encodings_dataset_name, targets_dataset_name)
        self.pid_dataset_name = pid_dataset_name
        
    def write(self, tensor, pids, targets=None):
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

    def write_pids(self, pids, pid_dset):
        pid_dset, start, stop = self.resize_pid_dataset(pid_dset, pids)
        pid_dset[start:stop] = pids

    def resize_pid_dataset(self, pid_dset, pids):
        current_pid_len = pid_dset.shape[0]
        new_pid_len = current_pid_len + len(pids)
        pid_dset.resize((new_pid_len,))
        return pid_dset, current_pid_len, new_pid_len
    
    def initialize(self, hidden_dim):
        """Initialize datasets within the HDF5 file."""
        super().initialize(hidden_dim)
        
        with h5py.File(self.output_path, 'a') as f:
            if self.pid_dataset_name not in f:
                dt = h5py.special_dtype(vlen=str)
                f.create_dataset(self.pid_dataset_name, shape=(0,), maxshape=(None,), dtype=dt)
class BaseHDF5Reader:
    def __init__(self, input_path, encodings_ds_name="X", target_ds_name="y"):
        self.input_path = input_path
        self.encodings_dataset_name = encodings_ds_name
        self.target_dataset_name = target_ds_name
    def read_arrays(self, start_idx=None, end_idx=None):
        with h5py.File(self.input_path, 'r') as f:
            Xds = f[self.encodings_dataset_name]
            yds = f[self.target_dataset_name]                
            return Xds[start_idx:end_idx], yds[start_idx:end_idx] 

class PatientHDF5Reader(BaseHDF5Reader):
    """Reads a encoded patients (pids and tensor of bsxhiddden_dim) from HDF5 file."""
    def __init__(self, input_path, encodings_ds_name="X", pid_ds_name="pids", target_ds_name="y"):
        super().__init__(input_path, encodings_ds_name, target_ds_name)        
        self.pid_dataset_name = pid_ds_name

    def read_pids(self, start_idx=None, end_idx=None):
        """Read a slice of the patient IDs dataset."""
        with h5py.File(self.input_path, 'r') as f:
            pid_dset = f[self.pid_dataset_name]
            return [pid.decode('utf-8') for pid in pid_dset[start_idx:end_idx]]

def save_hdf5(dic, path):
    """Simple method to save a dictionary to hdf5"""
    with h5py.File(path, 'w') as f:
        for key, value in dic.items():
            f.create_dataset(key, data=value)

def load_hdf5(path):
    """Simple method to load a dictionary from hdf5"""
    data_loaded = {}
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            data_loaded[key] = f[key][()]
    return data_loaded


# embeddings
def load_input_target(data_path:str, set_:str, censor_outcome:str, target:str, post:bool, acc_method:str='mean', layers='all'):
    """Load embedding and target"""
    time = 'pre'
    if post:
        time = 'post'
    if acc_method=='mean':
        file_name = f'{set_}_embedding_censor_{censor_outcome}_12h_{time}_{acc_method}.pt'
    elif acc_method=='weighted_sum':
        file_name = f'{set_}_embedding_censor_{censor_outcome}_12h_{time}_{acc_method}_layers_{layers}.pt'
    else:
        raise ValueError(f"{acc_method} not implemented")
    X = torch.load(join(data_path, file_name))
    y = torch.load(join(data_path, f'{set_}_outcomes.pt'))[target]
    return X, y


def get_train_test_for_embeddings(data_path:str, censor_outcome:str, target:str, post:bool, acc_method:str='mean', layers='all', filter_covid=True, test_set_only:bool=False):
    """Loads data and combines train and validation set."""
    X, y = [], [] # concat train and val
    sets = ['train', 'val', 'test']
    if test_set_only:
        sets = ['test']
    for set_ in sets:
        X_temp, y_temp = load_input_target(data_path, set_, censor_outcome, target, post, acc_method, layers)
        y_temp = torch.tensor([1 if isinstance(y, float) else 0 for y in y_temp])
        if filter_covid:
            covid_outcome = torch.load(join(data_path, f'{set_}_outcomes.pt'))['COVID']
            covid_mask = torch.tensor([1 if isinstance(x, float) else 0 for x in covid_outcome]).bool()
            X.append(X_temp[covid_mask])
            y.append(y_temp[covid_mask])
        else:
            X.append(X_temp)
            y.append(y_temp)
        if test_set_only:
            return X[0], y[0]
    # get both train and val set
    X_train = torch.cat(X[:2])
    y_train = torch.cat(y[:2])
    X_test = X[-1]
    y_test = y[-1]
    return X_train, y_train, X_test, y_test