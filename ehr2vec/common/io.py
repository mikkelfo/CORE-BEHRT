import h5py
import torch
from os.path import join


def save_hdf5(dic, path):
    with h5py.File(path, 'w') as f:
        for key, value in dic.items():
            f.create_dataset(key, data=value)
def load_hdf5(path):
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
        RaiseValueError(f"{acc_method} not implemented")
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