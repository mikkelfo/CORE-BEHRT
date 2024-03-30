from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ehr2vec.common.config import Config
from ehr2vec.common.utils import Data
from ehr2vec.data.filter import PatientFilter


def plot_and_save_hist(tensor_data: torch.Tensor, name: str, split: str, 
                       folder: str, positive_indices: list=None, density=True)->None:
    """Plot and save histogram of tensor_data to folder with name split_name.png.
    name: str: Name of the tensor_data
    split: str: Name of the split (train, val, test)
    folder: str: Folder to save the histogram to
    positive_indices: list: Indices of positive samples in tensor_data
    density: bool: If True, plot density histogram
    """
    fig, ax = plt.subplots()
    if positive_indices:
        bins = np.histogram_bin_edges(tensor_data, bins=50)
        negative_indices = [i for i in range(len(tensor_data)) if i not in positive_indices]
        ax.hist(tensor_data[negative_indices], bins=bins, color='b', alpha=0.5, 
                label='negative', density=density)
        ax.hist(tensor_data[positive_indices], bins=bins, color='r', alpha=0.5, 
                label='positive', density=density)
        ax.legend()
    else:
        ax.hist(tensor_data, bins=50)
    ax.set_xlabel(name)
    ax.set_title(f'{split} {name}')
    fig.savefig(join(folder, f'{split}_{name}.png'), dpi=150, bbox_inches='tight')

def calculate_statistics(tensor_data: torch.Tensor) -> tuple:
    """Calculate mean, standard deviation, median, lower and upper quartiles."""
    mean = round(tensor_data.mean().item(), 4)
    std = round(tensor_data.std().item(), 4)
    median = round(tensor_data.median().item(), 4)
    lower_quartile = round(tensor_data.quantile(0.25).item(), 4)
    upper_quartile = round(tensor_data.quantile(0.75).item(), 4)
    return mean, std, median, lower_quartile, upper_quartile

def get_number_of_women(data: Data)->int:
    temp_cfg = Config({'data':{'gender':'Kvinde'}})
    patient_filter = PatientFilter(temp_cfg)
    female_data = patient_filter.select_by_gender(data)
    return len(female_data)

def count_positves(data):
    """Count the number of non-NaN outcomes in a Data object"""
    return len([i for i, outcome in enumerate(data.outcomes) if pd.notna(outcome)])

def save_gender_distribution(data_dict: dict, folder: str)->None:
    """Save distribution of gender for all splits and positive patients to folder."""
    index = ['n_female', 'perc. female']
    gender_dist, gender_dist_positives = pd.DataFrame(), pd.DataFrame()
    for split, data in data_dict.items():
        n_positives = count_positves(data)
        female_data = data.copy() # copy to avoid changing original data
        n_women = get_number_of_women(female_data)
        n_positives_women = count_positves(female_data)
        gender_dist[split] = [n_women, n_women/len(data) if len(data) > 0 else 0]
        gender_dist_positives[split] = [n_positives_women, n_positives_women/n_positives if n_positives > 0 else 0]
    gender_dist.index = index
    gender_dist_positives.index = index
    gender_dist.to_csv(join(folder, 'gender_dist.csv'), index=True)
    gender_dist_positives.to_csv(join(folder, 'gender_dist_positives.csv'), index=True)