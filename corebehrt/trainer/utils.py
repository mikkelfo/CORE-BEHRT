import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


def get_tqdm(dataloader: DataLoader)->tqdm:
    return tqdm(dataloader, total=len(dataloader))

def compute_avg_metrics(metric_values: dict):
    """Computes the average of the metric values when metric is not zero and not NaN"""
    averages = {}
    for name, values in metric_values.items():
        values_array = np.array(values)
        select_mask = (values_array == 0) | (np.isnan(values_array))
        non_zero_values = values_array[~select_mask]
        
        if non_zero_values.size:
            averages[name] = np.mean(non_zero_values)
        else:
            averages[name] = 0
    return averages

