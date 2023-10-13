import subprocess
import os 
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, precision_recall_curve
import numpy as np
import logging
from tqdm import tqdm
from common.logger import TqdmToLogger


logger = logging.getLogger(__name__)  # Get the logger for this module

def get_nvidia_smi_output()->str:
    try:
        output = subprocess.check_output(['nvidia-smi']).decode('utf-8')
        return output
    except Exception as e:
        return str(e)

def get_tqdm(dataloader: DataLoader)->tqdm:
    return tqdm(dataloader, total=len(dataloader), file=TqdmToLogger(logger) if logger else None)
    
def save_curves(run_folder:str, logits:torch.Tensor, targets:torch.Tensor, epoch:int)->None:
    """Saves the ROC and PRC curves to a csv file in the run folder"""
    roc_name = os.path.join(run_folder, 'checkpoints', f'roc_curve_{epoch}.npz')
    prc_name = os.path.join(run_folder, 'checkpoints', f'prc_curve_{epoch}.npz')
    probas = torch.sigmoid(logits).cpu().numpy()
    fpr, tpr, threshold_roc = roc_curve(targets, probas)
    precision, recall, threshold_pr = precision_recall_curve(targets, probas)
    np.savez_compressed(roc_name, fpr=fpr, tpr=tpr, threshold=threshold_roc)
    np.savez_compressed(prc_name, precision=precision, recall=recall, threshold=np.append(threshold_pr, 1))

def save_metrics_to_csv(run_folder:str, metrics: dict, epoch: int)->None:
    """Saves the metrics to a csv file"""
    metrics_name = os.path.join(run_folder, 'checkpoints', f'validation_scores_{epoch}.csv')
    with open(metrics_name, 'w') as file:
        file.write('metric,value\n')
        for key, value in metrics.items():
            file.write(f'{key},{value}\n')

def compute_avg_metrics(metric_values: dict):
    """Computes the average of the metric values when metric is not zero and not NaN"""
    averages = {}
    for name, values in metric_values.items():
        values_array = np.array(values)
        select_mask = (values_array == 0) | (np.isnan(values_array))
        if select_mask.sum() > 0:
            logger.info(f'Warning: {select_mask.sum()} NaN or zero values for metric {name}')
        non_zero_values = values_array[~select_mask]
        
        if non_zero_values.size:
            averages[name] = np.mean(non_zero_values)
        else:
            averages[name] = 0
    return averages

