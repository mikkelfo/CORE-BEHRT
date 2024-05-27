import os
from os.path import join

import pandas as pd
import torch

from typing import Dict
from corebehrt.common.utils import Data

VOCABULARY_FILE = 'vocabulary.pt'

class Saver:
    """Save features, pids, vocabulary and sequence lengths to a folder"""
    def __init__(self, run_folder) -> None:
        self.run_folder = run_folder
        os.makedirs(self.run_folder, exist_ok=True)
    
    def save_sequence_lengths(self, data: Data)->Data:
        """Save sequence lengths to a file. If outcomes are present, save lengths for positive and negative patients separately."""
        if not data.outcomes:
            sequence_lens = torch.tensor([len(concepts) for concepts in data.features['concept']])
            torch.save(sequence_lens, join(self.run_folder, f'sequences_lengths_{data.mode}.pt'))
            return data
        else:
            pos_indices = set([i for i, outcome in enumerate(data.outcomes) if not pd.isna(outcome)])
            sequence_lens_neg = torch.tensor([len(concepts) for i, concepts in enumerate(data.features['concept']) if i in pos_indices])
            sequence_lens_pos = torch.tensor([len(concepts) for i, concepts in enumerate(data.features['concept']) if i not in pos_indices])
            torch.save(sequence_lens_neg, join(self.run_folder, f'sequences_lengths_{data.mode}_neg.pt'))
            torch.save(sequence_lens_pos, join(self.run_folder, f'sequences_lengths_{data.mode}_pos.pt'))
            return data
    
    def save_train_val_pids(self, train_pids: list, val_pids: list)->None:
        """Save train and val pids to a file."""
        torch.save(train_pids, join(self.run_folder, 'pids_train.pt'))
        torch.save(val_pids, join(self.run_folder, 'pids_val.pt'))

    def save_patient_nums(self, train_data: Data=None, val_data: Data=None, folder:str=None)->None:
        """Save patient numbers for train val including the number of positive patients to a csv file"""
        train_df = pd.DataFrame({'train': [len(train_data), len([t for t in train_data.outcomes if not pd.isna(t)])]}, 
                                index=['total', 'positive'])
        val_df = pd.DataFrame({'val': [len(val_data), len([t for t in val_data.outcomes if not pd.isna(t)])]},
                              index=['total', 'positive'])
        patient_nums = pd.concat([train_df, val_df], axis=1)
        patient_nums.to_csv(join(
            self.run_folder if folder is None else folder, 'patient_nums.csv'), 
                            index_label='Patient Group')
    
    def save_patient_nums_general(self, data_dic: Dict[str, Data], folder:str=None)->None:
        """Save patient numbers for arbitrary splits to a csv file"""
        patient_nums = pd.DataFrame()
        for key, data in data_dic.items():
            total = len(data)
            positive = len([t for t in data.outcomes if not pd.isna(t)])
            patient_nums[key] = [total, positive]
        patient_nums.set_index(pd.Index(['total', 'positive']), inplace=True)
        patient_nums.to_csv(join(
            self.run_folder if folder is None else folder, 'patient_nums.csv'), 
                            index_label='Patient Group')
        
    def save_data(self, data: Data)->None:
        """Save data (features, pids and outcomes (if present) to run_folder)"""
        torch.save(data.features, join(self.run_folder, 'features.pt'))
        torch.save(data.pids, join(self.run_folder, 'pids.pt'))
        torch.save(data.vocabulary, join(self.run_folder, 'vocabulary.pt'))
        if data.outcomes is not None:
            torch.save(data.outcomes, join(self.run_folder, 'outcomes.pt'))
        if data.censor_outcomes is not None:
            torch.save(data.censor_outcomes, join(self.run_folder, 'censor_outcomes.pt'))
    
    def save_list(self, list_: list, name: str)->None:
        """Save a list to a file"""
        torch.save(list_, join(self.run_folder, name))

    def save_vocab(self, vocabulary, name: str=VOCABULARY_FILE):
        """Save vocabulary to a file"""
        torch.save(vocabulary, join(self.run_folder, name))