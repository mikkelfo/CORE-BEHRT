import random
from os.path import join, split
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset


class BaseDataset(Dataset):
    def __init__(self, features: dict):
        self.features = features
        self.max_segments = self.get_max_segments()

    def __len__(self):
        return len(self.features['concept'])

    def __getitem__(self, index):
        return {key: torch.tensor(values[index]) for key, values in self.features.items()}

    def get_max_segments(self):
        if 'segment' not in self.features:
            return None
        return max([max(segment) for segment in self.features['segment']]) + 1


class MLMDataset(BaseDataset):
    def __init__(self, data_dir: str, mode: str, vocabulary=None, masked_ratio=0.3, ignore_special_tokens=True,  **kwargs):
        
        features = torch.load(join(data_dir, 'tokenized', f'tokenized_{mode}.pt'))
        super().__init__(features)
        if isinstance(vocabulary, type(None)):
            vocabulary = torch.load(join(data_dir, 'vocabulary.pt'))
        self.vocabulary = vocabulary
        self.masked_ratio = masked_ratio
        if ignore_special_tokens:
            self.n_special_tokens = len([token for token in vocabulary if token.startswith('[')])
        else:
            self.n_special_tokens = 0
        self.pids = torch.load(join(data_dir, f'{mode}_pids.pt'))
        
    def __getitem__(self, index):
        patient = super().__getitem__(index)

        masked_concepts, target = self._mask(patient)
        patient['concept'] = masked_concepts
        patient['target'] = target
    
        return patient

    def _mask(self, patient: dict):
        concepts = patient['concept']

        N = len(concepts)

        # Initialize
        masked_concepts = torch.clone(concepts)
        target = torch.ones(N, dtype=torch.long) * -100

        # Apply special token mask and create MLM mask
        eligible_mask = masked_concepts >= self.n_special_tokens
        eligible_concepts = masked_concepts[eligible_mask]      # Ignore special tokens
        rng = torch.rand(len(eligible_concepts))                # Random number for each token
        masked = rng < self.masked_ratio                        # Mask tokens with probability masked_ratio

        # Get masked MLM concepts
        selected_concepts = eligible_concepts[masked]           # Select set % of the tokens
        adj_rng = rng[masked].div(self.masked_ratio)            # Fix ratio to 0-100 interval

        # Operation masks
        rng_mask = adj_rng < 0.8                                # 80% - Mask token
        rng_replace = (0.8 <= adj_rng) & (adj_rng < 0.9)        # 10% - replace with random word
        # rng_keep = adj_rng >= 0.9                             # 10% - keep token (Redundant)

        # Apply operations (Mask, replace, keep)
        selected_concepts = torch.where(rng_mask, self.vocabulary['[MASK]'], selected_concepts) # Replace with [MASK]
        selected_concepts = torch.where(rng_replace, torch.randint(self.n_special_tokens, len(self.vocabulary), (len(selected_concepts),)), selected_concepts) # Replace with random word
        # selected_concepts = torch.where(rng_keep, selected_concepts, selected_concepts)       # Redundant

        # Update outputs (nonzero for double masking)
        target[eligible_mask.nonzero()[:,0][masked]] = eligible_concepts[masked]    # Set "true" token
        masked_concepts[eligible_mask.nonzero()[:,0][masked]]= selected_concepts    # Sets new concepts

        return masked_concepts, target

    @staticmethod
    def load_vocabulary(vocabulary):
        if isinstance(vocabulary, str):
            return torch.load(vocabulary)
        elif isinstance(vocabulary, dict):
            return vocabulary
        else:
            raise TypeError(f'Unsupported vocabulary input {type(vocabulary)}')
    
    def save_vocabulary(self, run_folder: str):
        torch.save(self.vocabulary, join(run_folder, 'vocabulary.pt'))

class MLMLargeDataset(IterableDataset):
    def __init__(self, data_dir:str, mode:str, **kwargs):
        """Initializes the dataset for masked language modeling
        mode is one of 'train', 'val' or 'test'"""
        self.kwargs = kwargs
        self.mode = mode
        self.data_dir = data_dir
        
        self.file_ids = self.get_file_ids()
        if self.kwargs.get('seed'):
            random.seed(self.kwargs['seed'])
            np.random.seed(self.kwargs['seed'])

        if not self.kwargs.get('num_patients'):
            self.pids = torch.load(join(self.data_dir, f'{mode}_pids.pt'))
            self.num_patients = len(self.pids)

        else:
            self.num_patients = self.kwargs['num_patients']
            self.pid_files = self.get_pid_files(self.file_ids)
            np.random.shuffle(self.pid_files)
            self.pids, self.file_ids = self.load_selected_pids(self.num_patients)

        self.data_files = self.get_data_files(self.file_ids)
        self.vocabulary = torch.load(join(data_dir, 'vocabulary.pt'))
        
        self.masked_ratio = self.kwargs.get('masked_ratio', 0.3)
        self.batch_size = kwargs.get('batch_size', 32)
        
        if kwargs.get('ignore_special_tokens', True):
            self.n_special_tokens = len([token for token in self.vocabulary if token.startswith('[')])
        else:
            self.n_special_tokens = 0

        
    def __iter__(self):
        patient_count = 0
        for file_name in self.data_files:
            if patient_count >= self.num_patients:
                return
            yield from self.get_patient(file_name) # test!
            patient_count += 1

    def get_patient(self, file_name: str):
        """Loads a single patient from a file"""
        features = torch.load(file_name)
        num_patients = len(features['concept'])
        for patient_index in range(num_patients):
            patient = self.get_patient_dic(features, patient_index)
            masked_concepts, target = self._mask(patient)
            patient['concept'] = masked_concepts
            patient['target'] = target
            yield patient

    def get_patient_dic(self, features: Dict, patient_index: int):
        """Get a patient dictionary from a patient index"""
        return {key: torch.tensor(values[patient_index]) for key, values in features.items()}

    def __len__(self):
        """Number of patients in the dataset"""
        return self.num_patients
    
    def get_file_ids(self):
        """Returns the file ids for the given mode"""
        return torch.load(join(self.data_dir, f'{self.mode}_file_ids.pt'))
    
    def get_data_files(self, file_ids):
        """Returns the data files for the given mode"""
        return [join(self.data_dir, 'tokenized', f'tokenized_{self.mode}_{file_id}.pt') for file_id in file_ids]
    
    def get_pid_files(self, file_ids):
        return [join(self.data_dir, 'features', f'pids_features_{file_id}.pt') for file_id in file_ids]

    def load_selected_pids(self, num_patients):
        """Loads the selected patient IDs from the files"""
        selected_pids = []
        selected_file_ids = []
        for pid_file_name in self.pid_files:
            file_id = split(pid_file_name)[-1].split('_')[-1][:-3]
            selected_file_ids.append(file_id)
            pids = torch.load(pid_file_name)
            selected_pids.extend(pids[:num_patients - len(selected_pids)])
            if len(selected_pids) >= num_patients:
                break
        return selected_pids, selected_file_ids

    def _mask(self, patient: dict):
        concepts = patient['concept']
        N = len(concepts)

        # Initialize
        masked_concepts = torch.clone(concepts)
        target = torch.ones(N, dtype=torch.long) * -100
        # Apply special token mask and create MLM mask
        eligible_mask = masked_concepts >= self.n_special_tokens
        eligible_concepts = masked_concepts[eligible_mask]      # Ignore special tokens
        rng = torch.rand(len(eligible_concepts))                # Random number for each token
        masked = rng < self.masked_ratio                        # Mask tokens with probability masked_ratio

        # Get masked MLM concepts
        selected_concepts = eligible_concepts[masked]           # Select set % of the tokens
        adj_rng = rng[masked].div(self.masked_ratio)            # Fix ratio to 0-100 interval

        # Operation masks
        rng_mask = adj_rng < 0.8                                # 80% - Mask token
        rng_replace = (0.8 <= adj_rng) & (adj_rng < 0.9)        # 10% - replace with random word
        # rng_keep = adj_rng >= 0.9                             # 10% - keep token (Redundant)
        # Apply operations (Mask, replace, keep)
        selected_concepts = torch.where(rng_mask, self.vocabulary['[MASK]'], selected_concepts) # Replace with [MASK]
        selected_concepts = torch.where(rng_replace, torch.randint(self.n_special_tokens, len(self.vocabulary), (len(selected_concepts),)), selected_concepts) # Replace with random word
        # selected_concepts = torch.where(rng_keep, selected_concepts, selected_concepts)       # Redundant
        # Update outputs (nonzero for double masking)
        target[eligible_mask.nonzero()[:,0][masked]] = eligible_concepts[masked]    # Set "true" token
        masked_concepts[eligible_mask.nonzero()[:,0][masked]]= selected_concepts    # Sets new concepts

        return masked_concepts, target

    def save_vocabulary(self, run_folder: str):
        torch.save(self.vocabulary, join(run_folder, 'vocabulary.pt'))


class HierarchicalDataset(MLMDataset):
    def __init__(
        self,
        data_dir,
        mode,
        masked_ratio=0.3,
        ignore_special_tokens=True,
        tree=None,
        tree_matrix=None,
    ):
        
        super().__init__(data_dir, mode, masked_ratio=masked_ratio, ignore_special_tokens=ignore_special_tokens)
        
        hierarchical_dir = join(data_dir, 'hierarchical')
        self.h_vocabulary = torch.load(join(hierarchical_dir, 'vocabulary.pt'))
        if isinstance(tree_matrix, type(None)):
            tree_matrix = torch.load(join(hierarchical_dir, 'tree_matrix.pt'))
        if isinstance(tree, type(None)):
            tree = torch.load(join(hierarchical_dir, 'tree.pt'))
        self.tree = tree
        self.tree_matrix = tree_matrix
        
        self.tree_matrix_sparse = self.tree_matrix.to_sparse()
        self.leaf_counts = self.tree.get_leaf_counts()
        self.target_mapping = self.get_target_mapping()

    def get_target_mapping(self):
        """Target mapping with the vocabulary used for tokenization."""
        target_mapping_temp = {
            self.h_vocabulary[k]: v for k, v in self.tree.create_target_mapping().items()
        }  
        return {
           self.vocabulary[k]:target_mapping_temp[self.h_vocabulary[k]] for k, v in self.vocabulary.items() if self.h_vocabulary[k] in target_mapping_temp
        }
        
    def __getitem__(self, index):
        patient = super().__getitem__(index)
        target_mask = patient["target"] != -100
        patient["attention_mask"] = target_mask

        patient["target"] = self._hierarchical_target(patient["target"][target_mask])

        return patient

    def _hierarchical_target(self, target):

        target_levels = torch.tensor(
            [self.target_mapping[t.item()] for t in target]
        )  # Converts target to target for each level
        return self.expand_to_class_probabilities(
            target_levels
        )  # Converts target for each level to probabilities

    def expand_to_class_probabilities(self, target_levels):
        levels = self.tree_matrix.shape[0]
        seq_len = len(target_levels)
        target_levels = target_levels.view(-1, levels)

        probabilities = torch.zeros(seq_len, levels, len(self.leaf_counts))
        mask = target_levels != -100

        if mask.any():
            # Set "class indices" to 1
            probabilities[mask, target_levels[mask]] = 1

            if (~mask).any():
                last_parents_idx = mask.sum(1) - 1
                seq_class_idx = zip(
                    last_parents_idx, target_levels[range(seq_len), last_parents_idx]
                )  # tuple indices of (class_level, class_idx)

                relevant_leaf_counts = torch.stack(
                    [
                        self.tree_matrix[class_lvl, class_idx] * self.leaf_counts
                        for class_lvl, class_idx in seq_class_idx
                    ]
                )
                relevant_leaf_probs = relevant_leaf_counts / relevant_leaf_counts.sum(
                    1
                ).unsqueeze(-1)

                unknown_targets_idx = zip(
                    *torch.where(~mask)
                )  # tuple indices of (seq_idx, level_idx)

                unknown_probabilities = torch.stack(
                    [
                        torch.matmul(
                            self.tree_matrix_sparse[lvl_idx],
                            relevant_leaf_probs[seq_idx],
                        )
                        for seq_idx, lvl_idx in unknown_targets_idx
                    ]
                )

                probabilities[~mask] = unknown_probabilities

        return probabilities
    
    def save_vocabulary(self, run_folder: str):
        torch.save(self.vocabulary, join(run_folder, 'vocabulary.pt'))
        torch.save(self.h_vocabulary, join(run_folder, 'h_vocabulary.pt'))

class HierarchicalLargeDataset(MLMLargeDataset):
    def __init__(self, data_dir:str, mode:str, **kwargs):
        super().__init__(data_dir, mode, **kwargs)
        self.ignore_index = self.kwargs.get('ignore_index', -100)
        
        self.tree = torch.load(join(data_dir, 'hierarchical', 'tree.pt'))
        self.tree_matrix = torch.load(join(data_dir, 'hierarchical', 'tree_matrix.pt'))
        self.levels = self.tree.get_max_level()
        self.tree_matrix_sparse = self.tree_matrix.to_sparse()
        self.leaf_counts = self.tree.get_leaf_counts()
        self.n_leafs = len(self.leaf_counts)
        self.h_vocabulary = torch.load(join(data_dir, 'hierarchical', 'vocabulary.pt'))
        self.target_mapping = self.get_target_mapping()    # adjusts target mapping to vocabulary

    def get_target_mapping(self):
        target_mapping_temp = {
            self.h_vocabulary[k]: v for k, v in self.tree.create_target_mapping().items()
        }  
        return {
           self.vocabulary[k]:target_mapping_temp[self.h_vocabulary[k]] for k, v in self.vocabulary.items() if self.h_vocabulary[k] in target_mapping_temp
        }

    def get_patient(self, file_name: str):
        features = torch.load(file_name)
        num_patients = len(features['concept'])
        for patient_index in range(num_patients):
            patient = self.get_patient_dic(features, patient_index)
            masked_concepts, target = self._mask(patient)
            patient['concept'] = masked_concepts
            patient['target'] = target
            target_mask = patient['target'] != -100
            patient['attention_mask'] = target_mask

            patient['target'] = self._hierarchical_target(patient['target'][patient['attention_mask']])
            yield patient

    def _hierarchical_target(self, target):
        target_levels = torch.tensor([self.target_mapping[t.item()] for t in target]) # Converts target to target for each level
        return self.expand_to_class_probabilities(target_levels)    # Converts target for each level to probabilities

    def expand_to_class_probabilities(self, target_levels):
        levels = self.tree_matrix.shape[0]
        seq_len = len(target_levels)
        target_levels = target_levels.view(-1, levels)

        probabilities = torch.zeros(seq_len, levels, len(self.leaf_counts))
        mask = target_levels != -100

        if mask.any():
            # Set "class indices" to 1
            probabilities = self.probs_above_target(probabilities, mask, target_levels)

            if (~mask).any():
                probabilities = self.probs_below_target(probabilities, mask, target_levels, seq_len)
                
        return probabilities
    
    def probs_above_target(self, probabilities, mask, target_levels):
        """Sets the probabilities for values up to target level. Simple one-hot encoding."""
        probabilities[mask, target_levels[mask]] = 1
        return probabilities

    def probs_below_target(self, probabilities, mask, target_levels, seq_len):
        """Sets the probabilities for values below target level. Uses the tree matrix to calculate the probabilities based on occurence frequency."""
        last_parents_idx = mask.sum(1)-1
        seq_class_idx = zip(last_parents_idx, target_levels[range(seq_len), last_parents_idx])  # tuple indices of (class_level, class_idx)

        relevant_leaf_counts = torch.stack([self.tree_matrix[class_lvl, class_idx] * self.leaf_counts for class_lvl, class_idx in seq_class_idx])
        relevant_leaf_probs = (relevant_leaf_counts / relevant_leaf_counts.sum(1).unsqueeze(-1))

        unknown_targets_idx = zip(*torch.where(~mask))        # tuple indices of (seq_idx, level_idx)

        unknown_probabilities = torch.stack([torch.matmul(self.tree_matrix_sparse[lvl_idx], relevant_leaf_probs[seq_idx]) for seq_idx, lvl_idx in unknown_targets_idx])

        probabilities[~mask] = unknown_probabilities
        return probabilities

    def save_vocabulary(self, run_folder: str):
        torch.save(self.vocabulary, join(run_folder, 'vocabulary.pt'))
        torch.save(self.h_vocabulary, join(run_folder, 'h_vocabulary.pt'))

    
class CensorDataset(BaseDataset):
    """
        n_hours can be both negative and positive (indicating before/after censor token)
        outcomes is a list of the outcome timestamps to predict
        censor_outcomes is a list of the censor timestamps to use
    """
    def __init__(self, features: dict, outcomes: list, censor_outcomes: list, n_hours: int):
        super().__init__(features)

        self.outcomes = outcomes
        self.censor_outcomes = censor_outcomes
        self.n_hours = n_hours

    def __getitem__(self, index: int) -> dict:
        patient = super().__getitem__(index)
        censor_timestamp = self.censor_outcomes[index]
        patient = self._censor(patient, censor_timestamp)
        patient['target'] = float(pd.notna(self.outcomes[index]))

        return patient

    def _censor(self, patient: dict, event_timestamp: float) -> dict:
        if pd.isna(event_timestamp):
            return patient
        else:
            # Only required when padding
            mask = patient['attention_mask']
            N_nomask = len(mask[mask==1])

            # Remove padding and replace background 0s with first non-background pos
            pos = patient['abspos'][:N_nomask]

            # censor the last n_hours
            dont_censor = (pos - event_timestamp - self.n_hours) <= 0

            # TODO: This removes padding as well - is this ok?
            for key, value in patient.items():
                patient[key] = value[dont_censor]

        return patient

