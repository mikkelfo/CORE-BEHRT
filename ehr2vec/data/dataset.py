import random
from os.path import join, split
from glob import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

class BaseEHRDataset(IterableDataset):
    def __init__(self, data_dir, mode, pids=None, num_patients=None, seed=None,):
        self.data_dir = data_dir
        self.mode = mode
        self.num_patients = num_patients
        self.seed = seed
        self.set_random_seed()
        
        self.tokenized_files = self.get_all_tokenized_files()
        self.file_ids = self.get_file_ids()
        self.pid_files = self.get_pid_files()
        self.pids = self.set_pids(pids)
        self.num_patients = len(self.pids)
        self.patient_integer_ids = self.set_patient_integer_ids() # file_id: integer_id. Needed to access the correct patient in the file
        self.tokenized_files = self.update_tokenized_files()

    def __iter__(self):
        for file_name in self.tokenized_files:
            yield from self.get_patient(file_name) 

    def __len__(self):
        """Number of patients in the dataset"""
        return self.num_patients
        
    def get_patient(self, file_name: str):
        """Loads a single patient from a file"""
        features = torch.load(file_name)
        for patient_index in self.patient_integer_ids[self.get_file_id(file_name)]:
            yield self.get_patient_dic(features, patient_index)

    def get_patient_dic(self, features: dict, patient_index: int):
        """Get a patient dictionary from a patient index"""
        return {key: torch.tensor(values[patient_index]) for key, values in features.items()}

    def get_file_ids(self):
        """Returns the file ids for the given mode"""
        return [self.get_file_id(f) for f in self.tokenized_files]

    def set_pids(self, pids):
        if not pids:
            if self.num_patients:
                pids = self.get_n_patients()
            else:
                pids = self.get_all_patients()
        else:
            self.update_file_ids(pids)
        return pids

    def get_n_patients(self):
        np.random.shuffle(self.pid_files)
        return self.get_n_pids()

    def get_all_patients(self):
        return torch.load(join(self.data_dir, f'{self.mode}_pids.pt'))
        
    def update_file_ids(self, pids):
        """Updates the file ids to only include the selected patients"""
        self.file_ids = []
        for file in self.pid_files:
            pids_batch = torch.load(file)
            if set(pids_batch).intersection(set(pids)):
                self.file_ids.append(self.get_file_id(file))
    
    def set_patient_integer_ids(self)->dict:
        patient_integer_ids = {}
        for pid_file in self.pid_files:
            file_id = self.get_file_id(pid_file)
            pids = torch.load(pid_file)
            patient_integer_ids[file_id] = [i for i, pid in enumerate(pids) if pid in self.pids]
        return patient_integer_ids
    
    def get_all_tokenized_files(self):
        """Returns the data files for the given mode"""
        return glob(join(self.data_dir, 'tokenized',f'tokenized_{self.mode}_*.pt'))
    
    def update_tokenized_files(self):
        """Updates the tokenized files to only include the selected patients"""
        return [file for file in self.tokenized_files if self.get_file_id(file) in self.file_ids]

    def get_pid_files(self):
        return [join(self.data_dir, 'features', f'pids_features_{file_id}.pt') for file_id in self.file_ids]

    def get_file_id(self, file_name: str):
        """Returns the file id for a given file name"""
        return int(file_name.split('_')[-1].split('.')[0])

    def get_n_pids(self):
        """Loads the selected patient IDs from the files"""
        selected_pids = []
        self.file_ids = [] # Reset file ids, use only those containing the relevant patients
        for pid_file_name in self.pid_files:
            self.file_ids.append(self.get_file_id(pid_file_name))
            pids = torch.load(pid_file_name)
            selected_pids.extend(pids[:self.num_patients - len(selected_pids)])
            if len(selected_pids) >= self.num_patients:
                break
        return selected_pids

    def set_random_seed(self):
        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

class MLMDataset(BaseEHRDataset):
    def __init__(self, data_dir: str, mode: str, vocabulary: dict=None, masked_ratio=0.3, ignore_special_tokens=True, num_patients=None, seed=None):
        super().__init__(data_dir, mode, num_patients=num_patients, seed=seed)
        
        if isinstance(vocabulary, type(None)):
            vocabulary = torch.load(join(data_dir, 'vocabulary.pt'))
        self.vocabulary = vocabulary
        self.masked_ratio = masked_ratio
        if ignore_special_tokens:
            self.n_special_tokens = len([token for token in vocabulary if token.startswith('[')])
        else:
            self.n_special_tokens = 0
        self.pids = torch.load(join(data_dir, f'{mode}_pids.pt'))

    def get_patient(self, file_name: str):
        """Loads a single patient from a file"""
        for patient in super().get_patient(file_name):
            masked_concepts, target = self._mask(patient)
            patient['concept'] = masked_concepts
            patient['target'] = target
            yield patient

    def _mask(self, patient: dict):
        """Masks a patient's concepts and create a target according to the MLM procedure"""
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
        mask_tokens_percentage = 0.8
        replace_tokens_percentage = 0.1
        rng_mask = adj_rng < mask_tokens_percentage             # 80% - Mask token
        rng_replace = (mask_tokens_percentage <= adj_rng) & (adj_rng < (mask_tokens_percentage+replace_tokens_percentage)) # 10% - replace with random word, remaining 10% are kept

        # Apply operations (Mask, replace, keep)
        selected_concepts = torch.where(rng_mask, self.vocabulary['[MASK]'], selected_concepts) # Replace with [MASK]
        selected_concepts = torch.where(rng_replace, torch.randint(self.n_special_tokens, len(self.vocabulary), (len(selected_concepts),)), selected_concepts) # Replace with random word

        # Update outputs (nonzero for double masking)
        target[eligible_mask.nonzero()[:,0][masked]] = eligible_concepts[masked]    # Set "true" token
        masked_concepts[eligible_mask.nonzero()[:,0][masked]]= selected_concepts    # Sets new concepts

        return masked_concepts, target

    def save_vocabulary(self, run_folder: str):
        torch.save(self.vocabulary, join(run_folder, 'vocabulary.pt'))
        if "h_vocabulary" in self:
            torch.save(self.h_vocabulary, join(run_folder, 'h_vocabulary.pt'))
        if "target_mapping" in self:
            torch.save(self.target_mapping, join(run_folder, 'target_mapping.pt'))

class HierarchicalMLMDataset(MLMDataset):
    """Hierarchical MLM Dataset"""
    def __init__(self, data_dir, mode, masked_ratio=0.3, ignore_special_tokens=True, tree=None, tree_matrix=None,num_patients=None, seed=None, ):
        super().__init__(data_dir, mode, masked_ratio=masked_ratio, ignore_special_tokens=ignore_special_tokens, num_patients=num_patients, seed=seed)
        
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
        self.levels = self.tree_matrix.shape[0]
    
    def get_patient(self, file_name: str):
        for patient in super().get_patient(file_name):
            target_mask = patient['target'] != -100
            patient['attention_mask'] = target_mask

            patient['target'] = self._hierarchical_target(patient['target'][patient['attention_mask']])
            yield patient

    def get_target_mapping(self):
        """Target mapping with the vocabulary used for tokenization."""
        target_mapping_temp = {
            self.h_vocabulary[k]: v for k, v in self.tree.create_target_mapping().items()
        }  
        return {
           self.vocabulary[k]:target_mapping_temp[self.h_vocabulary[k]] for k, v in self.vocabulary.items() if self.h_vocabulary[k] in target_mapping_temp
        }
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
    
class CensorDataset():
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

