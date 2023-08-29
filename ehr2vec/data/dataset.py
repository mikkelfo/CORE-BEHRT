import logging
import random
from glob import glob
from os.path import join

logger = logging.getLogger(__name__)  # Get the logger for this module
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset


class BaseEHRDataset(IterableDataset):
    def __init__(self, data_dir, mode, pids=None, num_patients=None, seed=None, vocabulary:dict=None, n_procs=None):  #
        logger.info(f"Initializing {mode} dataset")
        self.data_dir = data_dir
        self.mode = mode
        self.num_patients = num_patients
        self.seed = seed
        self.set_random_seed()
        self.n_procs = n_procs
        self.pid_cache = {}
        
        self.vocabulary = vocabulary or torch.load(join(data_dir, 'vocabulary.pt'))

        self.tokenized_files = self._get_all_tokenized_files()
        self.file_ids = self.get_file_ids_from_files(self.tokenized_files)
        self.pid_files = self.get_pid_files_from_file_ids(self.file_ids)
        self.pids = self._initialize_pids(pids)
        
        self.pid_files = self.get_pid_files_from_file_ids(self.file_ids) # update file ids to only include the selected patients
        self.num_patients = len(self.pids) # Update num_patients to the actual number of patients
        
        self.patient_integer_ids = self._create_patient_integer_ids() # file_id:{patient_index:pid}
        self.tokenized_files = self._update_tokenized_files() # Only keep files with relevant patients

        del self.pid_cache # Free up memory
        logger.info("Done initializing dataset")

    def __iter__(self):
        for file_name in self.tokenized_files:
            yield from self.get_patient(file_name) 

    def __len__(self):
        """Number of patients in the dataset"""
        return self.num_patients
        
    def get_patient(self, file_name: str):
        """Loads a single patient from a file"""
        features = torch.load(file_name)
        for patient_index in self.patient_integer_ids[self.extract_file_id(file_name)]:
            yield self.get_patient_dic(features, patient_index)

    def get_patient_dic(self, features, patient_index):
        return {key: torch.tensor(values[patient_index]) for key, values in features.items()}
    
    def get_file_ids_from_files(self, files):
        """Returns the file ids for the given mode"""
        return [self.extract_file_id(f) for f in files]

    def _initialize_pids(self, provided_pids):
        logger.info("Initializing patient IDs")
        
        if provided_pids and self.num_patients:
            logger.error("Cannot provide both pids and num_patients")
            raise ValueError("Cannot provide pids and num_patients")    
        
        if provided_pids:
            logger.info("Using provided patient IDs")
            return self.filter_pids_update_file_ids(provided_pids)

        if self.num_patients:
            logger.info(f"Using {self.num_patients} random patient IDs")
            return self.load_n_random_pids()
        
        logger.info("Using all patient IDs")
        return torch.load(join(self.data_dir, f'{self.mode}_pids.pt'))

    def load_n_random_pids(self):
        np.random.shuffle(self.pid_files)
        return self.get_n_pids()

    def _process_files(self, func, pids_set):
        if self.n_procs:
            logger.info(f"Using parallel processing with {self.n_procs}")
            with ProcessPoolExecutor() as executor:
                return list(executor.map(func, self.pid_files, [pids_set] * len(self.pid_files)))
        else:
            logger.info("Using sequential processing")
            return [func(file, pids_set) for file in self.pid_files]

    def _find_pids_in_file(self, file, pids_set):
        """
        This function will process each file.
        """
        logger.info("::: Loading file ::: " + file)
        pids_batch = torch.load(file)
        pids_intersection = pids_set.intersection(set(pids_batch))

        if len(pids_intersection) > 0:
            self.pid_cache[file] = pids_batch
            file_id = self.extract_file_id(file)
            return list(pids_intersection), file_id

        return [], None

    def filter_pids_update_file_ids(self, pids):
        """Updates the file ids to only include the selected patients"""
        logger.info("Filtering patient IDs")
        pids_set = set(pids)

        results = self._process_files(self._find_pids_in_file, pids_set)

        new_pids = []
        self.file_ids = [file_id for _, file_id in results if file_id is not None]

        for new_pids_chunk, _ in results:
            new_pids.extend(new_pids_chunk)

        return new_pids
    
    def _enumerate_pids_in_file(self, pid_file, pids_set):
        logger.info("::: Loading file ::: " + pid_file)
        file_id = self.extract_file_id(pid_file)
        pids = self.pid_cache.get(pid_file, torch.load(pid_file))
        int2pid = {i: pid for i, pid in enumerate(pids) if pid in pids_set}
        return file_id, int2pid

    def _create_patient_integer_ids(self)->dict:
        logger.info("Creating patient integer IDs")
        pids_set = set(self.pids)
        
        results = self._process_files(self._enumerate_pids_in_file, pids_set)
        
        file2intpid_dic = {file_id: data for file_id, data in results}

        return file2intpid_dic
    
    def _get_all_tokenized_files(self):
        """Returns the data files for the given mode"""
        return glob(join(self.data_dir, 'tokenized',f'tokenized_{self.mode}_*.pt'))
    
    def _update_tokenized_files(self):
        """Updates the tokenized files to only include the selected patients"""
        return [file for file in self.tokenized_files if self.extract_file_id(file) in self.file_ids]

    def get_pid_files_from_file_ids(self, file_ids):
        return [join(self.data_dir, 'features', f'pids_features_{file_id}.pt') for file_id in file_ids]
    @staticmethod
    def extract_file_id(file_name: str):
        """Returns the file id for a given file name"""
        return int(file_name.split('_')[-1].split('.')[0])

    def get_n_pids(self):
        """Loads the selected patient IDs from the files"""
        selected_pids = []
        self.file_ids = [] # Reset file ids, use only those containing the relevant patients
        for pid_file_name in self.pid_files:
            self.file_ids.append(self.extract_file_id(pid_file_name))
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
    
    def save_vocabulary(self, run_folder: str):
        torch.save(self.vocabulary, join(run_folder, 'vocabulary.pt'))
        if "h_vocabulary" in self:
            torch.save(self.h_vocabulary, join(run_folder, 'h_vocabulary.pt'))
        if "target_mapping" in self:
            torch.save(self.target_mapping, join(run_folder, 'target_mapping.pt'))

class MLMDataset(BaseEHRDataset):
    def __init__(self, data_dir: str, mode: str, vocabulary: dict=None, masked_ratio=0.3, ignore_special_tokens=True, num_patients=None, seed=None):
        super().__init__(data_dir, mode, num_patients=num_patients, seed=seed, vocabulary=vocabulary)
        
        self.masked_ratio = masked_ratio
        if ignore_special_tokens:
            self.n_special_tokens = len([token for token in self.vocabulary if token.startswith('[')])
        else:
            self.n_special_tokens = 0

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


class HierarchicalMLMDataset(MLMDataset):
    """Hierarchical MLM Dataset"""
    def __init__(self, data_dir, mode, masked_ratio=0.3, ignore_special_tokens=True, tree=None, tree_matrix=None,num_patients=None, seed=None, hierarchical_dir='hierarchical'):
        super().__init__(data_dir, mode, masked_ratio=masked_ratio, ignore_special_tokens=ignore_special_tokens, num_patients=num_patients, seed=seed)
        
        hierarchical_dir = join(data_dir, hierarchical_dir)
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
        """
        Creates a mapping between the vocabulary and the target mapping created from the tree.
        
        It first maps the tree's target mapping keys to the corresponding h_vocabulary keys, 
        storing this mapping in target_mapping_temp.

        Then, it goes through each item in vocabulary. If the key is also in h_vocabulary 
        and its corresponding value is in target_mapping_temp, it maps the key from vocabulary 
        to the value in target_mapping_temp.

        Returns:
            A dictionary mapping values of vocabulary to values from the tree's target mapping.
        """
        tree_target_mapping = self.tree.create_target_mapping()
        torch.save(tree_target_mapping, 'tree_target_mapping.pt')
        target_mapping_temp = {
            self.h_vocabulary[k]: v for k, v in tree_target_mapping.items()
        }  
        target_maping = {}
        for vocab_key, vocab_value in self.vocabulary.items():
            h_vocab_value = self.h_vocabulary.get(vocab_key, self.h_vocabulary['[UNK]'])
            target_maping[vocab_value] = target_mapping_temp.get(h_vocab_value)
            
        return target_maping 
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
    
class CensorDataset(BaseEHRDataset):
    """
        n_hours can be both negative and positive (indicating before/after censor token)
        outcomes is a list of the outcome timestamps to predict
        censor_outcomes is a list of the censor timestamps to use
    """
    def __init__(self, data_dir:str, mode:str, outcomes:list, censor_outcomes: list, n_hours: int, outcome_pids: list, num_patients=None, pids=None, seed=None, n_procs=None):
        super().__init__(data_dir, mode, num_patients=num_patients, pids=pids, seed=seed, n_procs=n_procs)
        self.outcomes = outcomes
        self.censor_outcomes = censor_outcomes
        self.n_hours = n_hours
        self.outcome_pids = self.validate_outcome_pids(outcome_pids)

    def get_patient(self, file_name: str):
        """Loads a single patient from a file"""
        features = torch.load(file_name)
        for patient_index, pid in self.patient_integer_ids[self.extract_file_id(file_name)].items():
            patient = self.get_patient_dic(features, patient_index)
            outcome_patient_index = self.outcome_pids.index(pid)
            censor_timestamp = self.censor_outcomes[outcome_patient_index]
            patient = self._censor(patient, censor_timestamp)
            patient['target'] = float(pd.notna(self.outcomes[outcome_patient_index]))
            yield patient

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

    def validate_outcome_pids(self, outcome_pids):
        """All the pids should be contained in outcome pids"""
        if set(self.pids).issubset(set(outcome_pids)):
            return outcome_pids
        else:
            logger.warn('Some patients in the dataset are not contained in the outcome pids. They will be removed.')
            self._filter_pids_by_outcome_pids(outcome_pids)
            return outcome_pids
    
    def _filter_pids_by_outcome_pids(self, outcome_pids):
        outcome_pids = set(outcome_pids)
        logger.info(f"Filtering {len(self.pids)} patients")
        self.pids = [pid for pid in self.pids if pid in outcome_pids]
        for file, int2pid in self.patient_integer_ids.items():
            filtered_dic = {int_ : pid for int_, pid in int2pid.items() if pid in outcome_pids}
            self.patient_integer_ids[file] = filtered_dic 
        logger.info(f"Remaining patients: {len(self.patient_integer_ids)}")
        self.num_patients = len(self.pids)