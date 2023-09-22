import logging
import os
from dataclasses import dataclass

from os.path import join
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from data.dataset import (BinaryOutcomeDataset, HierarchicalMLMDataset,
                          MLMDataset)
from data_fixes.adapt import BehrtAdapter
from data_fixes.censor import Censorer
from data_fixes.truncate import Truncator
from transformers import BertConfig

logger = logging.getLogger(__name__)  # Get the logger for this module

PID_KEY = 'PID'
VOCABULARY_FILE = 'vocabulary.pt'
MALE_KEY = 'BG_GENDER_M'
FEMALE_KEY = 'BG_GENDER_F'

def load_model(model_class, cfg, add_config={}):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = join(cfg.paths.model_path, 'checkpoints' ,f'checkpoint_epoch{cfg.paths.checkpoint_epoch}_end.pt')
    # Load the config from file
    config = BertConfig.from_pretrained(cfg.paths.model_path) 
    config.update(add_config)
    model = model_class(config)
    load_result = model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'], strict=False)
    logger.info("missing state dict keys: %s", load_result.missing_keys)
    return model

def create_binary_outcome_datasets(all_outcomes, cfg):
    """
    This function is used to create outcome datasets based on the configuration provided.
    """
    outcomes, censor_outcomes, pids = DatasetPreparer._retrieve_outcomes(all_outcomes, cfg)
    if cfg.get("encode_pos_only", False):
        outcomes, censor_outcomes, pids = select_positives(outcomes, censor_outcomes, pids)
        cfg.train_data.num_patients = None
        cfg.val_data.num_patients = None
    if cfg.train_data.num_patients == 0:
        train_dataset = None
    else:
        train_dataset = BinaryOutcomeDataset(cfg.paths.data_path, 'train', outcomes, 
                                    censor_outcomes=censor_outcomes, 
                                    outcome_pids=pids,
                                    num_patients=cfg.train_data.num_patients,
                                    pids=pids if cfg.get("encode_pos_only", False) else None,
                                    n_hours=cfg.outcome.n_hours,
                                    n_procs=cfg.train_data.get('n_procs', None))

    if cfg.val_data.num_patients == 0:
        val_dataset = None
    else:
        val_dataset = BinaryOutcomeDataset(cfg.paths.data_path, 'val',  outcomes, 
                                    censor_outcomes=censor_outcomes, 
                                    outcome_pids=pids,
                                    num_patients=cfg.val_data.num_patients,
                                    pids=pids if cfg.get("encode_pos_only", False) else None, 
                                    n_hours=cfg.outcome.n_hours,
                                    n_procs=cfg.val_data.get('n_procs', None),
                                    )
    
    return train_dataset, val_dataset, outcomes

def select_positives(outcomes, censor_outcomes, pids):
    """Select only positive outcomes."""
    logger.info("Selecting only positive outcomes")
    select_indices = set([i for i, outcome in enumerate(outcomes) if pd.notna(outcome)])
    outcomes = [outcomes[i] for i in select_indices]
    censor_outcomes = [censor_outcomes[i] for i in select_indices]
    pids = [pids[i] for i in select_indices]
    return outcomes, censor_outcomes, pids

# TODO: Add option to load test set only!

@dataclass
class Data:
    features: Dict
    pids: List
    outcomes: Optional[List] = None
    censor_outcomes: Optional[List] = None
    vocabulary: Optional[Dict] = None
    mode: Optional[str] = None
    
    def __len__(self):
        return len(self.pids)

    def check_lengths(self):
        """Check that all features have the same length"""
        for key, values in self.features.items():
            assert len(values) == len(self.pids), f"Length of {key} does not match length of pids"
        if self.outcomes is not None:
            assert len(self.outcomes) == len(self.pids), "Length of outcomes does not match length of pids"
        if self.censor_outcomes is not None:
            assert len(self.censor_outcomes) == len(self.pids), "Length of censor outcomes does not match length of pids"

    def pipe(self, func, *args, **kwargs):
        """Apply a function to the data instance and returns the result"""
        return func(self, *args, **kwargs)

class DatasetPreparer:

    def __init__(self, cfg):
        self.cfg = cfg
        self.run_folder = join(self.cfg.paths.output_path, self.cfg.paths.run_name)
        os.makedirs(self.run_folder, exist_ok=True)

    def prepare_mlm_dataset(self):
        """Load data, truncate, adapt features, create dataset"""
        train_features, val_features, vocabulary = self._prepare_mlm_features()
        train_dataset = MLMDataset(train_features, vocabulary, **self.cfg.data.dataset)
        val_dataset = MLMDataset(val_features, vocabulary, **self.cfg.data.dataset)
        
        return train_dataset, val_dataset
    
    def prepare_mlm_dataset_for_behrt(self):
        """Load data, truncate, adapt features, create dataset"""
        train_features, val_features, vocabulary = self._prepare_mlm_features()
        train_features = BehrtAdapter().adapt_features(train_features)
        val_features = BehrtAdapter().adapt_features(val_features)
        train_dataset = MLMDataset(train_features, vocabulary, **self.cfg.data.dataset)
        val_dataset = MLMDataset(val_features, vocabulary, **self.cfg.data.dataset)
        
        return train_dataset, val_dataset

    def prepare_hmlm_dataset(self):
        train_features, val_features, vocabulary = self._prepare_mlm_features()
        tree, tree_matrix, h_vocabulary = self._load_tree()
        
        torch.save(h_vocabulary, join(self.run_folder, 'h_vocabulary.pt'))

        train_dataset = HierarchicalMLMDataset(train_features, vocabulary, 
                                            h_vocabulary, tree, tree_matrix, 
                                            **self.cfg.data.dataset)
        val_dataset = HierarchicalMLMDataset(val_features, vocabulary, 
                                            h_vocabulary, tree, tree_matrix, 
                                            **self.cfg.data.dataset)
        return train_dataset, val_dataset
    
    def prepare_finetune_dataset(self):
        train_features, val_features, train_outcome, val_outcome = self.prepare_finetune_features()
        # Censoring
        train_dataset = BinaryOutcomeDataset(train_features, outcomes=train_outcome)
        val_dataset = BinaryOutcomeDataset(val_features, outcomes=val_outcome)
        return train_dataset, val_dataset
    
    def prepare_finetune_dataset_for_behrt(self):
        train_features, val_features, train_outcome, val_outcome = self.prepare_finetune_features()

        # Adapt features for Behrt
        train_features = BehrtAdapter().adapt_features(train_features)
        val_features = BehrtAdapter().adapt_features(val_features)
        # 8. Censoring
        train_dataset = BinaryOutcomeDataset(train_features, outcomes=train_outcome)
        val_dataset = BinaryOutcomeDataset(val_features, outcomes=val_outcome)
        return train_dataset, val_dataset

    def prepare_finetune_features(self):
        """
        Prepare the features for fine-tuning. 
        The process includes:
        1. Loading tokenized data
        2. Excluding pretrain patients
        3. Optional: Select patients by age
        4. Optional: Select male/female
        5. Loading and processing outcomes
        6. Optional: Select only patients with a censoring outcome
        7. Data censoring
        8. Optional: Select a random subset based on provided number of patients,
        9. Optional: Remove background tokens
        10. Truncation
        
        """
        data_cfg = self.cfg.data
        paths_cfg = self.cfg.paths

        # 1. Loading tokenized data
        train_features, train_pids, val_features, val_pids, vocabulary = self._load_tokenized_data()
        train_data = Data(train_features, train_pids, vocabulary=vocabulary, mode='train')
        val_data = Data(val_features, val_pids, vocabulary=vocabulary, mode='val')

        # 2. Excluding pretrain patients
        train_data = self._exclude_pretrain_patients(train_data)
        val_data = self._exclude_pretrain_patients(val_data)
        self.log_patient_nums('excluding pretrain patients', train_data, val_data)
       
        # 3. Patient selection
        if data_cfg.get('num_train_patients', False) or data_cfg.get('num_val_patients', False):
            train_data = self._select_random_subset(train_data, data_cfg.num_train_patients)
            val_data = self._select_random_subset(val_data, data_cfg.num_val_patients)
            self.log_patient_nums('selecting a random subset', train_data, val_data)
        
        # 4. Optinally select gender group
        if data_cfg.get('gender', False):
            train_data = self._select_by_gender(train_data)
            val_data = self._select_by_gender(val_data)
            self.log_patient_nums('selecting by gender', train_data, val_data)
        
        # 5. Loading and processing outcomes
        logger.info('Load outcomes')
        outcomes = torch.load(paths_cfg.outcome)
        censor_outcomes = torch.load(paths_cfg.censor) if paths_cfg.get('censor', False) else outcomes   

        logger.info("Assigning outcomes to data")
        train_data = self._retrieve_and_assign_outcomes(train_data, outcomes, censor_outcomes)
        val_data = self._retrieve_and_assign_outcomes(val_data, outcomes, censor_outcomes)

        # 6. Optionally select patients of interest
        if data_cfg.get("select_censored", False):
            train_data = self._select_censored(train_data)
            val_data = self._select_censored(val_data)
            self.log_patient_nums('selecting censored patients', train_data, val_data)

        # 7. Filter patients with outcome before censoring
        train_data = self._filter_outcome_before_censor(train_data)
        val_data = self._filter_outcome_before_censor(val_data)
        self.log_patient_nums('filtering outcome before censor', train_data, val_data)

        # 8. Data censoring
        train_data = self._censor_data(train_data)
        val_data = self._censor_data(val_data)
        self.log_patient_nums('censoring/excluding short sequences', train_data, val_data)

        # 9. Select Patients By Age
        if data_cfg.get('min_age', False) or data_cfg.get('max_age', False):
            train_data = self._select_by_age(train_data)
            val_data = self._select_by_age(val_data)
            self.log_patient_nums('selecting by age', train_data, val_data)
        
        # 10. Optionally Remove Background Tokens
        if data_cfg.get("remove_background", False):
            logger.info("Removing background tokens")
            train_data = self._remove_background(train_data)
            val_data = self._remove_background(val_data)

        # 11. Truncation
        logger.info('Truncating')
        truncator = Truncator(max_len=data_cfg.truncation_len, sep_token=vocabulary['[SEP]'])
        train_data.features = truncator(train_data.features)
        val_data.features = truncator(val_data.features)

        train_data.check_lengths()
        val_data.check_lengths()
        # ! outcome before censor outcome??
        logger.info(f"Positive train patients: {len([t for t in train_data.outcomes if not pd.isna(t)])}")
        logger.info(f"Positive val patients: {len([t for t in val_data.outcomes if not pd.isna(t)])}")

        self._save_pids(train_data.pids, val_data.pids) 

        return train_data.features, val_data.features, train_data.outcomes, val_data.outcomes
    

    # def prepare_onehot_dataset(self):
    #     train_features, train_pids, val_features, val_pids, vocabulary = self._load_tokenized_data()
    #     # extend by test!
    #     censorer = Censorer(n_hours=cfg.outcomes.n_hours, min_len=cfg.excluder.min_len, vocabulary=vocabulary)
    #     outcomes = torch.load(join(cfg.paths.data_path, cfg.paths.outcome))
    #     tokenized_features_train = censorer(tokenized_features_train, outcome, vocabulary, cfg)
    #     pass

    def _select_censored(self, data):
        """Select only censored patients"""
        kept_indices = [i for i, censor in enumerate(data.censor_outcomes) if not pd.isna(censor)]
        return self._select_entries(data, kept_indices)

    def _censor_data(self, data):
        """Censors data and removes patients with no data left"""
        censorer = Censorer(self.cfg.outcome.n_hours, vocabulary=data.vocabulary)
        _, kept_indices = censorer(data.features, data.censor_outcomes)
        return self._select_entries(data, kept_indices)

    def _retrieve_and_assign_outcomes(self, data: Data, outcomes: Dict, censor_outcomes: Dict)->Data:
        """Retrieve outcomes and assign them to the data instance"""
        outcomes = self._select_and_order_outcomes_for_patients(outcomes, data.pids)
        censor_outcomes = self._select_and_order_outcomes_for_patients(censor_outcomes, data.pids)
        assert outcomes[PID_KEY] == data.pids, "PIDs in outcome file and data file do not match"
        assert censor_outcomes[PID_KEY] == data.pids, "PIDs in censoring and data files do not match" 
        outcomes_group, censor_outcomes_group = self._retrieve_outcomes(outcomes, censor_outcomes)
        data.outcomes = outcomes_group
        data.censor_outcomes = censor_outcomes_group
        return data
        
    def _prepare_mlm_features(self):
        """Load data, truncate"""

        train_features, train_pids, val_features, val_pids, vocabulary = self._load_tokenized_data()
        torch.save(vocabulary, join(self.run_folder, VOCABULARY_FILE))
        train_data = Data(train_features, train_pids, vocabulary=vocabulary)
        val_data = Data(val_features, val_pids, vocabulary=vocabulary)
        # Patient Selection
        if self.cfg.data.get('num_train_patients', False) or self.cfg.data.get('num_val_patients', False):
            train_data = self._select_random_subset(train_data, self.cfg.data.num_train_patients)
            val_data = self._select_random_subset(val_data, self.cfg.data.num_val_patients)
            self.log_patient_nums('selecting a random subset', train_data, val_data)

        if self.cfg.data.get("remove_background", False):
            train_data = self._remove_background(train_data)
            val_data = self._remove_background(val_data)
            self.log_patient_nums('removing background tokens', train_data, val_data)
        
        # Truncation
        logger.info(f"Truncating data to {self.cfg.data.truncation_len} tokens")
        truncator = Truncator(max_len=self.cfg.data.truncation_len, 
                              sep_token=vocabulary['[SEP]'])
        train_data.features = truncator(train_data.features)
        val_data.features = truncator(val_data.features)

        train_data.check_lengths()
        val_data.check_lengths()
        self._save_pids(train_data.pids, val_data.pids)

        return train_data.features, val_data.features, train_data.vocabulary

    def _remove_background(self, data: Data)->Data:
        """Remove background tokens from features and the first sep token following it"""
        background_tokens = set([v for k, v in data.vocabulary.items() if k.startswith('BG_')])
        example_concepts = data.features['concept'][0] # Assume that all patients have the same background length
        remove_indices = [i for i, concept in enumerate(example_concepts) if concept in background_tokens]
        if data.vocabulary['[SEP]'] in example_concepts:
            remove_indices.append(len(remove_indices)+1)
        first_index, last_index = min(remove_indices), max(remove_indices)
        for k, token_lists in data.features.items():
            data.features[k] = [tokens[:first_index]+tokens[last_index+1:] \
                               for tokens in token_lists]
        return data

    def _load_tokenized_data(self):
        tokenized_dir = self.cfg.paths.get('tokenized_dir', 'tokenized')
        logger.info('Loading tokenized data from %s', tokenized_dir)
        tokenized_data_path = join(self.cfg.paths.data_path, tokenized_dir)
        logger.info("Loading tokenized data train")
        train_features  = torch.load(join(tokenized_data_path, 'tokenized_train.pt'))
        train_pids = torch.load(join(tokenized_data_path,  'pids_train.pt'))
        logger.info("Loading tokenized data val")
        val_features = torch.load(join(tokenized_data_path, 'tokenized_val.pt'))
        val_pids = torch.load(join(tokenized_data_path, 'pids_val.pt'))
        logger.info("Loading vocabulary")
        try:
            vocabulary = torch.load(join(tokenized_data_path, VOCABULARY_FILE))
        except:
            vocabulary = torch.load(join(self.cfg.paths.data_path, VOCABULARY_FILE))
        return train_features, train_pids, val_features, val_pids, vocabulary

    def _load_tree(self):
        hierarchical_path = join(self.cfg.paths.data_path, 
                                 self.cfg.paths.hierarchical_dir)
        tree = torch.load(join(hierarchical_path, 'tree.pt'))
        tree_matrix = torch.load(join(hierarchical_path, 'tree_matrix.pt'))
        h_vocabulary = torch.load(join(hierarchical_path, VOCABULARY_FILE))
        return tree, tree_matrix, h_vocabulary 

    def _exclude_pretrain_patients(self, data: Data):
        pretrain_pids = set(torch.load(join(self.cfg.paths.model_path, f'pids_{data.mode}.pt')))
        kept_indices = [i for i, pid in enumerate(data.pids) if pid not in pretrain_pids]
        return self._select_entries(data, kept_indices)
    
    def _retrieve_outcomes(self, all_outcomes: Dict, all_censor_outcomes: Dict)->Union[List, List]:
        """From the configuration, load the outcomes and censor outcomes."""
        outcomes = all_outcomes.get(self.cfg.outcome.type, [None]*len(all_outcomes[PID_KEY]))
        censor_outcomes = all_censor_outcomes.get(self.cfg.outcome.get('censor_type', None), [None]*len(outcomes))
        return outcomes, censor_outcomes

    def _select_random_subset(self, data, num_patients, seed=0):
        """Select a num_patients random patients"""
        np.random.seed(seed)
        indices = np.arange(len(data.pids))
        np.random.shuffle(indices)
        indices = indices[:num_patients]
        return self._select_entries(data, indices)

    def _select_by_age(self, data: Data)->Data:
        """
        Assuming that age is given in days. 
        We retrieve the last age of each patient and check whether it's within the range.
        """
        kept_indices = []
        min_age = self.cfg.data.get('min_age', 0)
        max_age = self.cfg.data.get('max_age', 120)
        kept_indices = [i for i, ages in enumerate(data.features['age']) 
                if min_age <= ages[-1] <= max_age]
        return self._select_entries(data, kept_indices)

    def _select_by_gender(self, data):
        """Select only patients of a certain gender"""
        gender_token = self._get_gender_token(data.vocabulary, self.cfg.data.gender)
        kept_indices = [i for i, concepts in enumerate(data.features['concept']) if gender_token in set(concepts)]
        return self._select_entries(data, kept_indices)
    
    def _get_gender_token(self, vocabulary, key):
        """Get the token from the vocabulary corresponding to the gender provided in the config"""
        male_list = ['M', 'male', 'Male', 'man', 'MAN']
        female_list = ['W', 'F', 'female', 'woman', 'WOMAN']
        if key in male_list:
            gender_token = vocabulary[MALE_KEY]
        elif key in female_list:
            gender_token = vocabulary[FEMALE_KEY]
        else:
            raise ValueError(f"Unknown gender {key}, please select one of {male_list + female_list}")
        return gender_token
    
    def _filter_outcome_before_censor(self, data: Data)->Data:
        """Filter patients with outcome before censoring"""
        kept_indices = []
        for i, (outcome, censor) in enumerate(zip(data.outcomes, data.censor_outcomes)):
            if (pd.isna(outcome) or pd.isna(censor)) or (outcome <= (censor + self.cfg.outcome.n_hours)):
                kept_indices.append(i)
        return self._select_entries(data, kept_indices)

    @staticmethod
    def _select_entries(data:Data, indices:List)->Data:
        """
        Select entries based on indices. 
        Optionally for outcomes and censor outcomes, if present returns dict of results.
        """
        indices = set(indices)
        data.features = {k: [v[i] for i in indices] for k, v in data.features.items()}
        data.pids = [data.pids[i] for i in indices]
        if data.outcomes is not None:
            data.outcomes = [data.outcomes[i] for i in indices]
        if data.censor_outcomes is not None:
            data.censor_outcomes = [data.censor_outcomes[i] for i in indices]
        return data

    def _save_pids(self, train_pids, val_pids):
        torch.save(train_pids, join(self.run_folder, 'pids_train.pt'))
        torch.save(val_pids, join(self.run_folder, 'pids_val.pt'))
    
    @staticmethod
    def _select_and_order_outcomes_for_patients(all_outcomes: Dict, pids: List) -> Dict:
        """Select outcomes for patients and order them based on the order of pids"""
        # Create a dictionary of positions for each PID for quick lookup
        pid_to_index = {pid: idx for idx, pid in enumerate(all_outcomes[PID_KEY])}
        
        outcome_pids = set(all_outcomes[PID_KEY])

        # Get the order of indices based on pids
        ordered_indices = [pid_to_index[pid] for pid in pids if pid in outcome_pids]

        # order outcomes based on the indices
        return {k: [v[idx] for idx in ordered_indices] for k, v in all_outcomes.items()}

    @staticmethod
    def log_patient_nums(operation:str, train_data:Data, val_data:Data):
        logger.info(f"After {operation}: {len(train_data)} train patients, {len(val_data)} val patients")
