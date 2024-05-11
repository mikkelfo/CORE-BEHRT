
import logging
import os
from os.path import join
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch

from ehr2vec.common.config import Config, instantiate, load_config
from ehr2vec.common.loader import (FeaturesLoader, get_pids_file,
                                   load_and_select_splits, load_exclude_pids)
from ehr2vec.common.saver import Saver
from ehr2vec.common.utils import Data
from ehr2vec.data.dataset import MLMDataset
from ehr2vec.data.filter import CodeTypeFilter, PatientFilter
from ehr2vec.data.utils import Utilities
from ehr2vec.data_fixes.handle import Handler
from ehr2vec.data_fixes.truncate import Truncator

logger = logging.getLogger(__name__)  # Get the logger for this module

PID_KEY = 'PID'
VOCABULARY_FILE = 'vocabulary.pt'

# TODO: Add option to load test set only!
class DatasetPreparer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.utils = Utilities
        self.loader = FeaturesLoader(cfg)
        
        run_folder = join(self.cfg.paths.output_path, self.cfg.paths.run_name)
        self.saver = Saver(run_folder)
        
        self.patient_filter = PatientFilter(cfg)
        self.code_type_filter = CodeTypeFilter(cfg)
        self.data_modifier = DataModifier(cfg)

    def prepare_mlm_dataset(self, val_ratio=0.2):
        """Load data, truncate, adapt features, create dataset"""
        data = self._prepare_mlm_features()
        if 'predefined_splits' in self.cfg.paths:
            train_data, val_data = load_and_select_splits(self.cfg.paths.predefined_splits, data)
        else:
            train_data, val_data = data.split(val_ratio)
        self.saver.save_train_val_pids(train_data.pids, val_data.pids)

        train_dataset = MLMDataset(train_data.features, train_data.vocabulary, **self.cfg.data.dataset)
        val_dataset = MLMDataset(val_data.features, train_data.vocabulary, **self.cfg.data.dataset)

        
        return train_dataset, val_dataset

    def prepare_finetune_data(self) -> Data:
        data_cfg = self.cfg.data

        # 1. Loading tokenized data
        data = self.loader.load_tokenized_data(mode='finetune')
        initial_pids = data.pids
        if self.cfg.paths.get('exclude_pids', None) is not None:
            logger.info(f"Pids to exclude: {self.cfg.paths.exclude_pids}")
            exclude_pids = load_exclude_pids(self.cfg.paths)
            data = self.utils.process_data(data, self.patient_filter.exclude_pids, args_for_func={'exclude_pids': exclude_pids})

        predefined_pids =  'predefined_splits' in self.cfg.paths
        if predefined_pids:
            logger.warning("Using predefined splits. Ignoring test_split parameter")
            logger.warning("Use original censoring time. Overwrite n_hours parameter.")
            if not os.path.exists(self.cfg.paths.predefined_splits):
                raise ValueError(f"Predefined splits folder {self.cfg.paths.predefined_splits} does not exist.")
            if os.path.exists(join(self.cfg.paths.predefined_splits, 'finetune_config.yaml')):
                original_config = load_config(join(self.cfg.paths.predefined_splits, 'finetune_config.yaml'))
            else:
                if 'model_path' not in self.cfg.paths:
                    raise ValueError("Model path must be provided if no finetune_config in predefined splits folder.")
                original_config = load_config(join(self.cfg.paths.model_path, 'finetune_config.yaml'))
            self.cfg.outcome.n_hours = original_config.outcome.n_hours
            data = self._select_predefined_pids(data)
            self._load_outcomes_to_data(data)

        if not predefined_pids:        
            # 2. Optional: Select gender group
            if data_cfg.get('gender'):
                data = self.utils.process_data(data, self.patient_filter.select_by_gender)
            
            # 4. Loading and processing outcomes
            outcomes, censor_outcomes = self.loader.load_outcomes()
            logger.info("Assigning outcomes to data")
            data = self.utils.process_data(data, self._retrieve_and_assign_outcomes, log_positive_patients_num=True,
                                            args_for_func={'outcomes': outcomes, 'censor_outcomes': censor_outcomes})

            # 5. Optional: select patients of interest
            if data_cfg.get("select_censored"):
                data = self.utils.process_data(data, self.patient_filter.select_censored, log_positive_patients_num=True)

            # 6. Optional: Filter patients with outcome before censoring
            if self.cfg.outcome.type != self.cfg.outcome.get('censor_type', None):
                data = self.utils.process_data(data, self.patient_filter.filter_outcome_before_censor, log_positive_patients_num=True) # !Timeframe (earlier instance of outcome)

            # 7. Optional: Filter code types
            if data_cfg.get('code_types'):
                data = self.utils.process_data(data, self.code_type_filter.filter)
                data = self.utils.process_data(data, self.patient_filter.exclude_short_sequences, log_positive_patients_num=True)

        # 8. Data censoring
        data = self.utils.process_data(data, self.data_modifier.censor_data, log_positive_patients_num=True,
                                               args_for_func={'n_hours': self.cfg.outcome.n_hours})
        if not predefined_pids:
            # 3. Optional: Select Patients By Age
            if data_cfg.get('min_age') or data_cfg.get('max_age'):
                data = self.utils.process_data(data, self.patient_filter.select_by_age)
        
        # 9. Exclude patients with less than k concepts
        data = self.utils.process_data(data, self.patient_filter.exclude_short_sequences, log_positive_patients_num=True)

        # 10. Optional: Patient selection
        if data_cfg.get('num_patients') and not predefined_pids:
            data = self.utils.process_data(data, self.patient_filter.select_random_subset, log_positive_patients_num=True,
                                              args_for_func={'num_patients':data_cfg.num_patients})

        # 12. Truncation
        logger.info(f"Truncating data to {data_cfg.truncation_len} tokens")
        data = self.utils.process_data(data, self.data_modifier.truncate, args_for_func={'truncation_len': data_cfg.truncation_len})

        # 13. Normalize segments
        data = self.utils.process_data(data, self.data_modifier.normalize_segments)

        # 14. Optional: Remove any unwanted features
        if 'remove_features' in data_cfg:
            for feature in data_cfg.remove_features:
                logger.info(f"Removing {feature}")
                data.features.pop(feature, None)

        # Verify and save
        data.check_lengths()
        data = self.utils.process_data(data, self.saver.save_sequence_lengths)
        
        excluded_pids = list(set(initial_pids).difference(set(data.pids)))
        self.saver.save_list(excluded_pids, 'excluded_pids.pt')
        
        self.saver.save_data(data)
        self._log_features(data)
        return data
    
    def _prepare_mlm_features(self) -> Data:   
        """
        1. Load tokenized data
        2. Optional: Remove background tokens
        3. Exclude short sequences
        4. Optional: Select subset of patients
        5. Truncation      
        6. Normalize segments
        """
        data_cfg = self.cfg.data
        model_cfg = self.cfg.model

        # 1. Load tokenized data
        data = self.loader.load_tokenized_data(mode='pretrain')
        
        if self.cfg.paths.get('exclude_pids', None) is not None:
            logger.info(f"Pids to exclude: {self.cfg.paths.exclude_pids}")
            exclude_pids = load_exclude_pids(self.cfg.paths)
            data = self.utils.process_data(data, self.patient_filter.exclude_pids, args_for_func={'exclude_pids': exclude_pids})

        predefined_pids =  'predefined_splits' in self.cfg.paths
        if predefined_pids:
            logger.warning("Using predefined splits. Ignoring test_split parameter")
            data = self._select_predefined_pids(data)

        # 3. Exclude short sequences
        data = self.utils.process_data(data, self.patient_filter.exclude_short_sequences)
        if not predefined_pids:
            # 4. Optional: Patient Subset Selection
            if data_cfg.get('num_patients'):
                data = self.utils.process_data(data, self.patient_filter.select_random_subset, args_for_func={'num_patients':data_cfg.num_patients})

        # 5. Truncation
        logger.info(f"Truncating data to {data_cfg.truncation_len} tokens")
        data = self.utils.process_data(data, self.data_modifier.truncate, args_for_func={'truncation_len': data_cfg.truncation_len})

        # 6. Normalize segments
        data = self.utils.process_data(data, self.data_modifier.normalize_segments)
      
        # Adjust max segment if needed
        self.utils.check_and_adjust_max_segment(data, model_cfg)

        # Verify and save
        data.check_lengths()
        data = self.utils.process_data(data, self.saver.save_sequence_lengths)

        self.saver.save_data(data)
        self._log_features(data)
        return data
    
    def prepare_onehot_features(self)->Tuple[np.ndarray, np.ndarray, Dict]:
        """Use ft features and map them onto one hot vectors with binary outcomes"""
        data = self.loader.load_finetune_data()
        token2index, new_vocab = self.utils.get_token_to_index_map(data.vocabulary)
        X, y = OneHotEncoder.encode(data, token2index)
        return X, y, new_vocab

    def _retrieve_and_assign_outcomes(self, data: Data, outcomes: Dict, censor_outcomes: Dict)->Data:
        """Retrieve outcomes and assign them to the data instance"""
        data.outcomes = self.utils.select_and_order_outcomes_for_patients(outcomes, data.pids, self.cfg.outcome.type)
        if self.cfg.outcome.get('censor_type') is not None:
            data.censor_outcomes = self.utils.select_and_order_outcomes_for_patients(censor_outcomes, data.pids, self.cfg.outcome.censor_type)
        else:
            data.censor_outcomes = [None]*len(outcomes)
        return data
    @staticmethod
    def _get_predefined_pids(predefined_splits_path)->List:
        """Return pids from predefined splits"""
        if os.path.exists(join(predefined_splits_path, 'pids.pt')):
            return torch.load(join(predefined_splits_path, 'pids.pt'))
        else:
            train_pids = torch.load(get_pids_file(predefined_splits_path, 'train'))
            val_pids = torch.load(get_pids_file(predefined_splits_path, 'val'))
            return train_pids + val_pids

    def _select_predefined_pids(self, data: Data):
        """ Validate predefined splits as subset of data."""
        predefined_splits_path = self.cfg.paths.predefined_splits
        predefined_pids = self._get_predefined_pids(predefined_splits_path)
        if not set(predefined_pids).issubset(set(data.pids)):
            raise ValueError(f"Pids in the predefined splits must be a subset of data.pids. There are {len(set(predefined_pids).difference(set(data.pids)))} pids in the data that are not in the predefined splits")
        data = data.select_data_subset_by_pids(predefined_pids, mode=data.mode)
        return data
    
    def _load_outcomes_to_data(self, data: Data)->None:
        """ Load outcomes and censor outcomes to data. """
        for outcome_type in ['outcomes', 'censor_outcomes']:
            setattr(data, outcome_type, torch.load(join(self.cfg.paths.predefined_splits, f'{outcome_type}.pt')))

    def _log_features(self, data:Data)->None:
        logger.info(f"Final features: {data.features.keys()}")
        logger.info("Example features: ")
        for k, v in data.features.items():
            logger.info(f"{k}: {v[0]}")

class OneHotEncoder:
    @staticmethod
    def encode(data:Data, token2index: dict) -> Tuple[np.ndarray, np.ndarray]:
        # ! Potentially map gender onto one index?
        """Encode features to one hot and age at the time of last event"""
        AGE_INDEX = 0
         # Initialize arrays
        num_samples = len(data)
        num_features = len(token2index) + 1 # +1 for age

        X, y = OneHotEncoder.initialize_Xy(num_samples, num_features)
        keys_array = np.array(list(token2index.keys())) # Create an array of keys for faster lookup
        token2index_map = np.vectorize(token2index.get) # Vectorized function to map tokens to indices

        for sample, (concepts, outcome) in enumerate(zip(data.features['concept'], data.outcomes)):
            y[sample] = OneHotEncoder.encode_outcome(outcome)
            X[sample, AGE_INDEX] = data.features['age'][sample][-1]   
            OneHotEncoder.encode_concepts(concepts, token2index_map, keys_array, X, sample)
        return X, y

    @staticmethod
    def encode_outcome(outcome) -> int:
        return int(pd.notna(outcome))

    @staticmethod
    def encode_concepts(concepts: List[int], token2index_map: np.vectorize, 
                        keys_array: np.ndarray, X:np.ndarray, sample: int) -> None:
        concepts = np.array(concepts)
        unique_concepts = np.unique(concepts)
        valid_concepts_mask = np.isin(unique_concepts, keys_array) # Only keep concepts that are in the token2index map
        filtered_concepts = unique_concepts[valid_concepts_mask]
        concept_indices = token2index_map(filtered_concepts) + 1
        X[sample, concept_indices] = 1

    @staticmethod
    def initialize_Xy(num_samples: int, num_features: int)->Tuple[np.ndarray, np.ndarray]:
        X = np.zeros((num_samples, num_features), dtype=np.int16)
        y = np.zeros(num_samples, dtype=np.int16)
        return X, y
    
class DataModifier:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    @staticmethod
    def truncate(data: Data, truncation_len: int) -> Data:
        truncator = Truncator(max_len=truncation_len, 
                              vocabulary=data.vocabulary)
        data.features = truncator(data.features)
        return data

    def censor_data(self, data: Data, n_hours: float) -> Data:
        """Censors data n_hours after censor_outcome."""
        censorer_cfg = self.cfg.data.get('censorer', {'_target_': 'data_fixes.censor.Censorer'})
        censorer = instantiate(censorer_cfg, vocabulary=data.vocabulary, n_hours=n_hours)
        logger.info(f"Censoring data {n_hours} hours after outcome with {censorer.__class__.__name__}")
        data.features, data.censor_outcomes = censorer(data.features, data.censor_outcomes, exclude=False)
        return data

    @staticmethod
    def normalize_segments(data: Data) -> Data:
        """Normalize segments after truncation to start with 1 and increase by 1
        or if position_ids present (org. BEHRT version) then normalize those."""
        segments_key = 'segment' if 'segment' in data.features else 'position_ids'

        for idx, segments in enumerate(data.features[segments_key]):
            data.features[segments_key][idx] = Handler.normalize_segments(segments)

        return data

def retrieve_outcomes(all_outcomes: Dict, all_censor_outcomes: Dict, cfg: Config)->Union[List, List]:
    """From the configuration, load the outcomes and censor outcomes."""
    pids = all_outcomes[PID_KEY]
    outcomes = all_outcomes.get(cfg.outcome.type, [None]*len(all_outcomes[PID_KEY]))
    censor_outcomes = all_censor_outcomes.get(cfg.outcome.get('censor_type'), [None]*len(outcomes))
    return outcomes, censor_outcomes, pids

def select_positives(outcomes: List, censor_outcomes: List, pids: List)->Tuple[List, List, List]:
    """Select only positive outcomes."""
    logger.info("Selecting only positive outcomes")
    select_indices = set([i for i, outcome in enumerate(outcomes) if pd.notna(outcome)])
    outcomes = [outcomes[i] for i in select_indices]
    censor_outcomes = [censor_outcomes[i] for i in select_indices]
    pids = [pids[i] for i in select_indices]
    return outcomes, censor_outcomes, pids

