import logging
import os
import re
from os.path import join
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from common.utils import Data, iter_patients
from data.dataset import (BinaryOutcomeDataset, HierarchicalMLMDataset,
                          MLMDataset)
from data_fixes.adapt import BehrtAdapter
from data_fixes.censor import Censorer
from data_fixes.exclude import Excluder
from data_fixes.handle import Handler
from data_fixes.truncate import Truncator
from transformers import BertConfig

logger = logging.getLogger(__name__)  # Get the logger for this module

PID_KEY = 'PID'
VOCABULARY_FILE = 'vocabulary.pt'
HIERARCHICAL_VOCABULARY_FILE = 'h_vocabulary.pt'
TREE_FILE = 'tree.pt'
TREE_MATRIX_FILE = 'tree_matrix.pt'
BG_GENDER_KEYS = {
    'male': ['M', 'Kvinde', 'male', 'Male', 'man', 'MAN', '1'],
    'female': ['W', 'Mand', 'F', 'female', 'woman', 'WOMAN', '0']
}
MIN_POSITIVES = {'train': 10, 'val': 5}
SPECIAL_CODES = ['[', 'BG_', 'BG']
CHECKPOINT_FOLDER = 'checkpoints'
# TODO: Add option to load test set only!
class DatasetPreparer:
    def __init__(self, cfg):
        self.cfg = cfg
       
        self.utils = Utilities()
        self.loader = Loader(cfg)
        
        run_folder = join(self.cfg.paths.output_path, self.cfg.paths.run_name)
        self.saver = Saver(run_folder)
        
        self.patient_filter = PatientFilter(cfg)
        self.code_type_filter = CodeTypeFilter(cfg)
        self.data_modifier = DataModifier()

    def prepare_mlm_dataset(self):
        """Load data, truncate, adapt features, create dataset"""
        train_features, val_features, vocabulary = self._prepare_mlm_features()

        train_dataset = MLMDataset(train_features, vocabulary, **self.cfg.data.dataset)
        val_dataset = MLMDataset(val_features, vocabulary, **self.cfg.data.dataset)
        
        return train_dataset, val_dataset

    def prepare_hmlm_dataset(self):
        train_features, val_features, vocabulary = self._prepare_mlm_features()
        tree, tree_matrix, h_vocabulary = self._load_tree()
        
        self.saver.save_vocab(h_vocabulary, HIERARCHICAL_VOCABULARY_FILE)
        train_dataset = HierarchicalMLMDataset(train_features, vocabulary, 
                                            h_vocabulary, tree, tree_matrix, 
                                            **self.cfg.data.dataset)
        val_dataset = HierarchicalMLMDataset(val_features, vocabulary, 
                                            h_vocabulary, tree, tree_matrix, 
                                            **self.cfg.data.dataset)
        return train_dataset, val_dataset

    def prepare_finetune_features(self)->Tuple[dict]:
        """
        Prepare the features for fine-tuning. 
        The process includes:
        1. Loading tokenized data
        2. Excluding pretrain patients 
        3. Optional: Select male/female
        4. Optional: Select age group
        5. Loading and processing outcomes
        6. Optional: Select only patients with a censoring outcome
        7. Filter patients with outcome before censoring
        8. Optional: Filter code types (e.g. only diagnoses)
        9. Data censoring
        10. Exclude patients with less than k concepts
        11. Optional select random subset
        12. Optional: Remove background tokens
        13. Truncation
        14. Normalize segments
        """
        data_cfg = self.cfg.data

        # 1. Loading tokenized data
        data = self.loader.load_tokenized_finetune_data('val')
        datasets = {'val': data}
        
        # 2. Excluding pretrain patients
        datasets = self.utils.process_datasets(datasets, self.patient_filter.exclude_pretrain_patients) 
        
        # 3. Optinally select gender group
        if data_cfg.get('gender', False):
            datasets = self.utils.process_datasets(datasets, self.patient_filter.select_by_gender)
        
        # 4. Select Patients By Age
        if data_cfg.get('min_age', False) or data_cfg.get('max_age', False):
            datasets = self.utils.process_datasets(datasets, self.patient_filter.select_by_age)

        # 5. Loading and processing outcomes
        outcomes, censor_outcomes = self.loader.load_outcomes()
        logger.info("Assigning outcomes to data")
        datasets = self.utils.process_datasets(datasets, self._retrieve_and_assign_outcomes, 
                                           {'val': {'outcomes': outcomes, 'censor_outcomes': censor_outcomes}})
        self.utils.log_pos_patients_num(datasets)
        # 6. Optionally select patients of interest
        if data_cfg.get("select_censored", False):
            datasets = self.utils.process_datasets(datasets, self.patient_filter.select_censored)
            self.utils.log_pos_patients_num(datasets)
        
        # 7. Filter patients with outcome before censoring
        if self.cfg.outcome.type != self.cfg.outcome.get('censor_type', None):
            datasets = self.utils.process_datasets(datasets, self.patient_filter.filter_outcome_before_censor) # !Timeframe (earlier instance of outcome)
            self.utils.log_pos_patients_num(datasets)
        
        # 8. Filter code types
        if data_cfg.get('code_types', False):
            datasets = self.utils.process_datasets(datasets, self.code_type_filter.filter)
            datasets = self.utils.process_datasets(datasets, self.patient_filter.exclude_short_sequences)
            self.utils.log_pos_patients_num(datasets)

        # 9. Data censoring
        datasets = self.utils.process_datasets(datasets, self.data_modifier.censor_data,
                                               {'val': {'n_hours': self.cfg.outcome.n_hours}})
        self.utils.log_pos_patients_num(datasets)

        # 10. Exclude patients with less than k concepts
        datasets = self.utils.process_datasets(datasets, self.patient_filter.exclude_short_sequences)
        self.utils.log_pos_patients_num(datasets)
        # 11. Patient selection
        if data_cfg.get('num_patients', False):
            datasets = self.utils.process_datasets(datasets, self.patient_filter.select_random_subset, 
                                              {'val': {'num_patients':data_cfg.num_patients}})
            self.utils.log_pos_patients_num(datasets)
        
        # 12. Optionally Remove Background Tokens
        if data_cfg.get("remove_background", False):
            datasets = self.utils.process_datasets(datasets, self.data_modifier.remove_background)

        # 13. Truncation
        logger.info('Truncating')
        datasets = self.utils.process_datasets(datasets, self.data_modifier.truncate,
                                               {'val': {'truncation_len': data_cfg.truncation_len}})

        # 14. Normalize Segments
        logger.info('Normalizing segments')
        datasets = self.utils.process_datasets(datasets, self.data_modifier.normalize_segments)

        datasets['val'].check_lengths()

        datasets = self.utils.process_datasets(datasets, self.saver.save_sequence_lengths)
        data = datasets['val']

        if self.cfg.model.get('behrt_embeddings', False):
            logger.info('Adapting features for behrt embeddings')
            data.features = BehrtAdapter().adapt_features(data.features)
        self.saver.save_data(data)
  
        return data
    
    def _prepare_mlm_features(self)->Tuple[dict, dict, dict]:   
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
        train_features, train_pids, val_features, val_pids, vocabulary = self.loader.load_tokenized_data()
        self.saver.save_vocab(vocabulary)
        datasets = {'train': Data(train_features, train_pids, vocabulary=vocabulary, mode='train'),
                    'val': Data(val_features, val_pids, vocabulary=vocabulary, mode='val')}
        
        # Optional: Remove background tokens
        if data_cfg.get("remove_background", False):
            datasets = self.utils.process_datasets(datasets, self.data_modifier.remove_background)

        # Exclude short sequences
        datasets = self.utils.process_datasets(datasets, self.patient_filter.exclude_short_sequences)

        # Patient Subset Selection
        if data_cfg.get('num_train_patients', False) or data_cfg.get('num_val_patients', False):
            datasets = self.utils.process_datasets(datasets, self.patient_filter.select_random_subset, 
                                              {'train': {'num_patients':data_cfg.num_train_patients},
                                               'val': {'num_patients':data_cfg.num_val_patients}})

        truncation_len = data_cfg.truncation_len
        logger.info(f"Truncating data to {truncation_len} tokens")
        datasets = self.utils.process_datasets(datasets, self.data_modifier.truncate,
                                               {'train': {'truncation_len': truncation_len},
                                                'val':{'truncation_len':truncation_len}},)

        datasets = self.utils.process_datasets(datasets, self.data_modifier.normalize_segments)

        datasets['train'].check_lengths()
        datasets['val'].check_lengths()
        self.saver.save_train_val_pids(datasets['train'].pids, datasets['val'].pids)
        datasets = self.utils.process_datasets(datasets, self.saver.save_sequence_lengths)
      
        self.utils.check_and_adjust_max_segment(datasets['train'], model_cfg)
        self.utils.check_and_adjust_max_segment(datasets['val'], model_cfg)
        
        if model_cfg.get('behrt_embeddings', False):
            logger.info("Adapting features for BEHRT embeddings")
            datasets['train'].features = BehrtAdapter().adapt_features(datasets['train'].features)
            datasets['val'].features = BehrtAdapter().adapt_features(datasets['val'].features)
        
        return datasets['train'].features, datasets['val'].features, datasets['train'].vocabulary
    
    def prepare_onehot_features(self)->Tuple[np.ndarray, np.ndarray, Dict]:
        """Use ft features and map them onto one hot vectors with binary outcomes"""
        data = self.loader.load_finetune_data(mode='val')
        token2index, new_vocab = self.utils.get_token_to_index_map(data.vocabulary)
        X, y = OneHotEncoder.encode(data, token2index)
        return X, y, new_vocab

    def _retrieve_and_assign_outcomes(self, data: Data, outcomes: Dict, censor_outcomes: Dict)->Data:
        """Retrieve outcomes and assign them to the data instance"""
        data.outcomes = self.utils.select_and_order_outcomes_for_patients(outcomes, data.pids, self.cfg.outcome.type)
        if self.cfg.outcome.get('censor_type', None) is not None:
            data.censor_outcomes = self.utils.select_and_order_outcomes_for_patients(censor_outcomes, data.pids, self.cfg.outcome.censor_type)
        else:
            data.censor_outcomes = [None]*len(outcomes)
        return data
        

class OneHotEncoder():
    @staticmethod
    def encode(data:Data, token2index: dict)->Tuple[np.ndarray, np.ndarray]:
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
    def encode_outcome(outcome: str)->int:
        return int(not pd.isna(outcome))
    @staticmethod
    def encode_concepts(concepts: List[int], token2index_map: np.vectorize, 
                        keys_array: np.ndarray, X:np.ndarray, sample: int)->None:
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


class DataModifier():
    @staticmethod
    def truncate(data: Data, truncation_len: int)->Data:
        truncator = Truncator(max_len=truncation_len, 
                              vocabulary=data.vocabulary)
        data.features = truncator(data.features)
        return data    
    @staticmethod
    def remove_background(data: Data)->Data:
        """Remove background tokens from features and the first sep token following it"""
        background_indices = Utilities.get_background_indices(data)
        first_index = min(background_indices)
        last_index = max(background_indices)
        for k, token_lists in data.features.items():
            new_tokens_lists = []
            for idx, tokens in enumerate(token_lists):
                new_tokens = [token for j, token in enumerate(tokens) if (j < first_index) or (j > last_index)]
                new_tokens_lists.append(new_tokens)
            data.features[k] = new_tokens_lists 
        return data
    @staticmethod
    def censor_data(data: Data, n_hours: float)->Data:
        """Censors data n_hours after censor_outcome."""
        censorer = Censorer(n_hours, vocabulary=data.vocabulary)
        data.features = censorer(data.features, data.censor_outcomes, exclude=False)
        return data
    @staticmethod
    def normalize_segments(data: Data)->Data:
        """Normalize segments after truncation to start with 1 and increase by 1
        or if position_ids present (org. BEHRT version) then normalize those."""
        segments_key = 'position_ids' if 'position_ids' in data.features.keys() else 'segment'
        
        segments = []
        for segment in data.features[segments_key]:
            segments.append(Handler.normalize_segments(segment))
        
        data.features[segments_key] = segments
        return data


class CodeTypeFilter():
    def __init__(self, cfg):
        self.cfg = cfg
        self.utils = Utilities()
        self.SPECIAL_CODES = SPECIAL_CODES

    def filter(self, data: Data)->Data:
        """Filter code types, e.g. keep only diagnoses. Remove patients with not sufficient data left."""
        keep_codes = self._get_keep_codes()
        keep_tokens = set([token for code, token in data.vocabulary.items() if self.utils.code_starts_with(code, keep_codes)])
        logger.info(f"Keep only codes starting with: {keep_codes}")
        for patient_index, patient in enumerate(iter_patients(data.features)):
            self._filter_patient(data, patient, keep_tokens, patient_index)
        return data

    def _filter_patient(self, data: Data, patient: dict, keep_tokens: set, patient_index: int)->None:
        """Filter patient in place by removing tokens that are not in keep_tokens"""
        concepts = patient['concept']
        keep_entries = {i:token for i, token in enumerate(concepts) if \
                            token in keep_tokens}
        for k, v in patient.items():
            filtered_list = [v[i] for i in keep_entries]
            data.features[k][patient_index] = filtered_list

    def _get_keep_codes(self)->set:
        """Return a set of codes to keep according to cfg.data.code_types."""
        return set(self.SPECIAL_CODES + self.cfg.data.code_types)
    

class PatientFilter():
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.utils = Utilities()

    def exclude_pretrain_patients(self, data: Data)->Data:
        pretrain_pids = set(torch.load(join(self.cfg.paths.model_path, f'pids_{data.mode}.pt')))
        kept_indices = [i for i, pid in enumerate(data.pids) if pid not in pretrain_pids]
        return self.select_entries(data, kept_indices)

    def filter_outcome_before_censor(self, data: Data)->Data:
        """Filter patients with outcome before censoring and missing censoring when outcome present."""
        kept_indices = []
        for i, (outcome, censor) in enumerate(zip(data.outcomes, data.censor_outcomes)):
            if pd.isna(censor):
                if pd.isna(outcome):
                    kept_indices.append(i)
            elif pd.isna(outcome) or outcome >= (censor + self.cfg.outcome.n_hours):
                kept_indices.append(i)
        return self.select_entries(data, kept_indices)
    
    def select_censored(self, data: Data)->Data:
        """Select only censored patients. This is only relevant for the fine-tuning data. 
        E.g. for pregnancy complications select only pregnant women."""
        kept_indices = [i for i, censor in enumerate(data.censor_outcomes) if not pd.isna(censor)]
        return self.select_entries(data, kept_indices)

    def exclude_short_sequences(self, data: Data)->Data:
        """Exclude patients with less than k concepts"""
        excluder = Excluder(min_len = self.cfg.data.get('min_len', 3),
                            vocabulary=data.vocabulary)
        kept_indices = excluder._exclude(data.features)
        return self.select_entries(data, kept_indices)
    
    def select_by_age(self, data: Data)->Data:
        """
        Assuming that age is given in days. 
        We retrieve the last age of each patient and check whether it's within the range.
        """
        kept_indices = []
        min_age = self.cfg.data.get('min_age', 0)
        max_age = self.cfg.data.get('max_age', 120)
        kept_indices = [i for i, ages in enumerate(data.features['age']) 
                if min_age <= ages[-1] <= max_age]
        return self.select_entries(data, kept_indices)
    
    def select_by_gender(self, data: Data)->Data:
        """Select only patients of a certain gender"""
        gender_token = self.utils.get_gender_token(data.vocabulary, self.cfg.data.gender)
        kept_indices = [i for i, concepts in enumerate(data.features['concept']) if gender_token in set(concepts)]
        return self.select_entries(data, kept_indices)
    
    def select_random_subset(self, data, num_patients, seed=0)->Data:
        """Select a num_patients random patients"""
        if len(data.pids) <= num_patients:
            return data
        np.random.seed(seed)
        indices = np.arange(len(data.pids))
        np.random.shuffle(indices)
        indices = indices[:num_patients]
        return self.select_entries(data, indices)

    @staticmethod
    def select_entries(data:Data, indices:List)->Data:
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


class Loader():
    def __init__(self, cfg):
        self.path_cfg = cfg.paths

    def load_tokenized_data(self)->Tuple[dict, list, dict, list, dict]:
        tokenized_dir = self.path_cfg.get('tokenized_dir', 'tokenized')
        logger.info('Loading tokenized data from %s', tokenized_dir)
        tokenized_data_path = join(self.path_cfg.data_path, tokenized_dir)

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
            vocabulary = torch.load(join(self.path_cfg.data_path, VOCABULARY_FILE))
        return train_features, train_pids, val_features, val_pids, vocabulary

    def load_tokenized_finetune_data(self, mode)->Data:
        tokenized_dir = self.path_cfg.get('tokenized_dir', 'tokenized')
        tokenized_file = self.path_cfg.get('tokenized_file', 'tokenized_val.pt')
        tokenized_pids = self.path_cfg.get('tokenized_pids', 'pids_val.pt')
        
        tokenized_data_path = join(self.path_cfg.data_path, tokenized_dir)
        
        logger.info(f"Loading tokenized data from {tokenized_data_path}")
        features  = torch.load(join(tokenized_data_path, tokenized_file))
        pids = torch.load(join(tokenized_data_path,  tokenized_pids))
        
        logger.info("Loading vocabulary")
        try:
            vocabulary = torch.load(join(tokenized_data_path, VOCABULARY_FILE))
        except:
            vocabulary = torch.load(join(self.path_cfg.data_path, VOCABULARY_FILE))
        return Data(features, pids, vocabulary=vocabulary, mode=mode)
    
    def load_outcomes(self)->Tuple[dict, dict]:
        logger.info(f'Load outcomes from {self.path_cfg.outcome}')
        censoring_timestamps_path = self.path_cfg.censor if self.path_cfg.get("censor", False) else self.path_cfg.outcome
        logger.info(f'Load censoring_timestamps from {censoring_timestamps_path}')
        outcomes = torch.load(self.path_cfg.outcome)
        censor_outcomes = torch.load(self.path_cfg.censor) if self.path_cfg.get('censor', False) else outcomes   
        return outcomes, censor_outcomes

    def load_tree(self)->Tuple[dict, torch.Tensor, dict]:
        hierarchical_path = join(self.path_cfg.data_path, 
                                 self.path_cfg.hierarchical_dir)
        tree = torch.load(join(hierarchical_path, TREE_FILE))
        tree_matrix = torch.load(join(hierarchical_path, TREE_MATRIX_FILE))
        h_vocabulary = torch.load(join(hierarchical_path, VOCABULARY_FILE))
        return tree, tree_matrix, h_vocabulary 
    
    def load_model(self, model_class, add_config:dict={}, checkpoint: dict=None):
        """Load model from config and checkpoint. model_class is the class of the model to be loaded."""
        checkpoint = self.load_checkpoint() if checkpoint is None else checkpoint
        # Load the config from file
        config = BertConfig.from_pretrained(self.path_cfg.model_path) 
        config.update(add_config)
        model = model_class(config)
        
        return self.load_state_dict_into_model(model, checkpoint)
    
    def load_state_dict_into_model(self, model: torch.nn.Module, checkpoint: dict)->torch.nn.Module:
        load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        missing_keys = load_result.missing_keys

        if len([k for k in missing_keys if k.startswith('embeddings')])>0:
            pretrained_model_embeddings = model.embeddings.__class__.__name__
            raise ValueError(f"Embeddings not loaded. Ensure that model.behrt_embeddings is compatible with pretrained model embeddings {pretrained_model_embeddings}.")
        logger.warning("missing state dict keys: %s", missing_keys)
        return model

    def load_checkpoint(self)->dict:
        """Load checkpoint, if checkpoint epoch provided. Else load last checkpoint."""
        checkpoints_path = join(self.path_cfg.model_path, CHECKPOINT_FOLDER)
        checkpoint_epoch = self.get_checkpoint_epoch()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.load(join(checkpoints_path,f'checkpoint_epoch{checkpoint_epoch}_end.pt'), map_location=device)
    
    def get_checkpoint_epoch(self)->int:
        """Get checkpoint epoch from config or return the last checkpoint_epoch for this model."""
        checkpoint_epoch = self.path_cfg.get('checkpoint_epoch', None)
        if checkpoint_epoch is None:
            logger.info("No checkpoint provided. Loading last checkpoint.")
            checkpoint_epoch = Utilities.get_last_checkpoint_epoch(join(self.path_cfg.model_path, CHECKPOINT_FOLDER))
        return checkpoint_epoch
    
    def load_finetune_data(self, path: str=None, mode: str='val')->Data:
        """Load features for finetuning"""
        path = self.path_cfg.finetune_features_path if path is None else path
        features = torch.load(join(path, f'features.pt'))
        outcomes = torch.load(join(path, f'outcomes.pt'))
        pids = torch.load(join(path, f'pids.pt'))
        vocabulary = torch.load(join(path, 'vocabulary.pt'))
        return Data(features, pids, outcomes, vocabulary=vocabulary, mode=mode)


class Saver():
    """Save features, pids, vocabulary and sequence lengths to a folder"""
    def __init__(self, run_folder) -> None:
        self.run_folder = run_folder
        os.makedirs(self.run_folder, exist_ok=True)
    
    def save_sequence_lengths(self, data: Data)->Data:
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
    
    def save_data(self, data: Data)->None:
        """Save data (features, pids and outcomes (if present) to run_folder)"""
        torch.save(data.features, join(self.run_folder, 'features.pt'))
        torch.save(data.pids, join(self.run_folder, 'pids.pt'))
        torch.save(data.vocabulary, join(self.run_folder, 'vocabulary.pt'))
        if data.outcomes is not None:
            torch.save(data.outcomes, join(self.run_folder, 'outcomes.pt'))

    def save_vocab(self, vocabulary, name: str=VOCABULARY_FILE):
        torch.save(vocabulary, join(self.run_folder, name))
        

class Utilities():
    def process_datasets(self, datasets: Dict, func: callable, args_for_func: Dict=None)->Dict:
        """Apply a function to all datasets in a dictionary"""
        if args_for_func is None:
            args_for_func = {}
        for split, data in datasets.items():
            # Get mode-specific arguments, or an empty dictionary if they don't exist
            mode_args = args_for_func.get(split, {})
            datasets[split] = func(data, **mode_args)
        self.log_patient_nums(func.__name__, datasets)
        return datasets
    @staticmethod
    def log_patient_nums(operation:str, datasets: Dict)->None:
        logger.info(f"After applying {operation}:")
        for split, data in datasets.items():
            logger.info(f"{split}: {len(data.pids)} patients")
    @staticmethod
    def select_and_order_outcomes_for_patients(all_outcomes: Dict, pids: List, outcome: str) -> List:
        """Select outcomes for patients and order them based on the order of pids"""
        outcome_pids = all_outcomes[PID_KEY]
        outcome_group = all_outcomes[outcome]
        assert len(outcome_pids) == len(outcome_group), "Mismatch between PID_KEY length and outcome_group length"

        # Create a dictionary of positions for each PID for quick lookup
        pid_to_index = {pid: idx for idx, pid in enumerate(outcome_pids)}
        
        outcome_pids = set(outcome_pids)
        if not set(pids).issubset(outcome_pids):
            logger.warn(f"PIDs is not a subset of outcome PIDs, there is a mismatch of {len(set(pids).difference(outcome_pids))} patients") 
        
        outcomes = [outcome_group[pid_to_index[pid]] if pid in outcome_pids else None for pid in pids]
        return outcomes
    
    @staticmethod
    def check_and_adjust_max_segment(data: Data, model_cfg)->None:
        """Check max segment. If it's larger or equal to the model type vocab size, change accordingly."""
        max_segment = max([max(seg) for seg in data.features['segment']])
        type_vocab_size = model_cfg.type_vocab_size
        if max_segment>=type_vocab_size:
            logger.warning(f"You've set type_vocab_size too low. Max segment {max_segment} >= type_vocab_size {type_vocab_size}\
                             We'll change it to {max_segment+1}.")
            model_cfg.type_vocab_size = max_segment+1
    @staticmethod
    def get_token_to_index_map(vocabulary:dict)->Tuple[dict]:
        """
        Creates a new mapping from vocbulary values to new integers excluding special tokens
        """
        filtered_tokens = set([v for k, v in vocabulary.items() if not k.startswith('[')])
        token2index = {token: i for i, token in enumerate(filtered_tokens)}        
        new_vocab = {k: token2index[v] for k, v in vocabulary.items() if v in token2index}
        return token2index, new_vocab
    @staticmethod
    def get_gender_token(vocabulary: dict, key: str)->int:
        """Get the token from the vocabulary corresponding to the gender provided in the config"""
        # Determine the gender category
        gender_category = None
        for category, keys in BG_GENDER_KEYS.items():
            if key in keys:
                gender_category = category
                break

        if gender_category is None:
            raise ValueError(f"Unknown gender {key}, please select one of {BG_GENDER_KEYS}")

        # Check the vocabulary for a matching token
        for possible_key in BG_GENDER_KEYS[gender_category]:
            gender_token = vocabulary.get('BG_GENDER_' + possible_key, None)
            if gender_token is not None:
                return gender_token
    
        raise ValueError(f"None of BG_GENDER_+{BG_GENDER_KEYS[gender_category]} found in vocabulary.")
    @staticmethod
    def get_background_indices(data: Data)->List[int]:
        """Get the length of the background sentence"""
        background_tokens = set([v for k, v in data.vocabulary.items() if k.startswith('BG_')])
        example_concepts = data.features['concept'][0] # Assume that all patients have the same background length
        background_indices = [i for i, concept in enumerate(example_concepts) if concept in background_tokens]
        if data.vocabulary['[SEP]'] in example_concepts:
            background_indices.append(max(background_indices)+1)
        return background_indices
    @staticmethod
    def code_starts_with(code: int, prefixes: set)->bool:
        """Check if the code starts with any of the given prefixes."""
        return any(code.startswith(prefix) for prefix in prefixes)
    @staticmethod
    def log_pos_patients_num(datasets: Dict)->None:
        for mode, data in datasets.items():
            num_positive_patiens = len([t for t in data.outcomes if not pd.isna(t)])
            if num_positive_patiens < MIN_POSITIVES[mode]:
                raise ValueError(f"Number of positive patients is less than {MIN_POSITIVES[mode]}: {num_positive_patiens}")
            logger.info(f"Positive {mode} patients: {num_positive_patiens}")
    @staticmethod
    def get_last_checkpoint_epoch(checkpoint_folder: str)->int:
        """Returns the epoch of the last checkpoint."""
        # Regular expression to match the pattern retry_XXX
        pattern = re.compile(r"checkpoint_epoch(\d+)_end\.pt$")
        max_epoch = None
        for filename in os.listdir(checkpoint_folder):
            match = pattern.match(filename)
            if match:
                epoch = int(match.group(1))
                if max_epoch is None or epoch > max_epoch:
                    max_epoch = epoch
        # Return the folder with the maximum retry number
        if max_epoch is None:
            raise ValueError("No checkpoint found in folder {}".format(checkpoint_folder))
        return max_epoch


def create_binary_outcome_datasets(all_outcomes, cfg):
    """
    This function is used to create outcome datasets based on the configuration provided.
    """
    outcomes, censor_outcomes, pids = retrieve_outcomes(all_outcomes, cfg)
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

def retrieve_outcomes(self, all_outcomes: Dict, all_censor_outcomes: Dict, cfg)->Union[List, List]:
    """From the configuration, load the outcomes and censor outcomes."""
    pids = all_outcomes[PID_KEY]
    outcomes = all_outcomes.get(cfg.outcome.type, [None]*len(all_outcomes[PID_KEY]))
    censor_outcomes = all_censor_outcomes.get(cfg.outcome.get('censor_type', None), [None]*len(outcomes))
    return outcomes, censor_outcomes, pids

def select_positives(outcomes, censor_outcomes, pids):
    """Select only positive outcomes."""
    logger.info("Selecting only positive outcomes")
    select_indices = set([i for i, outcome in enumerate(outcomes) if pd.notna(outcome)])
    outcomes = [outcomes[i] for i in select_indices]
    censor_outcomes = [censor_outcomes[i] for i in select_indices]
    pids = [pids[i] for i in select_indices]
    return outcomes, censor_outcomes, pids