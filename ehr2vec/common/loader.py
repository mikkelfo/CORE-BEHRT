import logging
import os
from os.path import join

import numpy as np
import pandas as pd
import torch
from data.dataset import CensorDataset, HierarchicalMLMDataset, MLMDataset
from data_fixes.adapt import BehrtAdapter
from data_fixes.censor import Censorer
from data_fixes.truncate import Truncator
from transformers import BertConfig

logger = logging.getLogger(__name__)  # Get the logger for this module

PID_KEY = 'PID'
VOCABULARY_FILE = 'vocabulary.pt'

def load_model(model_class, cfg, add_config={}):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = join(cfg.paths.model_path, 'checkpoints' ,f'checkpoint_epoch{cfg.paths.checkpoint_epoch}_end.pt')
    # Load the config from file
    config = BertConfig.from_pretrained(cfg.paths.model_path) 
    config.update(add_config)
    model = model_class(config)
    load_result = model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'], strict=False)
    logger.info("missing state dict keys", load_result.missing_keys)
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
        train_dataset = CensorDataset(cfg.paths.data_path, 'train', outcomes, 
                                    censor_outcomes=censor_outcomes, 
                                    outcome_pids=pids,
                                    num_patients=cfg.train_data.num_patients,
                                    pids=pids if cfg.get("encode_pos_only", False) else None,
                                    n_hours=cfg.outcome.n_hours,
                                    n_procs=cfg.train_data.get('n_procs', None))

    if cfg.val_data.num_patients == 0:
        val_dataset = None
    else:
        val_dataset = CensorDataset(cfg.paths.data_path, 'val',  outcomes, 
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
    select_indices = [i for i, outcome in enumerate(outcomes) if pd.notna(outcome)]
    outcomes = [outcomes[i] for i in select_indices]
    censor_outcomes = [censor_outcomes[i] for i in select_indices]
    pids = [pids[i] for i in select_indices]
    return outcomes, censor_outcomes, pids

# TODO: Add option to load test set only!
    
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
        """
        Prepare the dataset for fine-tuning. 
        The process includes:
        1. Loading tokenized data
        2. Excluding pretrain patients
        3. Loading and processing outcomes
        4. Data censoring
        5. Patient selection
        6. Truncating data
        7. Creating the dataset
        """
        
        # 1. Loading tokenized data
        train_features, train_pids, val_features, val_pids, vocabulary = self._load_tokenized_data()
        
        # 2. Excluding pretrain patients
        logger.info('Exclude pretrain patients')
        train_features, train_pids = self._exclude_pretrain_patients(train_features, train_pids, 'train')
        val_features, val_pids = self._exclude_pretrain_patients(val_features, val_pids, 'val')
        
        # 3. Loading and processing outcomes
        logger.info('Load outcomes')
        outcomes = torch.load(join(self.cfg.paths.data_path, self.cfg.paths.outcome))
        train_outcome, train_outcome_censor = self._process_outcomes(outcomes, train_pids)
        val_outcome, val_outcome_censor = self._process_outcomes(outcomes, val_pids)
        
        # 4. Data censoring
        logger.info('Censoring')
        train_features, train_pids = self._censor_data(train_features, train_pids,
                                                       train_outcome_censor, vocabulary)
        val_features, val_pids = self._censor_data(val_features, val_pids,
                                                    val_outcome_censor, vocabulary)
        
        # 5. Patient selection
        logger.info('Selecting patients')
        train_features, train_pids = DatasetPreparer._select_random_subset(
            train_features, train_pids, self.cfg.data.num_train_patients)
        val_features, val_pids = DatasetPreparer._select_random_subset(
            val_features, val_pids, self.cfg.data.num_val_patients)
        self._save_pids(train_pids, val_pids) # this has to be moved to after patient exclusion, once short sequences are removed

        # 6. Optionally Remove Background Tokens
        if self.cfg.data.get("remove_background", False):
            logger.info("Removing background tokens")
            train_features = self._remove_background(train_features, vocabulary)
            val_features = self._remove_background(val_features, vocabulary)

        # 7. Truncation
        logger.info('Truncating')
        truncator = Truncator(max_len=self.cfg.data.truncation_len, sep_token=vocabulary['[SEP]'])
        train_features = truncator(train_features)
        val_features = truncator(val_features)

        # 8. Censoring
        train_dataset = CensorDataset(train_features, outcomes=train_outcome)
        val_dataset = CensorDataset(val_features, outcomes=val_outcome)
        return train_dataset, val_dataset

    # def prepare_onehot_dataset(self):
    #     train_features, train_pids, val_features, val_pids, vocabulary = self._load_tokenized_data()
    #     # extend by test!
    #     censorer = Censorer(n_hours=cfg.outcomes.n_hours, min_len=cfg.excluder.min_len, vocabulary=vocabulary)
    #     outcomes = torch.load(join(cfg.paths.data_path, cfg.paths.outcome))
    #     tokenized_features_train = censorer(tokenized_features_train, outcome, vocabulary, cfg)
    #     pass

    def _censor_data(self, features, pids, outcome_censor, vocabulary):
        censorer = Censorer(self.cfg.outcome.n_hours, vocabulary=vocabulary)
        features, kept_ids = censorer(features, outcome_censor)
        pids = [pid for i, pid in enumerate(pids) if i in kept_ids]
        return features, pids

    def _process_outcomes(self, outcomes, pids):
        self._select_outcomes_for_patients(outcomes, pids)
        outcomes, censor_outcomes, pids = self._retrieve_outcomes(outcomes)
        return outcomes, censor_outcomes
        
    def _prepare_mlm_features(self):
        """Load data, truncate"""

        train_features, train_pids, val_features, val_pids, vocabulary = self._load_tokenized_data()
        torch.save(vocabulary, join(self.run_folder, VOCABULARY_FILE))
        
        # Patient Selection
        train_features, train_pids = DatasetPreparer._select_random_subset(
            train_features, train_pids, self.cfg.data.num_train_patients)
        val_features, val_pids = DatasetPreparer._select_random_subset(
            val_features, val_pids, self.cfg.data.num_val_patients)
        self._save_pids(train_pids, val_pids)
        if self.cfg.data.get("remove_background", False):
            logger.info("Removing background tokens")
            train_features = self._remove_background(train_features, vocabulary)
            val_features = self._remove_background(val_features, vocabulary)
        # Truncation
        logger.info(f"Truncating data to {self.cfg.data.truncation_len} tokens")
        truncator = Truncator(max_len=self.cfg.data.truncation_len, 
                              sep_token=vocabulary['[SEP]'])
        train_features = truncator(train_features)
        val_features = truncator(val_features)
        return train_features, val_features, vocabulary

    def _remove_background(self, features, vocabulary):
        """Remove background tokens from features and the first sep token following it"""
        background_tokens = set([v for k, v in vocabulary.items() if k.startswith('BG_')])
        example_concepts = features['concept'][0] # Assume that all patients have the same background length
        remove_indices = set([i for i, concept in enumerate(example_concepts) if concept in background_tokens])
        if vocabulary['[SEP]'] in example_concepts:
            remove_indices.add(len(remove_indices)+1)

        new_features = {}
        for k, token_lists in features.items():
            new_features[k] = [[token for idx, token in enumerate(tokens) if idx not in remove_indices] 
                           for tokens in token_lists]
        return new_features

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

    def _exclude_pretrain_patients(self, features, pids,  mode):
        pretrain_pids = set(torch.load(join(self.cfg.paths.model_path, f'pids_{mode}.pt')))
        kept_indices = [i for i, pid in enumerate(pids) if pid not in pretrain_pids]
        pids = [pid for i, pid in enumerate(pids) if i in kept_indices]
        for k, v in features.items():
            features[k] = [v[i] for i in kept_indices]
        return features, pids
    
    def _retrieve_outcomes(self, all_outcomes):
        """From the configuration, load the outcomes and censor outcomes.
        Access pids, the outcome of interest and the censoring outcome."""
        
        outcomes = all_outcomes.get(self.cfg.outcome.type, [None]*len(all_outcomes[PID_KEY]))
        censor_outcomes = all_outcomes.get(self.cfg.outcome.get('censor_type', None), [None]*len(outcomes))
        pids = all_outcomes[PID_KEY]
        return outcomes, censor_outcomes, pids

    @staticmethod
    def _select_random_subset(features, pids, num_patients, seed=0):#
        np.random.seed(seed)
        indices = np.arange(len(pids))
        np.random.shuffle(indices)
        indices = indices[:num_patients]
        pids = [pid for i, pid in enumerate(pids) if i in indices]
        for k, v in features.items():
            features[k] = [v[i] for i in indices]
        return features, pids

    def _save_pids(self, train_pids, val_pids):
        torch.save(train_pids, join(self.run_folder, 'pids_train.pt'))
        torch.save(val_pids, join(self.run_folder, 'pids_val.pt'))
    
    @staticmethod
    def _select_outcomes_for_patients(all_outcomes, pids):
        pids = set(pids)
        return {k:[v[i] for i, pid in enumerate(all_outcomes[PID_KEY]) if pid in pids] for k, v in all_outcomes.items()}




