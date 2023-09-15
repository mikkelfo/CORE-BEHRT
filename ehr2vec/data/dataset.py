import logging

import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)  # Get the logger for this module


class BaseEHRDataset(Dataset):
    def __init__(self, features:dict):
        self.features = features

    def _getpatient(self, index):
        return {
            key: torch.as_tensor(values[index]) for key, values in self.features.items()
        }

    def __len__(self):
        return len(self.features["concept"])

    def __getitem__(self, index):
        return self._getpatient(index)

class MLMDataset(BaseEHRDataset):
    def __init__(
        self,
        features: dict,
        vocabulary: dict,
        masked_ratio:float=0.3,
        ignore_special_tokens:bool=True,
    ):
        super().__init__(features)
        self.vocabulary = vocabulary
        self.masked_ratio = masked_ratio
        if ignore_special_tokens:
            self.n_special_tokens = len(
                [token for token in vocabulary if token.startswith("[")]
            )
        else:
            self.n_special_tokens = 0

    def __getitem__(self, index):
        patient = super().__getitem__(index)

        masked_concepts, target = self._mask(patient)
        patient["concept"] = masked_concepts
        patient["target"] = target

        return patient

    def _mask(self, patient: dict):
        concepts = patient["concept"]

        N = len(concepts)

        # Initialize
        masked_concepts = torch.clone(concepts)
        target = torch.ones(N, dtype=torch.long) * -100

        # Apply special token mask and create MLM mask
        eligible_mask = masked_concepts >= self.n_special_tokens
        eligible_concepts = masked_concepts[eligible_mask]  # Ignore special tokens
        rng = torch.rand(len(eligible_concepts))  # Random number for each token
        masked = rng < self.masked_ratio  # Mask tokens with probability masked_ratio

        # Get masked MLM concepts
        selected_concepts = eligible_concepts[masked]  # Select set % of the tokens
        adj_rng = rng[masked].div(self.masked_ratio)  # Fix ratio to 0-100 interval

        # Operation masks (80% mask, 10% replace with random word, 10% keep token)
        rng_mask = adj_rng < 0.8
        rng_replace = (0.8 <= adj_rng) & (adj_rng < 0.9)
        # rng_keep = adj_rng >= 0.9 # Redundant

        # Apply operations (Mask, replace, keep)
        selected_concepts = torch.where(
            rng_mask, self.vocabulary["[MASK]"], selected_concepts
        )  # Replace with [MASK]
        selected_concepts = torch.where(
            rng_replace,
            torch.randint(
                self.n_special_tokens, len(self.vocabulary), (len(selected_concepts),)
            ),
            selected_concepts,
        )  # Replace with random word
        # selected_concepts = torch.where(rng_keep, selected_concepts, selected_concepts)       # Redundant

        # Update outputs (nonzero for double masking)
        target[eligible_mask.nonzero()[:, 0][masked]] = eligible_concepts[
            masked
        ]  # Set "true" token
        masked_concepts[
            eligible_mask.nonzero()[:, 0][masked]
        ] = selected_concepts  # Sets new concepts

        return masked_concepts, target
    
class HierarchicalMLMDataset(MLMDataset):
    def __init__(
        self,
        features: dict,
        vocabulary: dict,
        h_vocabulary: dict,
        tree=None,
        tree_matrix=None,
        masked_ratio=0.3,
        ignore_special_tokens=True,
    ):
        super().__init__(features, vocabulary, masked_ratio, ignore_special_tokens)

        self.h_vocabulary = h_vocabulary
        self.tree_matrix = tree_matrix
        self.tree = tree
        
        self.tree_matrix = tree.get_tree_matrix()
        self.tree_matrix_sparse = self.tree_matrix.to_sparse()
        self.leaf_counts = tree.get_leaf_counts()
        self.target_mapping = self.get_target_mapping()
        self.levels = self.tree_matrix.shape[0]

    def __getitem__(self, index):
        patient = super().__getitem__(index)

        target_mask = patient['target'] != -100
        patient['attention_mask'] = target_mask

        patient['target'] = self._hierarchical_target(patient['target'][patient['attention_mask']])

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
    
    
class CensorDataset(BaseEHRDataset):
    """
    outcomes: absolute position when outcome occured for each patient 
    outcomes is a list of the outcome timestamps to predict
    """

    def __init__(
        self, features: dict, outcomes: list):
        super().__init__(features)
        self.outcomes = outcomes
        
    def __getitem__(self, index: int) -> dict:
        patient = super().__getitem__(index)
        patient["target"] = float(pd.notna(self.outcomes[index]))

        return patient






