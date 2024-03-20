import logging

import pandas as pd
import torch
from ehr2vec.data.mask import ConceptMasker
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)  # Get the logger for this module


class BaseEHRDataset(Dataset):
    def __init__(self, features:dict):
        self.features = features

    def _getpatient(self, index)->dict:
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
        select_ratio:float,
        masking_ratio:float=0.8,
        replace_ratio:float=0.1,
        ignore_special_tokens:bool=True,
    ):
        super().__init__(features)
        self.vocabulary = vocabulary
        self.masker = ConceptMasker(self.vocabulary, select_ratio, masking_ratio, replace_ratio, ignore_special_tokens)

    def __getitem__(self, index: int)->dict:
        patient = super().__getitem__(index)
        masked_concepts, target = self.masker.mask_patient_concepts(patient)
        patient["concept"] = masked_concepts
        patient["target"] = target
        if 'PLOS' in patient: # ! potential slowdown, think about better solution
            patient["PLOS"] = float(patient["PLOS"])
        return patient

    
class HierarchicalMLMDataset(MLMDataset):
    def __init__(
        self,
        features: dict,
        vocabulary: dict,
        h_vocabulary: dict,
        tree=None,
        tree_matrix=None,
        select_ratio=0.15,
        masking_ratio=0.8,
        replace_ratio=0.1,
        ignore_special_tokens=True,
    ):
        super().__init__(features, vocabulary, select_ratio, masking_ratio, replace_ratio, ignore_special_tokens)

        self.h_vocabulary = h_vocabulary
        self.tree_matrix = tree_matrix
        self.tree = tree
        
        self.tree_matrix = tree.get_tree_matrix()
        self.tree_matrix_sparse = self.tree_matrix.to_sparse()
        self.leaf_counts = tree.get_leaf_counts()
        self.target_mapping = self.get_target_mapping()
        self.levels = self.tree_matrix.shape[0]

    def __getitem__(self, index:int)->dict:
        patient = super().__getitem__(index)

        target_mask = patient['target'] != -100

        patient['attention_mask'] = target_mask
        patient['target'] = self._hierarchical_target(patient['target'][target_mask])

        return patient

    def _hierarchical_target(self, target: torch.Tensor)->torch.Tensor:
        target_levels = torch.tensor(
            [self.target_mapping[t.item()] for t in target]
        )  # Converts target to target for each level
        return self.expand_to_class_probabilities(
            target_levels
        )  # Converts target for each level to probabilities

    def expand_to_class_probabilities(self, target_levels: torch.Tensor)->torch.Tensor:
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
    
    def probs_above_target(self, probabilities:torch.Tensor, mask:torch.Tensor, target_levels:torch.Tensor)->torch.Tensor:
        """Sets the probabilities for values up to target level. Simple one-hot encoding."""
        probabilities[mask, target_levels[mask]] = 1
        return probabilities

    def probs_below_target(self, probabilities:torch.Tensor, mask:torch.Tensor, target_levels:torch.Tensor, seq_len:int)->torch.Tensor:
        """Sets the probabilities for values below target level. Uses the tree matrix to calculate the probabilities based on occurence frequency."""
        last_parents_idx = mask.sum(1)-1
        seq_class_idx = zip(last_parents_idx, target_levels[range(seq_len), last_parents_idx])  # tuple indices of (class_level, class_idx)

        relevant_leaf_counts = torch.stack([self.tree_matrix[class_lvl, class_idx] * self.leaf_counts for class_lvl, class_idx in seq_class_idx])
        relevant_leaf_probs = (relevant_leaf_counts / relevant_leaf_counts.sum(1).unsqueeze(-1))

        unknown_targets_idx = zip(*torch.where(~mask))        # tuple indices of (seq_idx, level_idx)

        unknown_probabilities = torch.stack([torch.matmul(self.tree_matrix_sparse[lvl_idx], relevant_leaf_probs[seq_idx]) for seq_idx, lvl_idx in unknown_targets_idx])

        probabilities[~mask] = unknown_probabilities
        return probabilities
    
    def get_target_mapping(self)->dict:
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
        target_mapping_temp = {
            self.h_vocabulary[k]: v for k, v in tree_target_mapping.items()
        }  
        target_maping = {}
        for vocab_key, vocab_value in self.vocabulary.items():
            h_vocab_value = self.h_vocabulary.get(vocab_key, self.h_vocabulary['[UNK]'])
            target_maping[vocab_value] = target_mapping_temp.get(h_vocab_value)
            
        return target_maping   
    
    
class BinaryOutcomeDataset(BaseEHRDataset):
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


class Time2EventDataset(BaseEHRDataset):
    """
    outcomes: absolute position when outcome occured for each patient
    censoring: absolute position when patient was censored
    """
    def __init__(
        self, features: dict, outcomes: list, censoring: list
    ):
        super().__init__(features)
        self.outcomes = outcomes
        self.censoring = censoring

    def __getitem__(self, index: int) -> dict:
        patient = super().__getitem__(index)
        patient["target"] = self.outcomes[index]- self.censoring[index]
        

        return patient



