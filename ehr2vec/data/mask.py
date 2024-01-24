import torch
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)  # Get the logger for this module


class ConceptMasker:
    def __init__(self, vocabulary: dict, 
                 select_ratio: float, masking_ratio: float = 0.8, replace_ratio: float=0.1,
                 ignore_special_tokens: bool=True) -> None:
        """Mask concepts for MLM.
        Args:
            vocabulary: Vocabulary
            select_ratio: Ratio of tokens to consider in the loss
            masking_ratio: Ratio of tokens to mask
            replace_ratio: Ratio of tokens to replace with random word
        """
        
        self.vocabulary = vocabulary
        self.n_special_tokens = len([token for token in vocabulary if token.startswith("[")]) if ignore_special_tokens else 0
        self.select_ratio = select_ratio # Select ratio of tokens to consider in the loss
        self.masking_ratio = masking_ratio
        self.replace_ratio = replace_ratio
        
        assert self.select_ratio <= 1.0, "Select ratio should not exceed 1.0"
        assert self.masking_ratio + self.replace_ratio <= 1.0, "Masking ratio + replace ratio should not exceed 1.0"
        logger.info(f"Select ratio: {self.select_ratio}, masking ratio: {self.masking_ratio}, replace ratio: {self.replace_ratio},\
                    unchanged ratio: {1.0 - self.masking_ratio - self.replace_ratio}")

    def mask_patient_concepts(self, patient: dict)->Tuple[torch.Tensor, torch.Tensor]:
        concepts, target = self._initialize_masked_concepts_and_target(patient)

        # Apply special token mask and create MLM mask
        selected_indices, selected_concepts, adj_rng = self._get_concepts_eligible_for_censoring_and_mask(concepts)

        # Operation masks (self.masking_ratio: mask, self.replace_ratio: replace with random word,  keep rest)
        rng_mask = adj_rng < self.masking_ratio
        selected_concepts = torch.where(rng_mask, self.vocabulary["[MASK]"], selected_concepts)  # Replace with [MASK]

        if self.replace_ratio > 0.0:
            # Replace with random word
            rng_replace = (self.masking_ratio <= adj_rng) & (adj_rng < self.masking_ratio + self.replace_ratio)
            selected_concepts = torch.where(rng_replace,
                torch.randint(
                    self.n_special_tokens, len(self.vocabulary), (len(selected_concepts),)
                ),
                selected_concepts,
            ) 

        # Update outputs (nonzero for double masking)
        target[selected_indices] = concepts[selected_indices]  # Set "true" token
        concepts[selected_indices] = selected_concepts  # Sets new concepts

        return concepts, target
    
    @staticmethod
    def _initialize_masked_concepts_and_target(patient: Dict[str, list])->Tuple[torch.Tensor, torch.Tensor]:
        concepts = patient["concept"]
        target = torch.ones(len(concepts), dtype=torch.long) * -100
        return concepts, target
    
    def _get_concepts_eligible_for_censoring_and_mask(self, concepts: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eligible_mask = concepts >= self.n_special_tokens
        rng = torch.rand(eligible_mask.sum())  # Random number for each token
        masked = rng < self.select_ratio  # Mask tokens with probability masked_ratio
        adj_rng = rng[masked].div(self.select_ratio) # Fix ratio to 0-1 interval
        selected_indices = eligible_mask.nonzero()[:, 0][masked]
        selected_concepts = concepts[selected_indices]
        return selected_indices, selected_concepts, adj_rng

