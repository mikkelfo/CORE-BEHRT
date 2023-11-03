import logging
from typing import Dict, Tuple

import torch

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
        if ignore_special_tokens:
            self.n_special_tokens = len([token for token in vocabulary if token.startswith("[")])
        else:
            self.n_special_tokens = 0
        self.select_ratio = select_ratio # Select ratio of tokens to consider in the loss
        self.masking_ratio = masking_ratio
        self.replace_ratio = replace_ratio
        logger.info(f"Select ratio: {self.select_ratio}, masking ratio: {self.masking_ratio}, replace ratio: {self.replace_ratio}")
        assert self.select_ratio <= 1.0, "Select ratio should not exceed 1.0"
        assert self.masking_ratio + self.replace_ratio <= 1.0, "Masking and replace ratio should not exceed 1.0"

    def mask_patient_concepts(self, patient: dict)->Tuple[torch.Tensor, torch.Tensor]:
        masked_concepts, target = self._initialize_masked_concepts_and_target(patient)

        # Apply special token mask and create MLM mask
        eligible_mask, eligible_concepts, masked, rng = self._get_concepts_eligible_for_censoring_and_mask(masked_concepts)

        # Get masked MLM concepts
        selected_concepts = eligible_concepts[masked]  # Select set % of the tokens
        adj_rng = rng[masked].div(self.select_ratio)  # Fix ratio to 0-1 interval

        # Operation masks (self.masking_ratio: mask, self.replace_ratio: replace with random word,  keep rest)
        print(adj_rng)
        rng_mask = adj_rng < self.masking_ratio
        if self.replace_ratio > 0.0:
            rng_replace = (self.masking_ratio <= adj_rng) & (adj_rng < self.masking_ratio + self.replace_ratio)

        # Apply operations (Mask, replace, keep)
        selected_concepts = torch.where(rng_mask, self.vocabulary["[MASK]"], selected_concepts)  # Replace with [MASK]
        selected_concepts = torch.where(rng_replace,
            torch.randint(
                self.n_special_tokens, len(self.vocabulary), (len(selected_concepts),)
            ),
            selected_concepts,
        )  # Replace with random word
        # selected_concepts = torch.where(rng_keep, selected_concepts, selected_concepts)       # Redundant

        # Update outputs (nonzero for double masking)
        selected_indices = eligible_mask.nonzero()[:, 0][masked]
        target[selected_indices] = eligible_concepts[masked]  # Set "true" token
        masked_concepts[selected_indices] = selected_concepts  # Sets new concepts

        return masked_concepts, target
    @staticmethod
    def _initialize_masked_concepts_and_target(patient: Dict[str, list])->Tuple[torch.Tensor, torch.Tensor]:
        concepts = patient["concept"]
        N = len(concepts)
        masked_concepts = torch.clone(concepts)
        target = torch.ones(N, dtype=torch.long) * -100
        return masked_concepts, target
    def _get_concepts_eligible_for_censoring_and_mask(
            self, 
            masked_concepts: torch.Tensor
            )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eligible_mask = masked_concepts >= self.n_special_tokens
        eligible_concepts = masked_concepts[eligible_mask]  # Ignore special tokens
        rng = torch.rand(len(eligible_concepts))  # Random number for each token
        masked = rng < self.select_ratio  # Mask tokens with probability masked_ratio
        return eligible_mask, eligible_concepts, masked, rng