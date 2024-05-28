import torch
from typing import Tuple


class ConceptMasker:
    def __init__(self, vocabulary: dict, 
                 select_ratio: float, masking_ratio: float = 0.8, replace_ratio: float=0.1,
                 ignore_special_tokens: bool=True) -> None:
        """Mask concepts for MLM.
        Args:
            vocabulary: Vocabulary
            select_ratio: Ratio of tokens to consider in the loss
            masking_ratio: Ratio of tokens to replace with [MASK]
            replace_ratio: Ratio of tokens to replace with random word
        """
        
        self.vocabulary = vocabulary
        self.n_special_tokens = len([token for token in vocabulary if token.startswith("[")]) if ignore_special_tokens else 0
        self.select_ratio = select_ratio # Select ratio of tokens to consider in the loss
        self.masking_ratio = masking_ratio
        self.replace_ratio = replace_ratio

    def mask_patient_concepts(self, concepts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target = concepts.clone()
        probability_vector = torch.full(target.shape, self.select_ratio)

        # Ignore special tokens
        special_token_mask = concepts < self.n_special_tokens
        probability_vector.masked_fill_(special_token_mask, value=0.0)

        # Get MLM mask
        selected_indices = torch.bernoulli(probability_vector).bool()
        target[~selected_indices] = -100  # Ignore loss for non-selected tokens

        # Replace with [MASK]
        indices_mask = torch.bernoulli(torch.full(target.shape, self.masking_ratio)).bool() & selected_indices
        concepts[indices_mask] = self.vocabulary["[MASK]"]

        # Replace with random word
        replace_ratio = self.replace_ratio / (1 - self.masking_ratio) # Account for already masked tokens
        indices_replace = torch.bernoulli(torch.full(target.shape, replace_ratio)).bool() & selected_indices & ~indices_mask
        random_words = torch.randint(self.n_special_tokens, len(self.vocabulary), target.shape, dtype=torch.long)
        concepts[indices_replace] = random_words[indices_replace]

        return concepts, target

