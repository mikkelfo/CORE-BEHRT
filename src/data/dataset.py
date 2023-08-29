import torch
import pandas as pd
from torch.utils.data import Dataset
from src.tree.helpers import build_tree


class BaseDataset(Dataset):
    def __init__(self, features: dict):
        self.features = features
        self.features = self.truncate(features)

    def truncate(self, features: dict, max_len: int = 512, sep_token: int = 2):
        truncated_features = {key: [] for key in features}
        for patient in self:  # Calls __iter__ method
            patient = self._truncate_patient(patient, max_len, sep_token)

            for key, value in patient.items():
                truncated_features[key].append(value)

        return truncated_features

    @staticmethod
    def _truncate_patient(patient, max_len, sep_token):
        # Do not truncate if patient is shorter than max_len
        if len(patient["concept"]) <= max_len:
            return patient

        # Get index of first [SEP] token (when background sentence ends)
        background_length = (patient["concept"] == sep_token).nonzero()[0][0]
        background_length += 1  # Adjust for 0-indexing
        truncation_length = max_len - background_length

        # Do not start seq with [SEP] token (SEP token is included in background sentence)
        if patient["concept"][-truncation_length] == sep_token:
            truncation_length -= 1

        return {
            key: torch.cat((value[:background_length], value[-truncation_length:]))
            for key, value in patient.items()
        }

    def _getpatient(self, index):
        return {
            key: torch.as_tensor(values[index]) for key, values in self.features.items()
        }

    def __len__(self):
        return len(self.features["concept"])

    def __getitem__(self, index):
        return self._getpatient(index)

    def __iter__(self):
        for i in range(len(self)):
            yield self._getpatient(i)


class MLMDataset(BaseDataset):
    def __init__(
        self,
        features: dict,
        vocabulary="data/processed/vocabulary.pt",
        masked_ratio=0.3,
        ignore_special_tokens=True,
    ):
        super().__init__(features)

        self.vocabulary = self.load_vocabulary(vocabulary)
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

    @staticmethod
    def load_vocabulary(vocabulary):
        if isinstance(vocabulary, str):
            return torch.load(vocabulary)
        elif isinstance(vocabulary, dict):
            return vocabulary
        else:
            raise TypeError(f"Unsupported vocabulary input {type(vocabulary)}")


class CensorDataset(BaseDataset):
    """
    n_hours can be both negative and positive (indicating before/after censor token)
    outcomes is a list of the outcome timestamps to predict
    censor_outcomes is a list of the censor timestamps to use
    """

    def __init__(
        self, features: dict, outcomes: list, censor_outcomes: list, n_hours: int
    ):
        self.features = features
        self.n_hours = n_hours
        self.outcomes = outcomes
        censored_features = self.censor(censor_outcomes)

        super().__init__(censored_features)

    def __getitem__(self, index: int) -> dict:
        patient = super().__getitem__(index)
        patient["target"] = float(pd.notna(self.outcomes[index]))

        return patient

    def censor(self, censor_outcomes: list) -> dict:
        censored_features = {key: [] for key in self.features}
        for i, patient in enumerate(self):  # Calls BaseDataset __iter__
            censor_timestamp = censor_outcomes[i]
            censored_patient = self._censor(patient, censor_timestamp)

            for key, value in censored_patient.items():
                censored_features[key].append(value)

        return censored_features

    def _censor(self, patient: dict, event_timestamp: float) -> dict:
        if pd.isna(event_timestamp):
            return patient
        else:
            # Only required when padding
            mask = patient["attention_mask"]
            N_nomask = torch.sum(mask)
            pos = patient["abspos"][:N_nomask]

            # censor the last n_hours
            dont_censor = (pos - event_timestamp - self.n_hours) <= 0

            for key, value in patient.items():
                patient[key] = value[dont_censor]

        return patient


class HierarchicalDataset(MLMDataset):
    def __init__(
        self,
        features: dict,
        tree=None,
        vocabulary="data/processed/vocabulary.pt",
        masked_ratio=0.3,
        ignore_special_tokens=True,
    ):
        super().__init__(features, vocabulary, masked_ratio, ignore_special_tokens)

        if tree is None:
            tree = build_tree()

        self.tree_matrix = tree.get_tree_matrix()
        self.tree_matrix_sparse = self.tree_matrix.to_sparse()
        self.leaf_counts = tree.get_leaf_counts()

        self.target_mapping = {
            self.vocabulary[k]: v for k, v in tree.create_target_mapping().items()
        }  # adjusts target mapping to vocabulary

    def __getitem__(self, index):
        patient = super().__getitem__(index)

        target_mask = patient["target"] != -100
        patient["target_mask"] = target_mask

        patient["target"] = self._hierarchical_target(patient["target"][target_mask])

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
            probabilities[mask, target_levels[mask]] = 1

            if (~mask).any():
                last_parents_idx = mask.sum(1) - 1
                seq_class_idx = zip(
                    last_parents_idx, target_levels[range(seq_len), last_parents_idx]
                )  # tuple indices of (class_level, class_idx)

                relevant_leaf_counts = torch.stack(
                    [
                        self.tree_matrix[class_lvl, class_idx] * self.leaf_counts
                        for class_lvl, class_idx in seq_class_idx
                    ]
                )
                relevant_leaf_probs = relevant_leaf_counts / relevant_leaf_counts.sum(
                    1
                ).unsqueeze(-1)

                unknown_targets_idx = zip(
                    *torch.where(~mask)
                )  # tuple indices of (seq_idx, level_idx)

                unknown_probabilities = torch.stack(
                    [
                        torch.matmul(
                            self.tree_matrix_sparse[lvl_idx],
                            relevant_leaf_probs[seq_idx],
                        )
                        for seq_idx, lvl_idx in unknown_targets_idx
                    ]
                )

                probabilities[~mask] = unknown_probabilities

        return probabilities
