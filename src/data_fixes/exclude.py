import os
import torch


class Excluder:
    def __init__(self, config):
        self.config = config
        self.dir = config.paths.extra_dir

        # NB: Order matters
        self.functions = {
            "underaged": self.exclude_underaged_visits,
            "rare_concepts": self.exclude_rare_concepts,
            "short_sequences": self.exclude_short_sequences,  # Always last
        }

    def __call__(self, features: dict, filename: str = "excluder_kept_indices") -> dict:
        for function in self.functions:
            if function not in self.config.excluder:
                continue

            # Call function (init, call, update)
            features = self._call_helper(
                features, function, **self.config.excluder[function]
            )

        # Finally, remove empty lists
        features = self.wrapup_exclusion(features, filename=filename)

        return features

    def _call_helper(self, features: dict, function: str, **kwargs) -> dict:
        # Initialize kept indices (empty dict)
        kept_indices = self.init_kept_indices(features)
        # Call function
        kept_indices = self.functions[function](features, kept_indices, **kwargs)
        # Update features based on kept indices
        features = self.update_features(features, kept_indices)

        return features

    def _call_standalone(
        self, features: dict, function: str, filename: str, **kwargs
    ) -> dict:
        features = self._call_helper(features, function, **kwargs)
        features = self.wrapup_exclusion(features, filename=filename)

        return features

    def wrapup_exclusion(self, features: dict, filename: str) -> dict:
        kept_patients = self.get_kept_patients(features)
        features = self.remove_empty_lists(features, kept_patients)

        torch.save(kept_patients, os.path.join(self.dir, f"{filename}.pt"))

        return features

    @staticmethod
    def exclude_rare_concepts(features: dict, kept_indices: dict, **kwargs):
        # First get count of all concepts
        unique_codes = {}
        for patient in features["concept"]:
            for code in patient:
                unique_codes[code] = unique_codes.get(code, 0) + 1

        # Then exclude all concepts with count below threshold
        for i, patient in enumerate(features["concept"]):
            valid_indices = [
                idx
                for idx, code in enumerate(patient)
                if unique_codes[code] >= kwargs["min_count"]
            ]
            kept_indices[i].update(valid_indices)

        return kept_indices

    @staticmethod
    def exclude_short_sequences(features: dict, kept_indices: dict, **kwargs):
        if kwargs.get("vocabulary") is not None:
            vocabulary = {v: k for k, v in kwargs["vocabulary"].items()}
        # Get all indices of patients with sequences shorter than threshold
        for i, concepts in enumerate(features["concept"]):
            if kwargs.get("vocabulary") is not None:
                unique_codes = set(
                    [code for code in concepts if not vocabulary[code].startswith("[")]
                )
            else:
                unique_codes = set(
                    [code for code in concepts if not code.startswith("[")]
                )
            if len(unique_codes) >= kwargs["min_concepts"]:
                kept_indices[i].update(range(len(concepts)))

        return kept_indices

    @staticmethod
    def exclude_underaged_visits(features: dict, kept_indices: dict, **kwargs) -> dict:
        for i in range(len(features["concept"])):
            # Find all valid ages (-1 is valid due to background sentence)
            valid_ages = [
                age >= kwargs["min_age"] or age == -1 for age in features["age"][i]
            ]

            # Find all segments with valid ages
            invalid_segments = set(
                [
                    segment
                    for j, segment in enumerate(features["segment"][i])
                    if not valid_ages[j]
                ]
            )

            # Find indices of segments to keep
            valid_indices = [
                j
                for j, segment in enumerate(features["segment"][i])
                if segment not in invalid_segments
            ]

            # Update kept indices
            kept_indices[i].update(valid_indices)

        return kept_indices

    @staticmethod
    def init_kept_indices(features: dict) -> dict:
        return {i: set() for i in range(len(features["concept"]))}

    @staticmethod
    def update_features(features: dict, kept_indices: dict) -> dict:
        for i, idxs in kept_indices.items():
            for key, values in features.items():
                features[key][i] = [values[i][j] for j in idxs]

        return features

    @staticmethod
    def get_kept_patients(features: dict) -> list:
        return [i for i, v in enumerate(features["concept"]) if v]

    @staticmethod
    def remove_empty_lists(features: dict, kept_patients: list) -> dict:
        for key, values in features.items():
            features[key] = [values[i] for i in kept_patients]

        return features
