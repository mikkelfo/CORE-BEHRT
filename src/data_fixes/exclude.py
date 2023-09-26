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

    def __call__(self, features: dict, name: str = "excluder_kept_indices") -> dict:
        for function in self.functions:
            if function not in self.config.excluder:
                continue

            kept_indices = {i: set() for i in range(len(features["concept"]))}
            # Get what indices to keep
            kept_indices = self.functions[function](
                features, kept_indices, **self.config.excluder[function]
            )
            # Exclude from patients and update features
            for i, idxs in kept_indices.items():
                for key, values in features.items():
                    features[key][i] = [values[i][j] for j in idxs]

        # Finally, remove empty lists
        for key, values in features.items():
            features[key] = [value for value in values if value]

        torch.save(kept_indices, os.path.join(self.dir, f"{name}.pt"))

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
        if kwargs["vocabulary"] is not None:
            vocabulary = {v: k for k, v in vocabulary.items()}
        # Get all indices of patients with sequences shorter than threshold
        for i, concepts in enumerate(features["concept"]):
            if kwargs["vocabulary"] is not None:
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
    def exclude_underaged_visits(features: dict, kept_indices: list, **kwargs) -> tuple:
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
