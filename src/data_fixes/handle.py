import pandas as pd


class Handler:
    def __init__(self, cfg) -> None:
        self.cfg = cfg.handler

        function_dict = {
            "ages": self.handle_incorrect_ages,
            "nans": self.handle_nans,
        }
        self.functions = {
            col: (function_dict[col], options if options else {})
            for col, options in self.cfg.functions.items()
        }

    def __call__(self, features: dict) -> dict:
        return self.handle(features)

    def handle(self, features: dict) -> dict:
        handled_patients = {k: [] for k in features}
        for patient in self._iter_patients(features):
            # Apply functions
            for func, options in self.functions.values():
                patient = func(patient, **options)

            # Normalize segments if necessary
            patient["segment"] = self.normalize_segments(patient["segment"])

            # Append patient to handled patients
            for key, values in patient.items():
                handled_patients[key].append(values)

        return handled_patients

    def handle_incorrect_ages(self, patient: dict, **kwargs) -> dict:
        # Get correct indices
        correct_indices = [
            i
            for i, age in enumerate(patient["age"])
            if kwargs["min"] <= age <= kwargs["max"]
        ]
        correct_indices = set(correct_indices)

        return self.drop_or_fill(patient, correct_indices, **kwargs)

    def handle_nans(self, patient: dict, **kwargs) -> dict:
        correct_indices = set()
        for values in patient.values():
            correct_indices.update([i for i, v in enumerate(values) if not pd.isna(v)])

        return self.drop_or_fill(patient, correct_indices, **kwargs)

    def fill(
        self,
        patient: dict,
        correct_indices: set,
        **kwargs,
    ) -> dict:
        concept_fill = kwargs.get("concept_fill", self.cfg.concept_fill)
        num_fill = kwargs.get("num_fill", self.cfg.num_fill)
        for key, values in patient.items():
            filler = concept_fill if key == "concept" else num_fill
            patient[key] = [
                v if i in correct_indices else filler for i, v in enumerate(values)
            ]

        return patient

    def drop_or_fill(self, patient: dict, correct_indices: set, **kwargs) -> dict:
        if kwargs.get("drop", self.cfg.drop):
            return self.drop(patient, correct_indices)
        else:
            return self.fill(patient, correct_indices, **kwargs)

    @staticmethod
    def drop(patient: dict, correct_indices: list) -> dict:
        for key, values in patient.items():
            patient[key] = [values[i] for i in correct_indices]

        return patient

    @staticmethod
    def _iter_patients(features: dict) -> dict:
        for i in range(len(features["concept"])):
            yield {k: v[i] for k, v in features.items()}

    @staticmethod
    def normalize_segments(segments: list) -> list:
        segment_set = sorted(set(segments))
        correct_segments = list(range(len(segment_set)))
        converter = {k: v for (k, v) in zip(segment_set, correct_segments)}

        return [converter[segment] for segment in segments]
