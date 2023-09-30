class DataAdapter:
    def adapt_to_single_visit(self, features: dict) -> dict:
        features = self.adapt_to_single_visit(features)
        features = self.remove_duplicate_codes(features)

        return features

    def adapt_to_behrt(self, features: dict) -> dict:
        """
        Delete abspos
        Convert age to ints (and make negatives to 0)
        Get position_ids from segments (almost identical, but [3, 4, 3] -> [3, 4, 5])
        Convert segments to 0, ..., 1, ..., 0
        """
        del features["abspos"]
        features["age"] = [self.convert_to_int(ages) for ages in features["age"]]
        features["position_ids"] = [
            self.convert_segments(segments, func=lambda x: x + 1)
            for segments in features["segment"]
        ]
        features["segment"] = [
            self.convert_segments(segments, func=lambda x: 1 - x)
            for segments in features["segment"]
        ]
        return features

    @staticmethod
    def one_hot(features: dict, vocabulary: dict, prepend_age=True) -> list:
        X = []
        for i, patient in enumerate(features["concept"]):
            x = []
            if prepend_age:
                x.append(features["age"][i][-1])  # First token is age
            x.append([0] * len(vocabulary))  # second and forward is one-hot
            for code in set(patient):
                x[code] = 1
            X.append(x)

        return X

    @staticmethod
    def convert_to_int(ages: list):
        return [int(age) if age > 0 else 0 for age in ages]

    @staticmethod
    def convert_segments(segments: list, func: callable):
        position_ids = []
        flag = 0
        for i, segment in enumerate(segments):
            position_ids.append(flag)
            if i < len(segments) - 1:
                if segment != segments[i + 1]:  # If segment changes
                    # flag += 1 for position_ids, flag = 1 - flag for segments
                    flag = func(flag)
        return position_ids

    @staticmethod
    def convert_to_singlevisits(features: dict) -> dict:
        single_visits = {key: [] for key in features}

        for i in range(len(features["concept"])):
            patient = {key: values[i] for key, values in features.items()}
            segments = patient["segment"]
            
            # Get idxs of background (first segment 1)
            background_idx = segments.index(1)

            # Initialize every visit with background info
            split_visits = {
                key: [values[:background_idx] for _ in range(max(segments))]
                for key, values in patient.items()
            }

            # Add age, abspos and segment (segment should just be 0, 1, ..., 1) and the rest normal
            for i in range(background_idx, len(segments)):
                visit_idx = segments[i] - 1
                for key, values in patient.items():
                    if key == "segment":
                        split_visits[key][visit_idx].append(1)
                    else:
                        split_visits[key][visit_idx].append(values[i])

            # Add each visit to the single_visits dict
            for key, values in split_visits.items():
                single_visits[key].extend(values)

        return single_visits

    @staticmethod
    def remove_duplicate_codes(features: dict) -> dict:
        for i, patient in enumerate(features["concept"]):
            idxs = []
            tracker = set()
            for j, concept in enumerate(patient):
                if concept not in tracker:
                    idxs.append(j)
                    tracker.add(concept)

            for key, values in features.items():
                features[key][i] = [values[i][j] for j in idxs]

        return features
