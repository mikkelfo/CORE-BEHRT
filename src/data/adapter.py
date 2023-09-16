class DataAdapter:
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
    def one_hot(features: dict, vocabulary: dict) -> list:
        X = []
        for patient in features["concept"]:
            x = [0] * len(vocabulary)
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
