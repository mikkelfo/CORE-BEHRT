

class BehrtAdapter:
    @classmethod
    def adapt_features(cls, features: dict)->dict:
        """Adapt features to behrt embeddings format. Continuous age is converted to integer and segment is stored as position_ids. 
        New segment is created from old segment."""
        del features['abspos']
        if 'age' in features:
            features['age'] = [cls.convert_to_int(ages) for ages in features['age']] 
        if 'segment' in features:
            features['position_ids'] = features['segment'] # segment is the same as position_ids
            features['segment'] = [cls.convert_segment(segments) for segments in features['segment']]
        return features

    @staticmethod
    def convert_to_int(ages: list, min_age=-1, max_age=120) -> list:
        """Convert ages to int and replace negative values with 0 and values over 119 with 119"""
        converted_ages = []
        for age in ages:
            if age < min_age:
                age = min_age
            elif age > max_age:
                age = max_age
            converted_ages.append(int(age))
        return converted_ages

    @staticmethod
    def convert_segment(segments: list) -> list:
        """From segment AABBCC to segment 001100111"""
        converted_segments = []
        flag = 0

        for i in range(len(segments)):
            converted_segments.append(flag)

            # Check if we're not at the last segment
            if i != len(segments) - 1:
                current_segment = segments[i]
                next_segment = segments[i + 1]

                # Check if the current segment is different from the next one
                if current_segment != next_segment:
                    flag = 1 - flag

        return converted_segments

