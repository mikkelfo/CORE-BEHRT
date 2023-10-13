
class BehrtAdapter:
    def adapt_features(self, features: dict):
        del features['abspos']
        features['age'] = [self.convert_to_int(ages) for ages in features['age']]
        features['position_ids'] = features['segment'] # segment is the same as position_ids
        features['segment'] = [self.convert_segment(segments) for segments in features['segment']]
        return features
    @staticmethod
    def convert_to_int(ages: list):
        """Convert ages to int and replace negative values with 0"""
        converted_ages = []
        for age in ages:
            if age<0:
                converted_ages.append(0)
            elif age>119:
                converted_ages.append(119)
            else:
                converted_ages.append(int(age))
        return converted_ages
    @staticmethod
    def convert_segment(segments: list):
        """From segment AABBCC to segment 001100111"""
        converted_segments = []
        flag = 0
        for i, segment in enumerate(segments):
            converted_segments.append(flag)
            if i < len(segments) - 1:
                if segment != segments[i+1]:
                    flag = 1 - flag
        return converted_segments