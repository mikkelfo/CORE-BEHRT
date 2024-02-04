from math import ceil, floor
from typing import List

MIN_ABSPOS_MONTHS = -120 # 10 years into the past
MAX_ABSPOS_MONTHS = 37 # 3 years into the future
HOURS_IN_MONTH = 730.001 # 30.42 days


class BaseAdapter:
    @staticmethod
    def adapt_features(features: dict)->dict:
        """Adapt features to behrt embeddings format. Continuous age is converted to integer and segment is stored as position_ids. 
        New segment is created from old segment."""
        if 'age' in features:
            features['age'] = [BehrtAdapter.convert_ages_to_int(ages) for ages in features['age']] 
        if 'segment' in features:
            features['position_ids'] = features['segment'] # segment is the same as position_ids
            features['segment'] = [BehrtAdapter.convert_segment(segments) for segments in features['segment']]
        return features
    
    @staticmethod
    def convert_ages_to_int(ages: list, min_age=0, max_age=120) -> list:
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
class BehrtAdapter(BaseAdapter):
    @staticmethod
    def adapt_features(features: dict)->dict:
        """Adapt features to behrt embeddings format. Continuous age is converted to integer and segment is stored as position_ids. 
        New segment is created from old segment."""
        features = BaseAdapter.adapt_features(features)
        del features['abspos']
        return features
    
class DiscreteAbsposAdapter(BaseAdapter):
    @staticmethod
    def adapt_features(features: dict)->dict:
        """Adapt features to behrt embeddings format. Continuous age is converted to integer and segment is stored as position_ids. 
        New segment is created from old segment."""
        features = BaseAdapter.adapt_features(features)
        if 'abspos' in features:
            # max_abpos = DiscreteAbsposAdapter.get_maximum(features['abspos'])
            # max_abspos_months = DiscreteAbsposAdapter.hours2months(max_abpos) 
            features['abspos'] = [DiscreteAbsposAdapter.convert_abspos(
                 abspos, min_abspos=MIN_ABSPOS_MONTHS, max_abspos=MAX_ABSPOS_MONTHS) \
                    for abspos in features['abspos']]
        return features

    @staticmethod
    def get_maximum(feature: List[List])->float:
        """Get the maximum value of a feature"""
        return max([max(f) for f in feature])

    @staticmethod
    def convert_abspos(abspos: list, min_abspos: float=None, max_abspos: float=None) -> list:
        """
        1. Convert abspos to weeks 
        2. Cutoff values with min_abspos and max_abspos given in months
        3. Map values to 0 to min_abpos+max_abspos.
        """
        converted_abspos = []
        for pos in abspos:
            pos = DiscreteAbsposAdapter.hours2months(pos)
            if pos < min_abspos:
                pos = min_abspos
            elif pos > max_abspos: # this should not occur since we're using the maximum value
                pos = max_abspos
            pos = ceil(pos) # round up 
            min_abspos = floor(min_abspos) # round down to make sure, we're not getting negative values
            pos = pos - min_abspos
            converted_abspos.append(pos)
        return converted_abspos
    
    @staticmethod
    def hours2months(hours: float)->float:
        """Convert hours to months"""
        return hours/HOURS_IN_MONTH