from math import ceil, floor
from typing import List

import numpy as np

from ehr2vec.common.utils import iter_patients
from ehr2vec.data.utils import Utilities
import logging

MIN_ABSPOS_MONTHS = -120 # 10 years into the past
MAX_ABSPOS_MONTHS = 37 # 3 years into the future
HOURS_IN_MONTH = 730.001 # 30.42 days


logger = logging.getLogger(__name__)

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
        # del features['abspos']
        return features
    
class PLOSAdapter:
    def __init__(self, threshold_in_days: int=None):
        self.threshold_in_hours = threshold_in_days*24 if threshold_in_days else None

    def adapt_features(self,features: dict)->dict:
        """Adapt features to behrt embeddings format. Continuous age is converted to integer and segment is stored as position_ids. 
        New segment is created from old segment."""
        features = self.get_prolonged_length_of_stay(features)
        return features

    def get_prolonged_length_of_stay(self,features: dict)->dict:
        """Calculate whether any hospital stay, which was longer than N days occured"""
        
        prolonged_lengths_of_stay = []
        logger.info("Examples of plos calculation")
        for patient in iter_patients(features):
            prolonged_lengths_of_stay.append(
                self.get_prolonged_length_of_stay_for_patient(patient))
        logger.info(f'Prevalence of prolonged length of stay: {sum(prolonged_lengths_of_stay)/len(prolonged_lengths_of_stay)}')
        features['PLOS'] = prolonged_lengths_of_stay
        return features
    
    def get_prolonged_length_of_stay_for_patient(self, patient):
        """Check if any hospital stay was longer than N days"""
        segments = np.array(patient['segment'])
        abspos = np.array(patient['abspos'])
        segments_one_hot = Utilities.get_segments_one_hot(segments)
        segment_abspos = segments_one_hot * abspos
        min_values = np.where(segment_abspos != 0, segment_abspos, np.inf).min(axis=1)
        max_values = np.where(segment_abspos != 0, segment_abspos, -np.inf).max(axis=1)
        diff = max_values - min_values

        return (diff > self.threshold_in_hours).any().astype(int)
    
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
    
