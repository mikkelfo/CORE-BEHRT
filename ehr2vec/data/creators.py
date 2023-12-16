import itertools

import pandas as pd
from data.utils import Utilities


class BaseCreator():
    def __init__(self, config: dict):
        self.config = config

    def __call__(self, concepts: pd.DataFrame, patients_info: pd.DataFrame)-> pd.DataFrame:
         # Create PID -> BIRTHDATE dict
        if 'BIRTHDATE' not in patients_info.columns:
            if 'DATE_OF_BIRTH' in patients_info.columns:
                patients_info = patients_info.rename(columns={'DATE_OF_BIRTH': 'BIRTHDATE'})
            else:
                raise KeyError('BIRTHDATE column not found in patients_info')
        return self.create(concepts, patients_info)

class AgeCreator(BaseCreator):
    feature = id = 'age'
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame)-> pd.DataFrame:
        birthdates = pd.Series(patients_info['BIRTHDATE'].values, index=patients_info['PID']).to_dict()
        # Calculate approximate age
        ages = (concepts['TIMESTAMP'] - concepts['PID'].map(birthdates)).dt.days / 365.25
        round = self.config.age.get('round', False)
        if round:
            ages = ages.round(self.config.age.round.get('decimal', 0))

        concepts['AGE'] = ages
        return concepts

class AbsposCreator(BaseCreator):
    
    feature = id = 'abspos'
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame)-> pd.DataFrame:
        abspos = Utilities.get_abspos_from_origin_point(concepts['TIMESTAMP'], self.config.abspos)
        concepts['ABSPOS'] = abspos
        return concepts

class SegmentCreator(BaseCreator):
    feature = id = 'segment'
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame)-> pd.DataFrame:
        if 'ADMISSION_ID' in concepts.columns:
            seg_col = 'ADMISSION_ID'
        elif 'SEGMENT' in concepts.columns:
            seg_col = 'SEGMENT'
        else:
            raise KeyError('No segment column found in concepts')
    
        segments = concepts.groupby('PID')[seg_col].transform(lambda x: pd.factorize(x)[0]+1) # change back
        
        concepts['SEGMENT'] = segments
        return concepts

class BackgroundCreator(BaseCreator):
    id = 'background'
    prepend_token = "BG_"
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame)-> pd.DataFrame:
        # Create background concepts
        background = {
            'PID': patients_info['PID'].tolist() * len(self.config.background),
            'CONCEPT': itertools.chain.from_iterable(
                [(self.prepend_token + col + '_' +patients_info[col].astype(str)).tolist() for col in self.config.background])
        }

        if 'segment' in self.config:
            background['SEGMENT'] = 0

        if 'age' in self.config:
            background['AGE'] = -1

        if 'origin_point' in self.config:
            abspos = Utilities.get_abspos_from_origin_point([patients_info['BIRTHDATE']], self.config.abspos)
            background['ABSPOS'] = abspos * len(self.config.background)
            

        # Prepend background to concepts
        background = pd.DataFrame(background)
        return pd.concat([background, concepts])


""" SIMPLE EXAMPLES """
class SimpleValueCreator(BaseCreator):
    id = 'value'
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        concepts['CONCEPT'] = concepts['CONCEPT'] + '_' + concepts['VALUE'].astype(str)

        return concepts
        
class QuartileValueCreator(BaseCreator):
    id = 'quartile_value'
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        quartiles = concepts.groupby('CONCEPT')['value'].transform(lambda x: pd.qcut(x, 4, labels=False))
        concepts['CONCEPT'] = concepts['CONCEPT'] + '_' + quartiles.astype(str)

        return concepts

