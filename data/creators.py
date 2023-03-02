import pandas as pd
from datetime import datetime
import glob
import itertools
from dateutil.parser import parse
import os

class BaseCreator():
    def __init__(self, config):
        self.config: dict = config

    def __call__(self, concepts):
        return self.create(concepts)

    def create(self, concepts=None):
        # Get all concept files
        concept_paths = os.path.join(self.config.data_dir, 'concept.*')
        path = glob.glob(concept_paths)

        # Filter out concepts files
        if self.config.get('concepts') is not None:
            path = [p for p in path if p.split('.')[1] in self.config.concepts]
        
        # Load concepts
        concepts = pd.concat([self.read_file(p) for p in path]).reset_index(drop=True)
        
        concepts = concepts.sort_values('TIMESTAMP')

        return concepts

    def read_file(self, file_path) -> pd.DataFrame:
        file_type = file_path.split(".")[-1]
        if file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type == 'parquet':
            df = pd.read_parquet(file_path)

        for col in self._detect_date_columns(df):
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = df[col].dt.tz_localize(None)

        return df

    def _detect_date_columns(self, df):
        date_columns = []
        for col in df.columns:
            if isinstance(df[col], datetime):
                continue
            if 'TIME' in col.upper() or 'DATE' in col.upper():
                try:
                    first_non_na = df.loc[df[col].notna(), col].iloc[0]
                    parse(first_non_na)
                    date_columns.append(col)
                except:
                    continue
        return date_columns

class AgeCreator(BaseCreator):
    feature = id = 'age'
    def create(self, concepts):
        pi_info_path = os.path.join(self.config.data_dir, 'patients_info.csv')
        patients_info = self.read_file(pi_info_path)
        # Create PID -> BIRTHDATE dict
        birthdates = pd.Series(patients_info['BIRTHDATE'].values, index=patients_info['PID']).to_dict()
        # Calculate approximate age
        ages = (concepts['TIMESTAMP'] - concepts['PID'].map(birthdates)).dt.days // 365.25

        concepts['AGE'] = ages
        return concepts

class AbsposCreator(BaseCreator):
    feature = id = 'abspos'
    def create(self, concepts):
        abspos = self.config.features.abspos
        origin_point = datetime(abspos.year, abspos.month, abspos.day)
        # Calculate days since origin point
        abspos = (concepts['TIMESTAMP'] - origin_point).dt.days

        concepts['ABSPOS'] = abspos
        return concepts

class SegmentCreator(BaseCreator):
    feature = id = 'segment'
    def create(self, concepts):
        # Infer NaNs in ADMISSION_ID
        concepts['ADMISSION_ID'] = self._infer_admission_id(concepts)

        segments = concepts.groupby('PID')['ADMISSION_ID'].transform(lambda x: pd.factorize(x)[0]+1)

        concepts['SEGMENT'] = segments
        return concepts

    def _infer_admission_id(self, df):
        bf = df.sort_values('PID')
        mask = bf['ADMISSION_ID'].fillna(method='ffill') != bf['ADMISSION_ID'].fillna(method='bfill')   # Find NaNs between similar admission IDs
        bf.loc[mask, 'ADMISSION_ID'] = bf.loc[mask, 'ADMISSION_ID'].map(lambda x: 'unq_') + list(map(str, range(mask.sum())))   # Assign unique IDs to non-inferred NaNs
        bf['ADMISSION_ID'] = bf['ADMISSION_ID'].fillna(method='ffill')  # Assign neighbour IDs to inferred NaNs
        
        return bf['ADMISSION_ID']

class BackgroundCreator(BaseCreator):
    id = 'background'
    prepend_token = "B_"
    def create(self, concepts):
        pi_info_path = os.path.join(self.config.data_dir, 'patients_info.csv')
        patients_info = self.read_file(pi_info_path)

        background = {
            'PID': patients_info['PID'].tolist() * len(self.config.features.background),
            'CONCEPT': itertools.chain.from_iterable(
                [(self.prepend_token + patients_info[col].astype(str)).tolist() for col in self.config.features.background])
        }

        for feature in self.config.features:
            if feature in ['age', 'abspos', 'segment']:
                background[feature.upper()] = 0

        background = pd.DataFrame(background)

        return pd.concat([background, concepts])


""" SIMPLE EXAMPLES """
class SimpleValueCreator(BaseCreator):
    id = 'value'
    def create(self, concepts):
        concepts['CONCEPT'] = concepts['CONCEPT'] + '_' + concepts['VALUE'].astype(str)

        return concepts
        
class QuartileValueCreator(BaseCreator):
    id = 'quartile_value'
    def create(self, concepts):
        quartiles = concepts.groupby('CONCEPT')['value'].transform(lambda x: pd.qcut(x, 4, labels=False))
        concepts['CONCEPT'] = concepts['CONCEPT'] + '_' + quartiles.astype(str)

        return concepts

    def _create_quartile_concepts(self, df):
        quartiles = pd.qcut(df['VALUE'], 4, labels=False)
        
        return df['CONCEPT'] + '_' + quartiles.astype(str)