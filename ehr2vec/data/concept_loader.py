import glob
import os
import random
from datetime import datetime
from os.path import join, split
from typing import Iterator, Tuple

import dateutil
import pandas as pd
import pyarrow.parquet as pq


class ConceptLoader():
    """Load concepts and patient data"""
    def __init__(self, concepts=['diagnose', 'medication'], data_dir: str = 'formatted_data', patients_info: str = 'patients_info.csv', **kwargs):
        self.concepts = concepts
        self.data_dir = data_dir
        self.patients_info = patients_info

    def __call__(self)->Tuple[pd.DataFrame, pd.DataFrame]:
        # Get all concept files
        concept_paths = os.path.join(self.data_dir, 'concept.*')
        path = glob.glob(concept_paths)
        # Filter out concepts files
        path = [p for p in path if (split(p)[1]).split('.')[1] in self.concepts]
        # Load concepts
        concepts = pd.concat([self._read_file(p) for p in path], ignore_index=True).drop_duplicates()
        
        concepts = concepts.sort_values('TIMESTAMP')

        # Load patient data
        patient_path = os.path.join(self.data_dir, self.patients_info)
        patients_info = self._read_file(patient_path)

        return concepts, patients_info    

    def _read_file(self, file_path: str) -> pd.DataFrame:
        file_type = file_path.split(".")[-1]
        if file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type == 'parquet':
            df = pd.read_parquet(file_path)

        for col in self.detect_date_columns(df):
            df[col] = df[col].apply(lambda x: x[:10] if isinstance(x, str) else x)
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = df[col].dt.tz_localize(None)

        return df
    @staticmethod
    def detect_date_columns(df: pd.DataFrame)-> list:
        date_columns = []
        for col in df.columns:
            if isinstance(df[col], datetime):
                continue
            if 'TIME' in col.upper() or 'DATE' in col.upper():
                try:
                    first_non_na = df.loc[df[col].notna(), col].iloc[0]
                    dateutil.parser.parse(first_non_na)
                    date_columns.append(col)
                except:
                    continue
        return date_columns

class ConceptLoaderLarge(ConceptLoader):
    """Load concepts and patient data in chunks"""
    def __init__(self, concepts: list = ['diagnosis', 'medication'], data_dir: str = 'formatted_data', patients_info: str = 'patients_info.csv', **kwargs):
        
        concept_paths = glob.glob(os.path.join(data_dir, 'concept.*'))
        # Filter out concepts files
        self.concept_paths = [p for p in concept_paths if (split(p)[1]).split('.')[1] in concepts]
        self.patients_df = self._read_file(join(data_dir, patients_info))
        self.patient_ids = self.patients_df['PID'].unique().tolist()
        random.shuffle(self.patient_ids)  # Shuffle the patient IDs
        self.chunksize = kwargs.get('chunksize', 10000)
        self.batch_size = kwargs.get('batch_size', 100000)
        self.test = kwargs.get('test', False)

    def __call__(self)->Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        for patients in self.get_patient_batch():
            # Load concepts
            concepts = pd.concat([self._read_file(p, patients) for p in self.concept_paths], ignore_index=True).drop_duplicates()
            concepts = concepts.sort_values(by=['PID','TIMESTAMP'])
            # Load patient data
            patients_info = self.patients_df[self.patients_df['PID'].isin(patients)]
            yield concepts, patients_info

    def _read_file(self, file_path: str, patients: list = None)-> pd.DataFrame:
        def get_dataframe_from_file(file_path: str, chunksize: int = None):
            file_type = file_path.split(".")[-1]
            
            if file_type == 'csv':
                return pd.read_csv(file_path, chunksize=chunksize)
            
            if file_type == 'parquet':
                if chunksize is None:
                    return pq.read_table(file_path).to_pandas()
                return ParquetIterator(file_path, chunksize)
            
            raise ValueError(f'Unknown file type {file_type}')

        # If patients list is not provided, read the entire file and handle datetime
        if patients is None:
            df = get_dataframe_from_file(file_path)
            return self.handle_datetime_columns(df)

        # If patients list is provided, read the file in chunks and filter it


        if patients is None:
            return self.handle_datetime_columns(df)
        else:
            chunks = []
            for chunk in get_dataframe_from_file(file_path, self.chunksize):
                chunk = self.handle_datetime_columns(chunk)
                filtered_chunk = chunk[chunk['PID'].isin(patients)]  # assuming 'PID' is the patient ID column in this file too
                chunks.append(filtered_chunk)

            return pd.concat(chunks, ignore_index=True)

    def handle_datetime_columns(self, df: pd.DataFrame)-> pd.DataFrame:
        """Convert all datetime columns to datetime objects"""
        for col in self.detect_date_columns(df):
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = df[col].dt.tz_localize(None)
        return df
    
    def get_patient_batch(self)-> Iterator[list]:
        """Yield successive batches of patient IDs from patient_ids."""
        for i in range(0, len(self.patient_ids), self.batch_size):
            if self.test:
                if i>5:
                    break
            yield self.patient_ids[i:i + self.batch_size]

class ParquetIterator:
    def __init__(self, filename, batch_size=100000):
        self.parquet_file = pq.ParquetFile(filename)
        self.batch_iterator = self.parquet_file.iter_batches(batch_size=batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.batch_iterator)
            return batch.to_pandas()
        except StopIteration:
            raise StopIteration