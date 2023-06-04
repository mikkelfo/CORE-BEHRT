import glob
import os
from datetime import datetime
from os.path import join, split

import dateutil
import pandas as pd
import pyarrow.parquet as pq


class ConceptLoader():
    def __init__(self, concepts: list = ['diagnose', 'medication'], data_dir: str = 'formatted_data', patients_info: str = 'patients_info.csv', batch_size:int = 100000, chunksize: int = 100000):

        concept_paths = glob.glob(os.path.join(data_dir, 'concept.*'))
        # Filter out concepts files
        self.concept_paths = [p for p in concept_paths if (split(p)[1]).split('.')[1] in concepts]
        self.patients_df = self._read_file(join(data_dir, patients_info))
        self.patient_ids = self.patients_df['PID'].unique().tolist()
        self.chunksize = chunksize
        self.batch_size = batch_size

    def __call__(self):
        for patients in self.get_patient_batch():
            # Load concepts
            concepts = pd.concat([self._read_file(p, patients) for p in self.concept_paths], ignore_index=True).drop_duplicates()
            concepts = concepts.sort_values('TIMESTAMP')
            # Load patient data
            patients_info = self.patients_df[self.patients_df['PID'].isin(patients)]
            yield concepts, patients_info

    def _read_file(self, file_path, patients: list = None):
        file_type = file_path.split(".")[-1]
        if file_type == 'csv':
            if patients is None:
                df = pd.read_csv(file_path)
            else:
                df_chunks = pd.read_csv(file_path, chunksize=self.chunksize)
        elif file_type == 'parquet':
            if patients is None:
                df = pq.read_table(file_path).to_pandas()
            else:
                df_chunks = ParquetIterator(file_path, self.chunksize)
        else:
            raise ValueError(f'Unknown file type {file_type}')
        if patients is None:
            return self.handle_datetime_columns(df)
        else:
            chunks = []
            for chunk in df_chunks:
                chunk = self.handle_datetime_columns(chunk)
                filtered_chunk = chunk[chunk['PID'].isin(patients)]  # assuming 'PID' is the patient ID column in this file too
                chunks.append(filtered_chunk)

            # Combine all the filtered chunks
            filtered_df = pd.concat(chunks, ignore_index=True)
            return filtered_df

    def handle_datetime_columns(self, df):
        for col in self.detect_date_columns(df):
            df[col] = df[col].apply(lambda x: x[:10] if isinstance(x, str) else x)
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = df[col].dt.tz_localize(None)
        return df

    @staticmethod
    def detect_date_columns(df: pd.DataFrame):
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
    
    def get_patient_batch(self):
        """Yield successive batches of patient IDs from patient_ids."""
        for i in range(0, len(self.patient_ids), self.batch_size):
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