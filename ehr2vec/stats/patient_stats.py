from typing import Dict, List

import numpy as np
import pandas as pd

from ehr2vec.common.utils import Data
from ehr2vec.data.utils import Utilities


class PatientStats:
    def __init__(self, data: Data, censoring_times:List[float], CLS:bool=False):
        """censoring_times given in months. Does the data contain a CLS token?"""
        self.data = data
        self.censoring_times = censoring_times
        self.CLS = CLS
        self.MIN_LEN = 3 if CLS else 2
        self.stats_dic = self.initialize_statistics_dict(data.outcomes.keys(), censoring_times)

    def initialize_statistics_dict(self, outcome_names:List[str], censoring_times:List[float]):
        stats_dic = {'ages': {name: [] for name in outcome_names},
                     'trajectory': {name: {time: [] for time in censoring_times} for name in outcome_names},
                     'sequence_length': {name: {time: [] for time in censoring_times} for name in outcome_names},
                     'visits':{name: {time: [] for time in censoring_times} for name in outcome_names},}
        return stats_dic

    def calculate_statistics_for_censoring(self, patient:Dict, outcome:str, outcome_name:str, censoring_time:float):
        """censoring_time given in hours since origin_point"""
        censored_patient = Utilities.censor(patient, outcome - censoring_time * 30.4 * 24)
        if len(censored_patient['concept']) < self.MIN_LEN:
            return

        if censoring_time == 0:
            age = censored_patient['age'][-1]
            self.stats_dic['ages'][outcome_name].append(age)

        seq_len = len(censored_patient['concept']) - (2 if self.CLS else 1)
        start_traj = censored_patient['abspos'][2] if self.CLS else censored_patient['abspos'][1]
        traj_len = round((censored_patient['abspos'][-1] - start_traj) / 24 / 30.4)
        num_visits = max(censored_patient['segment'])
        self.stats_dic['visits'][outcome_name][censoring_time].append(num_visits)
        self.stats_dic['sequence_length'][outcome_name][censoring_time].append(seq_len)
        self.stats_dic['trajectory'][outcome_name][censoring_time].append(traj_len)

    def process_patients(self):
        for patient, outcomes in Utilities.iter_patients(self.data):
            for outcome_name, outcome in outcomes.items():
                if pd.isna(outcome):
                    continue
                for time in self.censoring_times:
                    self.calculate_statistics_for_censoring(patient, outcome, outcome_name, time)

    def convert_to_numpy(self):
        for stat_type in self.stats_dic:
            for outcome in self.stats_dic[stat_type]:
                if isinstance(self.stats_dic[stat_type][outcome], dict):
                    for time in self.stats_dic[stat_type][outcome]:
                        self.stats_dic[stat_type][outcome][time] = np.array(self.stats_dic[stat_type][outcome][time])
                else:
                    self.stats_dic[stat_type][outcome] = np.array(self.stats_dic[stat_type][outcome])
    def compute_statistics(self, values):
        if isinstance(values, float) or isinstance(values, int) or isinstance(values, np.int64):
            return [values]
        if len(values) == 0:
            return [None, None, None]
        return [np.percentile(values, 25), np.median(values), np.percentile(values, 75)]
    def compute_cumulative_stats(self, min_codes_list:list=[5, 10]):
        """Returns 25th percentile, median and 75th percentile. and patient counts for different min_codes."""
        for min_codes in min_codes_list:
            patient_counts_key = f'patient_counts_min{min_codes}'
            if patient_counts_key not in self.stats_dic:
                self.stats_dic[patient_counts_key] = {outcome: {time: 0 for time in self.censoring_times} for outcome in self.stats_dic['sequence_length']}

        # First compute the patient counts based on unaccumulated sequence lengths
        for min_codes in min_codes_list:
            patient_counts_key = f'patient_counts_min{min_codes}'
            for outcome, times in self.stats_dic['sequence_length'].items():
                for time, seq_lengths in times.items():
                    self.stats_dic[patient_counts_key][outcome][time] = sum(seq_len >= min_codes for seq_len in seq_lengths)
        
        # Then compute the percentiles and medians
        for stat_type in self.stats_dic:
            for outcome in self.stats_dic[stat_type]:
                if isinstance(self.stats_dic[stat_type][outcome], dict):
                    for time in self.stats_dic[stat_type][outcome]:
                        if stat_type.startswith('patient_counts_min'):
                            continue
                        self.stats_dic[stat_type][outcome][time] = self.compute_statistics(self.stats_dic[stat_type][outcome][time])
                else:
                    self.stats_dic[stat_type][outcome] = self.compute_statistics(self.stats_dic[stat_type][outcome])



