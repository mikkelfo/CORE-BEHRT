import random
import unittest
from unittest.mock import Mock, patch
from ehr2vec.tests.helpers import ConfigMock
from ehr2vec.data.filter import CodeTypeFilter, PatientFilter


class TestCodeTypeFilter(unittest.TestCase):
    def setUp(self):
        self.cfg = ConfigMock()
        self.SPECIAL_CODES = ['[', 'BG_']
        self.cfg.data.code_types = ['D', 'M']
        self.filter = CodeTypeFilter(self.cfg)

    def test_combine_to_tuple(self):
        result = self.filter._combine_to_tuple(self.SPECIAL_CODES, self.cfg.data.code_types)
        self.assertEqual(result, ('[', 'BG_', 'D', 'M'))

    def test_filter_patient(self):
        data = Mock()
        # data.vocabulary = {'BG_GENDER_MALE': 10, 'Diagnosis1': 11, 'Medication1': 12, 'Labtest1': 13}
        data.features = {'concept': [[10, 11, 12, 13], [11, 13, 13]]}
        keep_tokens = set([10, 11, 15])

        patient1 = {'concept': data.features['concept'][0]}
        patient2 = {'concept': data.features['concept'][1]}

        self.filter._filter_patient(data, patient1, keep_tokens, 0)
        self.assertEqual(data.features['concept'][0], [10, 11])

        self.filter._filter_patient(data, patient2, keep_tokens, 1)
        self.assertEqual(data.features['concept'][1], [11])

    def test_filter(self):
        data = Mock()
        data.vocabulary = {'BG_GENDER_MALE': 10, 'Diagnosis1': 11, 'Medication1': 12, 'Labtest1': 13}
        data.features = {'concept': [[10, 11, 12, 13], [11, 13, 13]]}
        filtered_data = self.filter.filter(data)
        self.assertEqual(filtered_data.features['concept'], [[10, 11, 12], [11]])


class TestPatientFilter(unittest.TestCase):
    def setUp(self):
        self.cfg = ConfigMock()
        self.cfg.data.min_len = 3
        self.cfg.data.min_age = 0
        self.cfg.data.max_age = 120
        self.cfg.data.gender = 'male'
        self.cfg.paths.pretrain_model_path = '/path/to/pretrain_model'
        self.cfg.outcome.n_hours = 24
        self.filter = PatientFilter(self.cfg)

        self.data = Mock()
        self.data.features = {'concept': [[0, 2, 3], [1, 5, 6], [7, 8]], 
                              'age': [[10, 20, 30], [40, 50, 130], [120, 130, 140]],
                              'abspos':[[-1, 49.5, 51], [-1, 1, 2], [41.5, 51.0, 52]]}
        self.data.outcomes = [None, 100, 200]
        self.data.censor_outcomes = [50, None, 50]
        self.data.pids = ['pid1', 'pid2', 'pid3']
        self.data.vocabulary = {f"code{i}": i for i in range(2, 9)}
        self.data.vocabulary["BG_GENDER_Female"] = 0
        self.data.vocabulary["BG_GENDER_Male"] = 1


    @patch('torch.load')
    def test_exclude_pretrain_patients(self, mock_load):
        mock_load.return_value = ['pid1', 'pid2']
        result = self.filter.exclude_pretrain_patients(self.data)
        self.assertEqual(result.pids, ['pid3'])

    def test_filter_outcome_before_censor(self):
        result = self.filter.filter_outcome_before_censor(self.data)
        self.assertEqual(result.pids, ['pid1', 'pid3'])

    def test_select_censored(self):
        result = self.filter.select_censored(self.data)
        self.assertEqual(result.pids, ['pid1', 'pid3'])

    def test_exclude_short_sequences(self):
        self.data.vocabulary = {f"code{i}": i for i in range(1, 9)}
        result = self.filter.exclude_short_sequences(self.data)
        self.assertEqual(result.pids, ['pid1', 'pid2'])

    def test_select_by_age(self):
         result = self.filter.select_by_age(self.data)
         self.assertEqual(result.pids, ['pid1'])
        
    def test_calculate_ages_at_censor_date(self):
        ages_at_censor_date = self.filter.utils.calculate_ages_at_censor_date(self.data)
        ages_at_censor_date = [int(age) for age in ages_at_censor_date]
        self.assertEqual(ages_at_censor_date, [20, 130, 120])

    def test_select_by_gender(self):
        result = self.filter.select_by_gender(self.data)
        self.assertEqual(result.pids, ['pid2'])

    def test_select_random_subset(self):
        num_patients = 2
        result = self.filter.select_random_subset(self.data, num_patients, seed=42)
        result2 = self.filter.select_random_subset(self.data, num_patients, seed=42)

        random.seed(42)
        indices = list(range(len(self.data.pids)))
        random.shuffle(indices)
        
        self.assertCountEqual(result.pids, [self.data.pids[i] for i in indices[:num_patients]])
        self.assertEqual(result.pids, result2.pids)

if __name__ == '__main__':
    unittest.main()
