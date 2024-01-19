import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
from data.utils import Utilities

class TestUtilities(unittest.TestCase):
    def test_process_datasets(self):
        datasets = Mock()
        data = Mock(pids=[1,2,3])
        datasets.items.return_value = [("pretrain", data), ]
        datasets.__setitem__ = Mock()
        func = Mock()
        func.__name__ = "func"
        Utilities.process_data(datasets, func)
        func.assert_called_once_with(data)

    def test_log_patient_nums(self):
        pass

    def test_select_and_order_outcomes_for_patients(self):
        all_outcomes = {'PID': [3, 2, 1], "COVID": [1, 2, 0]}
        pids = [1, 2, 3]
        outcome = "COVID"
        outcomes = Utilities.select_and_order_outcomes_for_patients(all_outcomes, pids, outcome)

        self.assertEqual(outcomes, [0, 2, 1])

    def test_get_abspos_from_origin_point(self):
        timestamps = [datetime(1, 1, i) for i in range(1,11)]
        origin_point = {'year': 1, 'month': 1, 'day': 1}
        abspos = Utilities.get_abspos_from_origin_point(timestamps, origin_point)

        self.assertEqual(abspos, [24*i for i in range(10)])

    def test_get_relative_timestamps_in_hours(self):
        timestamps = pd.Series(pd.to_datetime([f'2020-01-0{i}' for i in range(1,10)]))
        origin_point = datetime(**{'year': 2020, 'month': 1, 'day': 1})
        rel_timestamps = Utilities.get_relative_timestamps_in_hours(timestamps, origin_point)

        self.assertEqual(rel_timestamps.tolist(), [24*i for i in range(9)])

    def test_check_and_adjust_max_segment(self):
        data = Mock(features={'segment': [[1,2,3], [4,5,6]]})
        model_config = Mock(type_vocab_size=5)
        Utilities.check_and_adjust_max_segment(data, model_config)

        self.assertEqual(model_config.type_vocab_size, 7)

    def test_get_token_to_index_map(self):
        vocab = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3, '[UNK]': 4, 'A': 5, 'B': 6}
        token2index, new_vocab = Utilities.get_token_to_index_map(vocab)
        self.assertEqual(token2index, {5: 0, 6: 1})
        self.assertEqual(new_vocab, {'A': 0, 'B': 1})

    def test_get_gender_token(self):
        BG_GENDER_KEYS = {
            'male': ['M', 'Mand',  'male', 'Male', 'man', 'MAN', '1'],
            'female': ['W', 'Kvinde', 'F', 'female', 'Female', 'woman', 'WOMAN', '0']
        }
        vocabulary = {'BG_GENDER_Male': 0, 'BG_GENDER_Female': 1}
        for gender, values in BG_GENDER_KEYS.items():
            for value in values:
                result = Utilities.get_gender_token(vocabulary, value)
                if gender == "male":
                    self.assertEqual(result, 0)
                elif gender == "female":
                    self.assertEqual(result, 1)

    def test_get_background_indices(self):
        data = Mock(vocabulary={'[SEP]': -1, 'BG_Gender': 0, 'BG_Age': 1, 'BG_Country': 2, 'Foo': 3}, features={'concept': [[0, 1, 3], [0, 1, 3]]})
        data_none = Mock(vocabulary={'[SEP]': -1, 'Foo': 3}, features={'concept': [[3], [3]]})

        background_indices = Utilities.get_background_indices(data)
        self.assertEqual(background_indices, [0, 1])

        background_indices = Utilities.get_background_indices(data_none)
        self.assertEqual(background_indices, [])

    def test_code_starts_with(self):
        self.assertTrue(Utilities.code_starts_with('123', ('1', '2')))
        self.assertFalse(Utilities.code_starts_with('345', ('1', '2')))

    def test_log_pos_patients_num(self):
        pass

    @patch('os.listdir', return_value=[f"checkpoint_epoch{i}_end.pt" for i in range(10)])
    def test_get_last_checkpoint_epoch(self, mock_listdir):
        last_epoch = Utilities.get_last_checkpoint_epoch("dir")

        self.assertEqual(last_epoch, 9)

    def test_split_train_val(self):
        features = {'concept': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
        pids = [1, 2, 3]
        val_ratio=0.35

        train_features, train_pids, val_features, val_pids = Utilities.split_train_val(features, pids, val_ratio)

        self.assertEqual(train_features, {'concept': [[1, 2, 3], [4, 5, 6]]})
        self.assertEqual(val_features, {'concept': [[7, 8, 9]]})
        self.assertEqual(train_pids, [1, 2])
        self.assertEqual(val_pids, [3])

    def test_filter_and_order_outcomes(self):
        outcomes_dic = {
            'GROUP1': {
                'PID': [3, 2, 1],
                "COVID": [1, 2, 0],
                "DEATH": [0, 1, 0],
                "ICU": [2, 0, 1],
            },
        }
        pids = [1, 2, 3]
        outcomes = Utilities.filter_and_order_outcomes(outcomes_dic, pids)

        self.assertEqual(outcomes, {'COVID': [0, 2, 1], 'DEATH': [0, 1, 0], 'ICU': [1, 0, 2]})

    def test_iter_patients(self):
        pass

    def test_censor(self):
        patient0 = {'concept': [1, 2, 3], 'abspos': [1, 2, 3]}
        patient1 = {'concept': [4, 5, 6], 'abspos': [4, 5, 6]}

        result = Utilities.censor(patient0, 2)
        result1 = Utilities.censor(patient1, 10)

        self.assertEqual(result, {'concept': [1, 2], 'abspos': [1, 2]})
        self.assertEqual(result1, {'concept': [4, 5, 6], 'abspos': [4, 5, 6]})

    def test__generate_censor_flags(self):
        abspos = [1, 2, 3, 4, 5]
        event_timestamp = 3

        result = Utilities._generate_censor_flags(abspos, event_timestamp)

        self.assertEqual(result, [True, True, True, False, False])

if __name__ == '__main__':
    unittest.main()