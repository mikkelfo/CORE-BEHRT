import unittest
from unittest.mock import patch, MagicMock, call
from tests.helpers import ConfigMock
from data.batch import Batches, BatchTokenize, Split
from data.tokenizer import EHRTokenizer

class TestBatches(unittest.TestCase):
    @patch('data.batch.load_exclude_pids', return_value=['100'])
    @patch('data.batch.load_assigned_pids', return_value={'pretrain': [str(i) for i in range(10)], 'finetune': [str(i) for i in range(90, 100)]})
    def setUp(self, *args):
        self.cfg = ConfigMock()
        self.cfg.split_ratios = {'pretrain': 0.7, 'finetune': 0.2, 'test': 0.1}
        self.cfg.n_splits = 2

        self.pids = [[str(i) for i in range(50)], [str(i) for i in range(50, 101)]]
        self.assigned_pids = {'pretrain': [str(i) for i in range(10)], 'finetune': [str(i) for i in range(90, 100)]}

        self.batches = Batches(self.cfg, self.pids)

    def test_split_batches(self):
        splits = self.batches.split_batches()
        self.assertEqual(len(splits['pretrain'].pids), 70)
        self.assertEqual(len(splits['finetune'].pids), 20)
        self.assertEqual(len(splits['test'].pids), 10)

    @patch('os.listdir', return_value=['pids_pretrain.pt', 'pids_finetune.pt', 'pids_test.pt'])
    @patch('torch.load', side_effect=[["1", "2", "3"], ["4", "5"], ["6"]])
    def test_predefined_split_batches(self, *args):
        self.cfg.predefined_splits_dir = 'test_dir'
        batches = Batches(self.cfg, self.pids)
        splits = batches.split_batches()
        self.assertEqual(len(splits['pretrain'].pids), 3)
        self.assertEqual(len(splits['finetune'].pids), 2)
        self.assertEqual(len(splits['test'].pids), 1)

    def test_reproduceability_split_batches(self):
        batches1 = Batches(self.cfg, self.pids)
        splits1 = batches1.split_batches()
        
        batches2 = Batches(self.cfg, self.pids)
        splits2 = batches2.split_batches()

        self.assertEqual(splits1, splits2)


class TestBatchTokenize(unittest.TestCase):
    @patch('os.makedirs', return_value=None)
    def setUp(self, mock_makedirs):
        tokenizer_config = ConfigMock()
        vocabulary = {'[UNK]': 0, 'BG_GENDER_MALE': 5, 'BG_GENDER_FEMALE': 6, 'Diagnosis1': 7, 'Diagnosis2': 8, 'Diagnosis3': 9, 'Medication1': 10}
        self.tokenizer = EHRTokenizer(tokenizer_config, vocabulary=vocabulary)

        self.cfg = MagicMock()
        self.cfg.output_dir = '/path/to/output'
        self.pids = [['1', '2', '3'], ['4', '5', '6']]
        self.batch_tokenize = BatchTokenize(self.pids, self.tokenizer, self.cfg)

    @patch('data.batch.BatchTokenize.load_and_filter_batch', return_value=({
                'concept': [['BG_GENDER_MALE', 'Diagnosis1', 'Medication1', 'Diagnosis2'], ['BG_GENDER_FEMALE', 'Diagnosis3', 'Medication1']],
                'segment': [[0, 1, 1, 2], [0, 1, 2]]},
            ["1", "2"]))
    @patch('torch.save', return_value=None)
    def test_batch_tokenize(self, *args):
        split = Split(mode='pretrain', pids=['1', '2'])
        result, result_pids = self.batch_tokenize.batch_tokenize(split)
        expected_result, expected_pids = ({
            'concept': [[5, 7, 10, 8], [6, 9, 10]],
            'segment': [[0, 1, 1, 2], [0, 1, 2]],
            'attention_mask': [[1]*4, [1]*3],
        }, ['1', '2'])
        self.assertEqual(result, expected_result)
        self.assertEqual(result_pids, expected_pids)

    @patch('data.batch.BatchTokenize.load_and_filter_batch', side_effect=[
        ({'concept': [['BG_GENDER_MALE', 'Diagnosis1', 'Medication1']], 'segment': [[0, 1, 2]]}, ["1"]),
        ({'concept': [['BG_GENDER_MALE', 'Diagnosis1']], 'segment': [[0, 1]]}, ["2"]),
        ({'concept': [['UNKNOWN']], 'segment': [[0]]}, ["3"]),
    ])
    @patch('torch.save', return_value=None)
    @patch('data.batch.BatchTokenize.save_tokenized_data', return_value=None)
    def test_tokenize(self, tokenize, save, save_tokenized_data):
        splits = {'pretrain': Split(mode='pretrain', pids=['1']), 'finetune': Split(mode='finetune', pids=["2"]), 'test': Split(mode='test', pids=["3"])}
        self.batch_tokenize.tokenize(splits)

        self.assertEqual(len(save_tokenized_data.call_args_list), 3)
        expected_results = ({
            'concept': [[5, 7, 10]],
            'segment': [[0, 1, 2]],
            'attention_mask': [[1]*3],
        }, ['1'], 'pretrain')
        self.assertEqual(self.batch_tokenize.save_tokenized_data.call_args_list[0], call(*expected_results, save_dir=None))

        expected_results2 = ({
            'concept': [[5, 7]],
            'segment': [[0, 1]],
            'attention_mask': [[1]*2],
        }, ['2'], 'finetune')
        self.assertEqual(self.batch_tokenize.save_tokenized_data.call_args_list[1], call(*expected_results2, save_dir=None))

        expected_results3 = ({
            'concept': [[0]],
            'segment': [[0]],
            'attention_mask': [[1]],
        }, ['3'], 'test')
        self.assertEqual(self.batch_tokenize.save_tokenized_data.call_args_list[2], call(*expected_results3, save_dir=None))
   
if __name__ == '__main__':
    unittest.main()