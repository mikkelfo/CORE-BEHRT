import unittest
from unittest.mock import patch, MagicMock
from ehr2vec.data.dataset import MLMDataset   # replace with the actual module name
import torch

class TestMLMDataset(unittest.TestCase):
    @patch('torch.load')
    def test_init(self, mock_torch_load):
        # Set up the mock objects
        mock_torch_load.return_value = {
            'concept': [[1, 2, 1,3,], [4,1, 5, 6]],
            'segment': [[0, 0, 0,1,], [0,0, 1, 2]]
        }
        mock_data_dir = 'data_dir'
        mock_mode = 'mode'
        mock_vocabulary = {'[MASK]': 0, '[SEP]': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6}
        # Call the function and check the result
        dataset = MLMDataset(mock_data_dir, mock_mode, vocabulary=mock_vocabulary, masked_ratio=0.6)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.max_segments, 3)

    @patch('torch.load')
    def test_getitem(self, mock_torch_load):
        # Set up the mock objects
        mock_torch_load.return_value = {
            'concept': [[1, 2, 1,3,], [4,1, 5, 6]],
            'segment': [[0, 0, 0,1,], [0,0, 1, 2]]
        }
        mock_vocabulary = {'[MASK]': 0, '[SEP]': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6}
        mock_data_dir = 'data_dir'
        mock_mode = 'mode'
        print('original concepts', mock_torch_load.return_value['concept'])
        # Call the function and check the result
        dataset = MLMDataset(mock_data_dir, mock_mode, vocabulary=mock_vocabulary, masked_ratio=0.6)
        patient = dataset[0]
        print('masked patient',patient)
        # Check that none of the special tokens are masked
        special_token_mask = (torch.tensor(mock_torch_load.return_value['concept'][0]) < 2)
        print("Special token MASK", special_token_mask)
        self.assertTrue(torch.all(patient['concept'][special_token_mask] != 0))
        # Check that tokens which have a target not equal to -100 are either a mask token, the original token, or a random non-special token
        target_indices = (patient['target'] != -100).nonzero(as_tuple=True)[0]
        self.assertTrue(torch.all((patient['concept'][target_indices] == 0) | (patient['concept'][target_indices] == patient['target'][target_indices]) | ((patient['concept'][target_indices] >= 2) & (patient['concept'][target_indices] < len(mock_vocabulary)))))
