import unittest
from unittest.mock import patch, MagicMock

from tests.helpers import ConfigMock
from data.tokenizer import EHRTokenizer

class TestEHRTokenizer(unittest.TestCase):
    def setUp(self):
        self.config = ConfigMock()
        self.config.truncation = 512
        self.config.padding = True
        self.config.cutoffs = {'D': 4, 'M': 5}
        self.config.sep_tokens = True
        self.config.cls_token = True
        self.tokenizer = EHRTokenizer(self.config)

    def test_init(self):
        self.assertEqual(self.tokenizer.config, self.config)
        self.assertEqual(self.tokenizer.vocabulary, {
            '[PAD]': 0,
            '[CLS]': 1, 
            '[SEP]': 2,
            '[UNK]': 3,
            '[MASK]': 4,
        })
        self.assertEqual(self.tokenizer.truncation, self.config['truncation'])
        self.assertEqual(self.tokenizer.padding, self.config['padding'])
        self.assertEqual(self.tokenizer.cutoffs, self.config['cutoffs'])

    def test_encode(self):
        concepts = ['concept1', 'concept2', 'concept3']
        encoded = self.tokenizer.encode(concepts)
        self.assertEqual(encoded, [5, 6, 7])
        self.assertEqual(self.tokenizer.vocabulary, {
            '[PAD]': 0,
            '[CLS]': 1, 
            '[SEP]': 2,
            '[UNK]': 3,
            '[MASK]': 4,
            'concept1': 5,
            'concept2': 6,
            'concept3': 7
        })

    def test_truncate(self):
        patient = {
            'concept': ['[CLS]', 'BG_1', '[SEP]', 'Diagnosis_1', 'Diagnosis_2', 'Diagnosis_3', 'Diagnosis_4', 'Diagnosis_5', 'Diagnosis_6'],
            'segment': [0, 0, 0, 1, 2, 3, 1, 4, 5]
        }
        truncated = self.tokenizer.truncate(patient, 7)
        self.assertEqual(truncated, {
            'concept': ['[CLS]', 'BG_1', '[SEP]', 'Diagnosis_3', 'Diagnosis_4', 'Diagnosis_5', 'Diagnosis_6'],
            'segment': [0, 0, 0, 2, 1, 3, 4]
        })

    def test_pad(self):
        patient = {
            'concept': [[10, 20, 30], [10, 20], [10]],
            'segment': [[1, 1, 2], [1, 2], [1]],
            'age': [[1, 2, 3], [1, 2], [1]]
        }
        padded = self.tokenizer.pad(patient, 3)
        self.assertEqual(padded, {
            'concept': [[10, 20, 30], [10, 20, 0], [10, 0, 0]],
            'segment': [[1, 1, 2], [1, 2, 0], [1, 0, 0]],
            'age': [[1, 2, 3], [1, 2, 0], [1, 0, 0]]
        })

    def test_insert_special_tokens(self):
        patient = {
            'concept': ['concept1', 'concept2', 'concept3'], 
            'segment': [1, 1, 2], 
            'age': [1, 2, 3]
        }
        with_special_tokens = self.tokenizer.insert_special_tokens(patient)
        self.assertEqual(with_special_tokens, {
            'concept': ['[CLS]', 'concept1', 'concept2', '[SEP]', 'concept3', '[SEP]'],
            'segment': [1, 1, 1, 1, 2, 2],
            'age': [1, 1, 2, 2, 3, 3]
        })

    def test_insert_sep_tokens(self):
        patient = {
            'concept': ['concept1', 'concept2', 'concept3'], 
            'segment': [1, 2, 3], 'age': [1, 2, 3]
        }
        with_sep_tokens = self.tokenizer.insert_sep_tokens(patient)
        self.assertEqual(with_sep_tokens, {
            'concept': ['concept1', '[SEP]', 'concept2', '[SEP]', 'concept3', '[SEP]'],
            'segment': [1, 1, 2, 2, 3, 3],
            'age': [1, 1, 2, 2, 3, 3]
        })

    def test_insert_cls_token(self):
        patient = {
            'concept': ['concept1', 'concept2', 'concept3'], 
            'segment': [1, 1, 2], 'age': [1, 2, 3]
        }
        with_cls_token = self.tokenizer.insert_cls_token(patient)
        self.assertEqual(with_cls_token, {
            'concept': ['[CLS]', 'concept1', 'concept2', 'concept3'],
            'segment': [1, 1, 1, 2],
            'age': [1, 1, 2, 3]
        })

    def test_limit_concepts_length(self):
        concepts = ['D12345', 'M12345']
        limited = self.tokenizer.limit_concepts_length(concepts)
        self.assertEqual(limited, ['D123', 'M1234'])

    @patch('torch.save', new_callable=MagicMock)
    def test_save_vocab(self, mock_save):
        self.tokenizer.save_vocab('vocab.pt')
        mock_save.assert_called_once_with(self.tokenizer.vocabulary, 'vocab.pt')

    def test_freeze_vocabulary(self):
        self.assertTrue(self.tokenizer.new_vocab)
        self.tokenizer.freeze_vocabulary()
        self.assertFalse(self.tokenizer.new_vocab)

if __name__ == '__main__':
    unittest.main()