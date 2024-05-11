import torch
import unittest
from unittest.mock import patch
from ehr2vec.data.mask import ConceptMasker

class TestConceptMasker(unittest.TestCase):
    def setUp(self):
        self.vocabulary = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "[MASK]": 101, "Diagnosis1": 5, "Medication1": 6, "Diagnosis2": 7, "Medication2": 8}
        self.select_ratio = 0.15
        self.masking_ratio = 0.8
        self.replace_ratio = 0.1
        self.masker = ConceptMasker(self.vocabulary, self.select_ratio, self.masking_ratio, self.replace_ratio)

    def test_init(self):
        self.assertEqual(self.masker.vocabulary, self.vocabulary)
        self.assertEqual(self.masker.select_ratio, self.select_ratio)
        self.assertEqual(self.masker.masking_ratio, self.masking_ratio)
        self.assertEqual(self.masker.replace_ratio, self.replace_ratio)
        self.assertEqual(self.masker.n_special_tokens, 5)

    @patch('torch.rand')
    @patch('torch.randint')
    def test_mask_patient_concepts(self, mock_randint, mock_rand):
        patient = {"concept": torch.tensor([1, 5, 2, 7, 6, 8])}
        mock_rand.return_value = torch.tensor([0.1, 0.2, 0.13, 0.15])
        mock_randint.return_value = torch.tensor([102, 102])

        concepts, target = self.masker.mask_patient_concepts(patient)
        mock_rand.assert_called_once()

        self.assertTrue(torch.equal(concepts, torch.tensor([1, 101, 2, 7, 102, 8])))
        self.assertTrue(torch.equal(target, torch.tensor([-100, 5, -100, -100, 6, -100])))

if __name__ == '__main__':
    unittest.main()