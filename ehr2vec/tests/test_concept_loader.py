import unittest
import pandas as pd
from unittest.mock import patch
from data.concept_loader import ConceptLoader

class TestConceptLoader(unittest.TestCase):
    @patch('data.concept_loader.ConceptLoader._verify_input', return_value=None)
    @patch('data.concept_loader.ConceptLoader._verify_paths', return_value=None)
    def setUp(self, *args):
        self.conceptloader = ConceptLoader()
        #self.conceptloaderlarge = ConceptLoaderLarge() # Only need to test ConceptLoader since they share the functions

    def test_detect_date_columns(self):
        df = pd.DataFrame({
            'testdate': ['2020-01-01', '2020-01-02'],
            'notadate': ["foo", "bar"],
            'testtime': ['20:20:30', '20:20:50'],
            'dateandtime': ['2023-07-04 23:00:11', '2021-04-22 12:59:22']
        })
        self.assertEqual(list(self.conceptloader._detect_date_columns(df)), ['testdate', 'testtime', 'dateandtime'])

    def test_invalid_creation(self):
        with self.assertRaises(AssertionError):
            ConceptLoader(concepts='diagnose')
        with self.assertRaises(AssertionError):
            ConceptLoader(concepts=['diagnose'], data_dir=1)
        with self.assertRaises(AssertionError):
            ConceptLoader(concepts=[])

if __name__ == '__main__':
    unittest.main()