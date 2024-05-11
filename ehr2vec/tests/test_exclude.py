import unittest
from ehr2vec.data_fixes.exclude import Excluder

class TestExcluder(unittest.TestCase):

    def setUp(self):
        self.excluder = Excluder(min_len=3, vocabulary={'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, 'A': 3, 'B': 4, 'C': 5})

    def test_exclude_short_sequences(self):
        features = {
            'concept': [[3, 4, 5], [3, 4], [3, 4, 5, 5]]
        }
        outcomes = [1, 0, 1]
        expected_result = (
            {'concept': [[3, 4, 5], [3, 4, 5, 5]]}, # features
            [1, 1], # outcomes
            [0, 2]  # kept_indices
        )
        result = self.excluder.exclude_short_sequences(features, outcomes)
        self.assertEqual(result, expected_result)

    def test__exclude(self):
        features = {
            'concept': [[3, 4, 5], [3, 4], [3, 4, 5, 5]]
        }
        expected_result = [0, 2]
        result = self.excluder._exclude(features)
        self.assertEqual(result, expected_result)

    def test__is_tokenized(self):
        concepts_list = [[3, 4, 5], [3, 4], [3, 4, 5, 5]]
        concepts_list2 = [['A', 'B', 'C'], ['A', 'B'], ['A', 'B', 'C', 'C']]
        
        result = self.excluder._is_tokenized(concepts_list)
        self.assertTrue(result)

        result = self.excluder._is_tokenized(concepts_list2)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()