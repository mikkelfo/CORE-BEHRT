import unittest
from data_fixes.truncate import Truncator

class TestTruncator(unittest.TestCase):

    def setUp(self):
        self.truncator = Truncator(max_len=6, vocabulary={'[CLS]': 1, '[SEP]': 2, 'BG_GENDER': 3, 'A': 4, 'B': 5, 'C': 6})

    def test_truncate(self):
        # Test normal input
        features = {
            'concept': [[1, 3, 2, 4, 2, 5, 6, 2], [1, 2, 3]]
        }
        expected_result = {
            'concept': [[1, 3, 2, 5, 6, 2], [1, 2, 3]]
        }
        result = self.truncator.truncate(features)
        self.assertEqual(result, expected_result)

        # Test where truncation_length ends on a SEP token [1, 3, 2] + [2, 6, 2]
        features = {
            'concept': [[1, 3, 2, 4, 2, 6, 2]]
        }
        expected_result = {
            'concept': [[1, 3, 2, 6, 2]]    # [SEP] is removed, length is 5
        }
        result = self.truncator.truncate(features)
        self.assertEqual(result, expected_result)

        # Test without [CLS]
        features = {
            'concept': [[3, 2, 4, 2, 6, 2]]
        }
        expected_result = {
            'concept': [[3, 2, 4, 2, 6, 2]]
        }
        result = self.truncator.truncate(features)
        self.assertEqual(result, expected_result)

    def test__truncate_patient(self):
        patient = {
            'concept': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
        background_length = 4
        expected_result = {
            'concept': [1, 2, 3, 4, 9, 10]
        }
        result = self.truncator._truncate_patient(patient, background_length)
        self.assertEqual(result, expected_result)

    def test__get_background_length(self):
        self.truncator.vocabulary['BG_LOCATION'] = 7
        self.truncator.vocabulary["BG_FOO"] = 8
        features = {
            'concept': [[1, 3, 7, 8, 2, 4, 5, 6]]
        }
        expected_result = 5
        result = self.truncator._get_background_length(features)
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()