import unittest
from unittest.mock import Mock
from data_fixes.handle import Handler

class TestHandler(unittest.TestCase):
    def setUp(self):
        self.handler = Handler(min_age=-1, max_age=120)

    def test_handle_incorrect_ages(self):
        patient = {'age': [-2, -1, 0, 5, 120, 150]}
        expected_result = {'age': [-1, 0, 5, 120]}
        result = self.handler.handle_incorrect_ages(patient)
        self.assertEqual(result, expected_result)

    def test_handle_nans(self):
        patient = {'age': [None, 0, 5, 120, 150]}
        expected_result = {'age': [0, 5, 120, 150]}
        result = self.handler.handle_nans(patient)
        self.assertEqual(result, expected_result)

    def test_normalize_segments(self):
        segments = [0, 1, 3, 6, 3, 4, 8, 9]
        expected_result = [0, 1, 2, 4, 2, 3, 5, 6]
        result = self.handler.normalize_segments(segments)
        self.assertEqual(result, expected_result)

    def test_handle(self):
        features = {
            'concept': [[0, 1, 2, 3, 4]],
            'age': [[-2, 0, 5, 120, 150]],
            'segment': [[0, 1, 2, 3, 4]]
        }
        expected_result = {
            'concept': [[1, 2, 3]],
            'age': [[0, 5, 120]],
            'segment': [[0, 1, 2]]
        }
        result = self.handler.handle(features)
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()