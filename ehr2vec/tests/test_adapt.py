import unittest
from data_fixes.adapt import BehrtAdapter

class TestBehrtAdapter(unittest.TestCase):

    def setUp(self):
        self.adapter = BehrtAdapter()

    def test_convert_to_int(self):
        ages = [-1, 0, 5, 120, 150]
        expected_result = [0, 0, 5, 119, 119]
        result = self.adapter.convert_to_int(ages)
        self.assertEqual(result, expected_result)

    def test_convert_segment(self):
        segments = ['A', 'A', 'B', 'B', 'C', 'C']
        expected_result = [0, 0, 1, 1, 0, 0]
        result = self.adapter.convert_segment(segments)
        self.assertEqual(result, expected_result)

    def test_adapt_features(self):
        features = {
            'abspos': 'abspos_value',
            'age': [[-1, 0, 5, 120, 150]],
            'segment': [['A', 'A', 'B', 'B', 'C', 'C']]
        }
        expected_result = {
            'age': [[0, 0, 5, 119, 119]],
            'position_ids': [['A', 'A', 'B', 'B', 'C', 'C']],
            'segment': [[0, 0, 1, 1, 0, 0]]
        }
        result = self.adapter.adapt_features(features)
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()