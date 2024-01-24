import unittest
import pandas as pd
from data_fixes.infer import Inferrer

class TestInferrer(unittest.TestCase):

    def setUp(self):
        self.inferrer = Inferrer(functions=['SEGMENT', 'TIMESTAMP'])

    def test_infer_admission_id(self):
        df = pd.DataFrame({
            'PID': [1, 2, 3, 4, 5],
            'SEGMENT': ['A', None, 'A', None, 'C']
        })
        expected_result = pd.Series(['A', 'A', 'A', 'unq_0', 'C'], name='SEGMENT')
        result = self.inferrer.infer_admission_id(df)
        pd.testing.assert_series_equal(result, expected_result)

    def test_infer_timestamps_from_admission_id(self):
        df = pd.DataFrame({
            'SEGMENT': ['A', 'A', 'B', 'B', 'C', 'D'],
            'TIMESTAMP': [1, None, 3, None, None, 5]
        })
        expected_result = pd.Series([1.0, 1.0, 3.0, 3.0, None, 5.0], name='TIMESTAMP')
        result = self.inferrer.infer_timestamps_from_admission_id(df, strategy="last")
        pd.testing.assert_series_equal(result, expected_result)

    def test_infer(self):
        df = pd.DataFrame({
            'PID': [1, 2, 3, 4, 5],
            'SEGMENT': ['A', None, 'A', None, 'C'],
            'TIMESTAMP': [1, None, 3, None, 5]
        })
        expected_result = pd.DataFrame({
            'PID': [1, 2, 3, 4, 5],
            'SEGMENT': ['A', 'A', 'A', 'unq_0', 'C'],
            'TIMESTAMP': [1, 1, 3, None, 5]
        })
        result = self.inferrer(df)
        pd.testing.assert_frame_equal(result, expected_result)

if __name__ == '__main__':
    unittest.main()