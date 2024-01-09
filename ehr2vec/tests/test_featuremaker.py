import unittest
import pandas as pd
from data.featuremaker import FeatureMaker
from tests.helpers import FeaturesConfig


class TestFeatureMaker(unittest.TestCase):
    def setUp(self):
        self.config = FeaturesConfig()
        self.feature_maker = FeatureMaker(self.config)

        self.concepts = pd.DataFrame({
            'PID': ['1', '2', '3', '1'],
            'CONCEPT': ['DA1', 'DA2', 'MA1', 'DA2'],
            'TIMESTAMP': pd.to_datetime(['2020-01-02', '2021-03-20', '2022-05-08', '2023-01-02'])
        })
        self.patients_info = pd.DataFrame({
            'PID': ['1', '2', '3'],
            'BIRTHDATE': pd.to_datetime(['2000-01-02', '2000-03-20', '2000-05-08']),
            'GENDER': ['Male', 'Female', 'Male']
        })

    def test_reordering(self):
        for key, pos in self.order.items():
            if key in self.config.keys():
                self.assertEqual(self.pipeline[pos].id, key)

    def test_call(self):
        features, pids = self.feature_maker(self.concepts, self.patients_info)
        self.assertIsInstance(features, dict)
        self.assertIsInstance(pids, list)
        for key in self.config.keys():
            if key == 'background':
                continue
            self.assertIn(key, features.keys())

        self.assertCountEqual(pids, self.concepts['PID'].unique().tolist())

    def test_create_pipeline(self):
        pipeline = self.feature_maker.create_pipeline()
        self.assertIsInstance(pipeline, list)
        self.assertEqual(len(pipeline), len(self.config))

if __name__ == '__main__':
    unittest.main()