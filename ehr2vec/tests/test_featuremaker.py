import unittest

import pandas as pd

from ehr2vec.data.featuremaker import FeatureMaker
from ehr2vec.tests.helpers import ConfigMock


class TestFeatureMaker(unittest.TestCase):
    def setUp(self):
        self.cfg = ConfigMock()
        self.cfg.age = {'round': 2}
        self.cfg.abspos = {'year': 2020, 'month': 1, 'day': 26}
        self.cfg.segment = True
        self.cfg.background = ['GENDER']
        self.feature_maker = FeatureMaker(self.cfg)

        self.concepts = pd.DataFrame({
            'PID': ['1', '2', '3', '1'],
            'CONCEPT': ['DA1', 'DA2', 'MA1', 'DA2'],
            'TIMESTAMP': pd.to_datetime(['2020-01-02', '2021-03-20', '2022-05-08', '2023-01-02']),
            'ADMISSION_ID': ['A', 'B', 'C', 'D']
        })
        self.patients_info = pd.DataFrame({
            'PID': ['1', '2', '3'],
            'BIRTHDATE': pd.to_datetime(['2000-01-02', '2000-03-20', '2000-05-08']),
            'GENDER': ['Male', 'Female', 'Male']
        })

    def test_call(self):
        features, pids = self.feature_maker(self.concepts, self.patients_info)
        self.assertIsInstance(features, dict)
        self.assertIsInstance(pids, list)
        for key in self.cfg.keys():
            if key == 'background':
                continue
            self.assertIn(key, features.keys())

        self.assertCountEqual(pids, self.concepts['PID'].unique().tolist())

    def test_create_pipeline(self):
        pipeline = self.feature_maker.create_pipeline()
        self.assertIsInstance(pipeline, list)
        self.assertEqual(len(pipeline), len(self.cfg))

        for key, pos in self.feature_maker.order.items():
            if key in self.cfg.keys():
                self.assertEqual(pipeline[pos].id, key)

if __name__ == '__main__':
    unittest.main()