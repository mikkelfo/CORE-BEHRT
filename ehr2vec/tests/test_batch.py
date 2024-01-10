import unittest
from tests.helpers import ConfigMock
from data.batch import Batches

class TestBatches(unittest.TestCase):
    def setUp(self):
        self.cfg = ConfigMock()
        self.cfg.split_ratios = {'pretrain': 0.7, 'finetune': 0.2, 'test': 0.1}
        self.cfg.n_splits = 2

        self.pids = [[str(i) for i in range(50)], [str(i) for i in range(50, 101)]]
        self.exclude_pids = ['100']
        self.assigned_pids = {'pretrain': [str(i) for i in range(10)], 'finetune': [str(i) for i in range(90, 100)]}
        self.batches = Batches(self.cfg, self.pids, self.exclude_pids, self.assigned_pids)

    def test_split_batches(self):
        splits = self.batches.split_batches()
        self.assertEqual(len(splits['pretrain'].pids), 56+10)  # 56 from original pids and 10 assigned
        self.assertEqual(len(splits['finetune'].pids), 16+10)  # 16 from original pids and 10 assigned
        self.assertEqual(len(splits['test'].pids), 8)  # 8 from original pids

    def test_reproduceability_split_batches(self):
        batches1 = Batches(self.cfg, self.pids, self.exclude_pids, self.assigned_pids)
        splits1 = batches1.split_batches()
        
        batches2 = Batches(self.cfg, self.pids, self.exclude_pids, self.assigned_pids)
        splits2 = batches2.split_batches()

        self.assertEqual(splits1, splits2)
   
if __name__ == '__main__':
    unittest.main()