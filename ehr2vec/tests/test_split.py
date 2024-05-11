import unittest
from itertools import combinations
from ehr2vec.data.split import get_n_splits_cv, get_n_splits_cv_k_over_n

class TestDataObject:
    def __init__(self, pids):
        self.pids = pids

class TestSplit(unittest.TestCase):
    def setUp(self):
        self.data = TestDataObject(list(range(10)))

    def test_get_n_splits_cv(self):
        n_splits = 5
        splits = list(get_n_splits_cv(self.data, n_splits))
        self.assertEqual(len(splits), n_splits)
        for train_indices, val_indices in splits:
            self.assertEqual(len(train_indices), len(self.data.pids) - len(self.data.pids) // n_splits)
            self.assertEqual(len(val_indices), len(self.data.pids) // n_splits)

    def test_reproduceability(self):
        n_splits = 5
        splits1 = list(get_n_splits_cv(self.data, n_splits))
        splits2 = list(get_n_splits_cv(self.data, n_splits))
        self.assertEqual(splits1, splits2)

    def test_get_n_splits_cv_k_over_n(self):
        k = 5
        n = 3
        splits = list(get_n_splits_cv_k_over_n(self.data, k, n))
        self.assertEqual(len(splits), len(list(combinations(range(k), n))))
        for train_indices, validation_subsets, validation_keys in splits:
            self.assertEqual(len(train_indices), len(self.data.pids) // k * n)
            self.assertEqual(len(validation_subsets), k - n)
            for subset in validation_subsets:
                self.assertEqual(len(subset), len(self.data.pids) // k)

if __name__ == '__main__':
    unittest.main()