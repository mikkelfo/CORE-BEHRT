import unittest
from ehr2vec.common.config import Config


class TestConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config({'a':1, 'b':2, 'c':3})
        self.update_config = Config({'a':4, 'd':6})
    def test_update(self):
        self.cfg.update(self.update_config)
        self.assertEqual(self.cfg.a, 1)
        self.assertEqual(self.cfg.b, 2)
        self.assertEqual(self.cfg.c, 3)
        self.assertEqual(self.cfg.d, 6)

if __name__ == '__main__':
    unittest.main()
