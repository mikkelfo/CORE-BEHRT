import unittest

from ehr2vec.data_fixes.censor import Censorer, EQ_Censorer


class TestCensorer(unittest.TestCase):

    def setUp(self):
        self.censorer = Censorer(n_hours=1, min_len=3)

    def test_censor(self):
        features = {
            'concept': [['[CLS]', 'BG_GENDER_Male', '[SEP]', 'Diagnosis1', '[SEP]', 'Diagnosis2', 'Medication1', '[SEP]']],
            'abspos': [[0, 0, 0, 1, 1, 2, 3, 3]],
            'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1]]   
        }
        censor_outcomes = [1]
        expected_result = {
            'concept': [['[CLS]', 'BG_GENDER_Male', '[SEP]', 'Diagnosis1', '[SEP]', 'Diagnosis2']],
            'abspos': [[0, 0, 0, 1, 1, 2]],
            'attention_mask': [[1, 1, 1, 1, 1, 1]]
        }
        result = self.censorer.censor(features, censor_outcomes)
        self.assertEqual(result, expected_result)

    def test_if_tokenized(self):
        self.assertFalse(self.censorer._identify_if_tokenized(['[CLS]', 'BG_GENDER_Male', '[SEP]', 'Diagnosis1', '[SEP]', 'Diagnosis2']))
        self.assertTrue(self.censorer._identify_if_tokenized([0, 6, 1, 7, 1, 8]))

    def test_identify_background(self):
        self.censorer.vocabulary = {'[CLS]': 0, '[SEP]': 1, 'BG_GENDER_Male': 2, 'Diagnosis1': 3, 'Diagnosis2': 4}

        background_flags = self.censorer._identify_background(['[CLS]', 'BG_GENDER_Male', '[SEP]', 'Diagnosis1', '[SEP]', 'Diagnosis2'], tokenized_flag=False)
        self.assertEqual(background_flags, [True, True, True, False, False, False])

        background_flags_tokenized = self.censorer._identify_background([0, 2, 1, 3, 1, 4], tokenized_flag=True)
        self.assertEqual(background_flags_tokenized, [True, True, True, False, False, False])

    def test_generate_censor_flags(self):
        abspos = [0, 0, 0, 1, 1, 2, 3, 3]
        background_flags = [False, True, False, False, False, False, False, False]
        event_timestamp = 1
        censor_flags = self.censorer._generate_censor_flags(abspos, background_flags, event_timestamp)

        self.assertEqual(censor_flags, [True, True, True, True, True, True, False, False])

class TestEQCensorer(unittest.TestCase):

    def setUp(self):
        self.eq_censorer = EQ_Censorer(n_hours=1, min_len=3)

    def test_get_censor_outcomes_for_negatives_reproducibility(self):
        censor_outcomes = [1.0, None, 2.0, None, 3.0, None, 4.0, None, None, None, None, None, None, None, None, None, None, None, None, None]
        result = self.eq_censorer.get_censor_outcomes_for_negatives(censor_outcomes)
        result2 = self.eq_censorer.get_censor_outcomes_for_negatives(censor_outcomes)

        # Check that all patients are censored now
        self.assertNotIn(None, result)

        # Check reproducibility
        self.assertEqual(result, result2)


if __name__ == '__main__':
    unittest.main()