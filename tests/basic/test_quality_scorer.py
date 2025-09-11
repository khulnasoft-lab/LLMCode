import unittest
from llmcode.quality_scorer import QualityScorer

class TestQualityScorer(unittest.TestCase):

    def test_calculate_score(self):
        """
        Tests the calculate_score method.
        """
        code = "def hello():\n    print('Hello, world!')"
        scorer = QualityScorer(code)
        self.assertEqual(scorer.calculate_score(), 1.0)

if __name__ == '__main__':
    unittest.main()

