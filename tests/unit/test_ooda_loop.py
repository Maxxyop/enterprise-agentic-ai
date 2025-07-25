import unittest
from orchestrator.core.ooda_loop import OODA

class TestOODA(unittest.TestCase):
    def test_run_loop(self):
        ooda = OODA()
        ooda.run_loop("test data")
        self.assertIsNotNone(ooda.observation)
        self.assertIsNotNone(ooda.orientation)
        self.assertIsNotNone(ooda.decide)
        self.assertIsNotNone(ooda.action)

if __name__ == "__main__":
    unittest.main()
