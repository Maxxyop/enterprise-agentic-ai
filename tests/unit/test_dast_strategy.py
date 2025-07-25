import unittest
from orchestrator.strategies.dast_strategy import DASTStrategy

class TestDASTStrategy(unittest.TestCase):
    def test_run(self):
        strategy = DASTStrategy()
        strategy.add_recon_task('recon1')
        strategy.add_exploitation_task('exploit1')
        # Should not raise exception
        try:
            strategy.run()
        except Exception as e:
            self.fail(f"DASTStrategy run raised exception: {e}")
