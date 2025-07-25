import unittest
from execution_engine.core.executor import Executor

class TestExecutor(unittest.TestCase):
    def setUp(self):
        self.executor = Executor()

    def test_validate_command(self):
        self.assertTrue(self.executor.validate_command('nmap -sV 127.0.0.1'))
        self.assertTrue(self.executor.validate_command('zap scan'))
        self.assertFalse(self.executor.validate_command('invalid'))

    def test_run_nmap(self):
        result = self.executor.run('nmap', '127.0.0.1')
        self.assertIn('output', result)

    def test_run_zap(self):
        result = self.executor.run('zap', 'http://example.com')
        self.assertIn('output', result)

    def test_run_invalid(self):
        with self.assertRaises(ValueError):
            self.executor.run('invalid', 'target')

if __name__ == '__main__':
    unittest.main()
