import unittest
from execution_engine.core.executor import Executor

class TestExecutionIntegration(unittest.TestCase):

    def setUp(self):
        self.executor = Executor()

    def test_nmap_execution(self):
        result = self.executor.execute_nmap(target='127.0.0.1')
        self.assertIn('Nmap done:', result)

    def test_zap_execution(self):
        result = self.executor.execute_zap(target='http://example.com')
        self.assertIn('ZAP scan completed', result)

    def test_sqlmap_execution(self):
        result = self.executor.execute_sqlmap(target='http://example.com/vuln')
        self.assertIn('sqlmap identified the following injection point', result)

    def test_burp_integration(self):
        result = self.executor.integrate_burp(target='http://example.com')
        self.assertIn('Burp Suite integration successful', result)

if __name__ == '__main__':
    unittest.main()