import unittest
from ai_agents.foundation_models.gemini_25_pro import GeminiAgent

class TestGeminiAgent(unittest.TestCase):
    def setUp(self):
        self.agent = GeminiAgent()

    def test_generate_report(self):
        report_data = {'type': 'executive_summary', 'content': 'Test'}
        result = self.agent.generate_report(report_data)
        self.assertIn('summary', result)
        self.assertTrue(result['summary'].startswith('Report generated'))

if __name__ == '__main__':
    unittest.main()
