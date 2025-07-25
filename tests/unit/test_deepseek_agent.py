import unittest
from ai_agents.foundation_models.deepseek_r1_agent import DeepSeekR1Agent

class TestDeepSeekAgent(unittest.TestCase):
    def setUp(self):
        self.agent = DeepSeekR1Agent()

    def test_infer(self):
        result = self.agent.infer('test input')
        self.assertTrue(result['success'])
        self.assertIn('output', result)

    def test_process_task_reasoning(self):
        result = self.agent.process_task({'type': 'reasoning'})
        self.assertTrue(result['success'])
        self.assertEqual(result['result'], 'Reasoning complete')

    def test_process_task_invalid(self):
        with self.assertRaises(ValueError):
            self.agent.process_task({'type': 'invalid'})

if __name__ == '__main__':
    unittest.main()
