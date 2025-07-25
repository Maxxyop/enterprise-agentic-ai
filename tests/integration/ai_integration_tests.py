import unittest
from ai_agents.foundation_models.deepseek_r1_agent import DeepSeekR1Agent
from ai_agents.foundation_models.gemini_25_pro import GeminiAgent
from ai_agents.foundation_models.llm_interface import LLMInterface

class TestAIIntegration(unittest.TestCase):
    def setUp(self):
        self.deepseek_agent = DeepSeekR1Agent()
        self.gemini_agent = GeminiAgent()
        self.llm_interface = LLMInterface()

    def test_deepseek_agent_inference(self):
        """Test DeepSeek agent inference with valid input."""
        input_data = "Sample input for DeepSeek R1"
        result = self.deepseek_agent.infer(input_data)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)

    def test_gemini_agent_report_generation(self):
        """Test Gemini agent report generation with valid input."""
        report_data = {"type": "executive_summary", "content": "Test report"}
        result = self.gemini_agent.generate_report(report_data)
        self.assertIsNotNone(result)
        self.assertIn("summary", result)

    def test_llm_interface_prompt_handling(self):
        """Test LLM interface prompt handling."""
        prompt = "What is the capital of France?"
        response = self.llm_interface.get_response(prompt)
        self.assertEqual(response, "Paris")

    def test_invalid_task_type(self):
        """Test agent coordinator with invalid task type."""
        with self.assertRaises(ValueError):
            self.deepseek_agent.process_task({'type': 'invalid'})

if __name__ == '__main__':
    unittest.main()