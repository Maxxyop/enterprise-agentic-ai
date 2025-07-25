import unittest
from ai_agents.foundation_models.deepseek_r1_agent import DeepSeekR1Agent
from ai_agents.foundation_models.qwen_7b_agent import Qwen7BAgent
from ai_agents.foundation_models.llm_interface import LLMInterface

class TestAIIntegration(unittest.TestCase):

    def setUp(self):
        self.deepseek_agent = DeepSeekR1Agent()
        self.qwen_agent = Qwen7BAgent()
        self.llm_interface = LLMInterface()

    def test_deepseek_agent_inference(self):
        input_data = "Sample input for DeepSeek R1"
        result = self.deepseek_agent.infer(input_data)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)

    def test_qwen_agent_command_execution(self):
        command = "nmap -sP 192.168.1.0/24"
        result = self.qwen_agent.execute_command(command)
        self.assertIsNotNone(result)
        self.assertIn("Nmap done", result)

    def test_llm_interface_prompt_handling(self):
        prompt = "What is the capital of France?"
        response = self.llm_interface.get_response(prompt)
        self.assertEqual(response, "Paris")

if __name__ == '__main__':
    unittest.main()