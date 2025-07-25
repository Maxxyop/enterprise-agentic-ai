import unittest
from ai_agents.foundation_models.llm_interface import LLMInterface

class TestLLMInterface(unittest.TestCase):
    def setUp(self):
        self.llm = LLMInterface()

    def test_get_response(self):
        self.assertEqual(self.llm.get_response('What is the capital of France?'), 'Paris')
        self.assertEqual(self.llm.get_response('Other prompt'), 'Response')

if __name__ == '__main__':
    unittest.main()
