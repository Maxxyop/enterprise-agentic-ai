import unittest
from orchestrator.core.agent_coordinator import AgentCoordinator

class DummyAgent:
    def process_task(self, task):
        if task.get('type') == 'reasoning':
            return {'success': True}
        raise ValueError('Invalid task type')
    def execute_command(self, task):
        return {'executed': True}
    def get_status(self):
        return 'ready'
    def update(self, config):
        pass

class TestAgentCoordinator(unittest.TestCase):
    def setUp(self):
        self.coordinator = AgentCoordinator(DummyAgent(), DummyAgent())

    def test_coordinate_task_reasoning(self):
        result = self.coordinator.coordinate_task({'type': 'reasoning'})
        self.assertTrue(result['success'])

    def test_coordinate_task_command_execution(self):
        result = self.coordinator.coordinate_task({'type': 'command_execution'})
        self.assertTrue(result['executed'])

    def test_coordinate_task_invalid(self):
        with self.assertRaises(ValueError):
            self.coordinator.coordinate_task({'type': 'invalid'})

    def test_get_agent_status(self):
        status = self.coordinator.get_agent_status()
        self.assertEqual(status['deepseek_status'], 'ready')
        self.assertEqual(status['qwen_status'], 'ready')

    def test_update_agents(self):
        self.coordinator.update_agents({'deepseek': {}, 'qwen': {}})
        # No exception should be raised

if __name__ == '__main__':
    unittest.main()
