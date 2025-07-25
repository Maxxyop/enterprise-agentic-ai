import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class AgentCoordinator:
    def __init__(self, deepseek_agent, qwen_agent):
        self.deepseek_agent = deepseek_agent
        self.qwen_agent = qwen_agent

    def coordinate_task(self, task: Dict[str, Any]) -> Any:
        """Coordinate tasks between agents with validation and error handling."""
        if not isinstance(task, dict) or 'type' not in task:
            logger.error('Invalid task format')
            raise ValueError('Invalid task format')
        if task['type'] == 'reasoning':
            return self.deepseek_agent.process_task(task)
        elif task['type'] == 'command_execution':
            return self.qwen_agent.execute_command(task)
        else:
            logger.error(f"Unknown task type: {task['type']}")
            raise ValueError("Unknown task type")

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        try:
            return {
                'deepseek_status': self.deepseek_agent.get_status(),
                'qwen_status': self.qwen_agent.get_status()
            }
        except Exception as e:
            logger.error(f"Failed to get agent status: {e}")
            return {"error": str(e)}

    def update_agents(self, updates: Dict[str, Any]) -> None:
        """Update agent configurations."""
        try:
            self.deepseek_agent.update(updates.get('deepseek', {}))
            self.qwen_agent.update(updates.get('qwen', {}))
            logger.info('Agents updated successfully')
        except Exception as e:
            logger.error(f"Failed to update agents: {e}")