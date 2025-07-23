class AgentCoordinator:
    def __init__(self, deepseek_agent, qwen_agent):
        self.deepseek_agent = deepseek_agent
        self.qwen_agent = qwen_agent

    def coordinate_task(self, task):
        if task['type'] == 'reasoning':
            return self.deepseek_agent.process_task(task)
        elif task['type'] == 'command_execution':
            return self.qwen_agent.execute_command(task)
        else:
            raise ValueError("Unknown task type")

    def get_agent_status(self):
        return {
            'deepseek_status': self.deepseek_agent.get_status(),
            'qwen_status': self.qwen_agent.get_status()
        }

    def update_agents(self, updates):
        self.deepseek_agent.update(updates.get('deepseek', {}))
        self.qwen_agent.update(updates.get('qwen', {}))