class DecisionEngine:
    def __init__(self, task_planner, agent_coordinator):
        self.task_planner = task_planner
        self.agent_coordinator = agent_coordinator

    def route_task(self, task):
        """
        Routes the given task to the appropriate agent (DeepSeek R1 or Qwen-7B).
        """
        if task['type'] == 'recon':
            return self.agent_coordinator.assign_to_deepseek(task)
        elif task['type'] == 'exploit':
            return self.agent_coordinator.assign_to_qwen(task)
        else:
            raise ValueError("Unknown task type: {}".format(task['type']))

    def handle_decision(self, task):
        """
        Handles decision-making based on the task's context and requirements.
        """
        planned_task = self.task_planner.plan_task(task)
        return self.route_task(planned_task)