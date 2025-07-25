import unittest
from orchestrator.core.task_planner import TaskPlanner

class TestTaskPlanner(unittest.TestCase):
    def test_plan_tasks(self):
        planner = TaskPlanner()
        planner.add_task('recon')
        plans = planner.plan_tasks()
        self.assertTrue(any(p['task_name'] == 'recon' for p in plans))
