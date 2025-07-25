class GoalManager:
    def __init__(self):
        self.goals = []

    def add_goal(self, goal):
        """Add a new goal to the list of goals."""
        self.goals.append(goal)

    def remove_goal(self, goal):
        """Remove a goal from the list of goals."""
        if goal in self.goals:
            self.goals.remove(goal)

    def get_goals(self):
        """Retrieve the current list of goals."""
        return self.goals

    def clear_goals(self):
        """Clear all goals."""
        self.goals.clear()

    def prioritize_goals(self):
        """Prioritize goals based on predefined criteria."""
        # Placeholder for prioritization logic
        self.goals.sort()  # Example: sort goals alphabetically

    def display_goals(self):
        """Display all current goals."""
        for goal in self.goals:
            print(f"- {goal}")