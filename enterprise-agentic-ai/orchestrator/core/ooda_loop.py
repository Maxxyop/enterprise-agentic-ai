# OODA Loop Implementation for Task Orchestration

class OODA:
    def __init__(self):
        self.observation = None
        self.orientation = None
        self.decide = None
        self.action = None

    def observe(self, data):
        """Collects data from various sources."""
        self.observation = data
        print("Observation completed.")

    def orient(self):
        """Analyzes the observed data to understand the context."""
        if self.observation:
            self.orientation = f"Analyzed data: {self.observation}"
            print("Orientation completed.")
        else:
            print("No observation data to orient.")

    def decide(self):
        """Makes decisions based on the oriented data."""
        if self.orientation:
            self.decide = f"Decision based on: {self.orientation}"
            print("Decision made.")
        else:
            print("No orientation data to decide.")

    def act(self):
        """Implements the decision made."""
        if self.decide:
            self.action = f"Action taken based on: {self.decide}"
            print("Action executed.")
        else:
            print("No decision made to act upon.")

    def run_loop(self, data):
        """Runs the OODA loop."""
        self.observe(data)
        self.orient()
        self.decide()
        self.act()

# Example usage
if __name__ == "__main__":
    ooda = OODA()
    ooda.run_loop("Sample data for task orchestration")