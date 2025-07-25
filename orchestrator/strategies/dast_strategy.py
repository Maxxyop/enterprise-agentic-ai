class DASTStrategy:
    def __init__(self):
        self.recon_tasks = []
        self.exploitation_tasks = []

    def add_recon_task(self, task):
        self.recon_tasks.append(task)

    def add_exploitation_task(self, task):
        self.exploitation_tasks.append(task)

    def execute_recon(self):
        for task in self.recon_tasks:
            print(f"Executing reconnaissance task: {task}")
            # Implement the logic for executing the reconnaissance task here

    def execute_exploitation(self):
        for task in self.exploitation_tasks:
            print(f"Executing exploitation task: {task}")
            # Implement the logic for executing the exploitation task here

    def run(self):
        print("Starting DAST strategy execution...")
        self.execute_recon()
        self.execute_exploitation()
        print("DAST strategy execution completed.")