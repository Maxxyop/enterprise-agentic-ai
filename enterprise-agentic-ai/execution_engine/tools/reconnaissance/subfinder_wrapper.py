# Subfinder Wrapper for Execution

import subprocess

class SubfinderWrapper:
    def __init__(self, target):
        self.target = target

    def run_subfinder(self):
        command = ["subfinder", "-d", self.target]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return result.stdout.splitlines()
        except subprocess.CalledProcessError as e:
            print(f"Error running subfinder: {e.stderr}")
            return []

# Example usage
if __name__ == "__main__":
    target_domain = "example.com"  # Replace with the target domain
    subfinder = SubfinderWrapper(target_domain)
    subdomains = subfinder.run_subfinder()
    print("Found subdomains:", subdomains)