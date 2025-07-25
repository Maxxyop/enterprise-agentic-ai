# tool_generator.py

import os
import subprocess

class ToolGenerator:
    def __init__(self, tool_name, output_directory):
        self.tool_name = tool_name
        self.output_directory = output_directory

    def generate_script(self):
        script_content = self._create_script_content()
        script_path = os.path.join(self.output_directory, f"{self.tool_name}_script.py")
        
        with open(script_path, 'w') as script_file:
            script_file.write(script_content)
        
        print(f"Script generated at: {script_path}")
        return script_path

    def _create_script_content(self):
        return f"""# This script is generated for the tool: {self.tool_name}

def main():
    print("Executing {self.tool_name}...")

if __name__ == "__main__":
    main()
"""

    def validate_script(self, script_path):
        try:
            subprocess.run(['python', script_path], check=True)
            print("Script validation successful.")
        except subprocess.CalledProcessError as e:
            print(f"Script validation failed: {e}")

# Example usage:
# generator = ToolGenerator("example_tool", "/path/to/output")
# generator.generate_script()
# generator.validate_script("/path/to/output/example_tool_script.py")