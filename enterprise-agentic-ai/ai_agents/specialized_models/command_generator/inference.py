# Inference logic for command generation

import json

class CommandGenerator:
    def __init__(self, command_templates_path):
        self.command_templates = self.load_command_templates(command_templates_path)

    def load_command_templates(self, path):
        with open(path, 'r') as file:
            return json.load(file)

    def generate_command(self, command_type, parameters):
        if command_type not in self.command_templates:
            raise ValueError(f"Command type '{command_type}' not found in templates.")
        
        template = self.command_templates[command_type]
        command = template.format(**parameters)
        return command

# Example usage
if __name__ == "__main__":
    generator = CommandGenerator('command_templates.json')
    command = generator.generate_command('nmap_scan', {'target': '192.168.1.1'})
    print(command)