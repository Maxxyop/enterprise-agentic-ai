# This file implements a lightweight container execution environment for running tasks in a Docker sandbox.

import docker

class DockerSandbox:
    def __init__(self, image_name):
        self.client = docker.from_env()
        self.image_name = image_name

    def create_container(self, command):
        container = self.client.containers.run(self.image_name, command, detach=True)
        return container

    def execute_command(self, container, command):
        exec_command = self.client.api.exec_create(container.id, command)
        output = self.client.api.exec_start(exec_command['Id'])
        return output.decode('utf-8')

    def stop_container(self, container):
        container.stop()
        container.remove()

    def run_task(self, command):
        container = self.create_container(command)
        output = self.execute_command(container, command)
        self.stop_container(container)
        return output

# Example usage:
# sandbox = DockerSandbox('your_docker_image')
# result = sandbox.run_task('your_command_here')
# print(result)