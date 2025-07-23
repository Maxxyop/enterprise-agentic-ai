class CommandValidator:
    def __init__(self):
        self.valid_commands = ["nmap", "sqlmap", "zap", "burp"]

    def validate_command(self, command):
        """
        Validates the command before execution.

        Args:
            command (str): The command to validate.

        Returns:
            bool: True if the command is valid, False otherwise.
        """
        return command in self.valid_commands

    def validate_parameters(self, command, parameters):
        """
        Validates the parameters for the given command.

        Args:
            command (str): The command for which parameters are being validated.
            parameters (dict): The parameters to validate.

        Returns:
            bool: True if parameters are valid, False otherwise.
        """
        # Placeholder for parameter validation logic
        # This can be expanded based on command requirements
        return True

    def run_validation(self, command, parameters):
        """
        Runs the validation for the command and its parameters.

        Args:
            command (str): The command to validate.
            parameters (dict): The parameters to validate.

        Returns:
            bool: True if both command and parameters are valid, False otherwise.
        """
        return self.validate_command(command) and self.validate_parameters(command, parameters)