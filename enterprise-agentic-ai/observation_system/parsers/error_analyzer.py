class ErrorAnalyzer:
    def __init__(self):
        self.error_logs = []

    def analyze_error(self, error_message):
        """
        Analyzes the given error message and categorizes it.
        """
        # Example categorization logic
        if "timeout" in error_message:
            category = "Timeout Error"
        elif "connection" in error_message:
            category = "Connection Error"
        else:
            category = "General Error"

        self.error_logs.append({
            "message": error_message,
            "category": category
        })

    def get_error_summary(self):
        """
        Returns a summary of analyzed errors.
        """
        summary = {}
        for log in self.error_logs:
            category = log["category"]
            summary[category] = summary.get(category, 0) + 1
        return summary

    def clear_logs(self):
        """
        Clears the error logs.
        """
        self.error_logs = []