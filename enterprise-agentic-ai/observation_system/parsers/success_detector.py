class SuccessDetector:
    def __init__(self):
        self.success_criteria = []

    def add_success_criterion(self, criterion):
        self.success_criteria.append(criterion)

    def detect_success(self, scan_results):
        successful_outcomes = []
        for result in scan_results:
            if self.is_successful(result):
                successful_outcomes.append(result)
        return successful_outcomes

    def is_successful(self, result):
        for criterion in self.success_criteria:
            if criterion in result:
                return True
        return False

    def clear_criteria(self):
        self.success_criteria.clear()