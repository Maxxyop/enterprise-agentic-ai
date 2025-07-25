class ReportingService:
    def __init__(self, database):
        self.database = database

    def generate_report(self, engagement_id):
        findings = self.database.get_findings(engagement_id)
        report = self._format_report(findings)
        return report

    def _format_report(self, findings):
        report = "Vulnerability Assessment Report\n"
        report += "=" * 40 + "\n"
        for finding in findings:
            report += f"Title: {finding['title']}\n"
            report += f"Severity: {finding['severity']}\n"
            report += f"Description: {finding['description']}\n"
            report += f"Recommendations: {finding['recommendations']}\n"
            report += "-" * 40 + "\n"
        return report

    def save_report(self, report, file_path):
        with open(file_path, 'w') as file:
            file.write(report)