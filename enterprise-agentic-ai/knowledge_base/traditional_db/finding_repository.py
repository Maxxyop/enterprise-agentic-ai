class FindingRepository:
    def __init__(self, db_connection):
        self.db_connection = db_connection

    def add_finding(self, finding):
        """Add a new finding to the repository."""
        with self.db_connection:
            self.db_connection.execute(
                "INSERT INTO findings (description, severity, date_found) VALUES (?, ?, ?)",
                (finding['description'], finding['severity'], finding['date_found'])
            )

    def get_findings(self):
        """Retrieve all findings from the repository."""
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT * FROM findings")
        return cursor.fetchall()

    def get_findings_by_severity(self, severity):
        """Retrieve findings filtered by severity."""
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT * FROM findings WHERE severity = ?", (severity,))
        return cursor.fetchall()

    def delete_finding(self, finding_id):
        """Delete a finding from the repository by its ID."""
        with self.db_connection:
            self.db_connection.execute(
                "DELETE FROM findings WHERE id = ?", (finding_id,)
            )