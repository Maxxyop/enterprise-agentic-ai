import sqlite3

class SQLiteVulnDB:
    def __init__(self, db_name='vulnerabilities.db'):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS vulnerabilities (
                id INTEGER PRIMARY KEY,
                cve_id TEXT NOT NULL,
                description TEXT,
                severity TEXT,
                published_date TEXT,
                modified_date TEXT
            )
        ''')
        self.connection.commit()

    def add_vulnerability(self, cve_id, description, severity, published_date, modified_date):
        self.cursor.execute('''
            INSERT INTO vulnerabilities (cve_id, description, severity, published_date, modified_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (cve_id, description, severity, published_date, modified_date))
        self.connection.commit()

    def get_vulnerability(self, cve_id):
        self.cursor.execute('''
            SELECT * FROM vulnerabilities WHERE cve_id = ?
        ''', (cve_id,))
        return self.cursor.fetchone()

    def get_all_vulnerabilities(self):
        self.cursor.execute('SELECT * FROM vulnerabilities')
        return self.cursor.fetchall()

    def close(self):
        self.connection.close()