from datetime import datetime
import requests

class ApplicationMonitor:
    def __init__(self, target_url):
        self.target_url = target_url
        self.last_checked = None
        self.status = None

    def check_application(self):
        try:
            response = requests.get(self.target_url)
            self.status = response.status_code
            self.last_checked = datetime.now()
            self.log_status()
        except requests.exceptions.RequestException as e:
            self.status = 'Error'
            self.last_checked = datetime.now()
            self.log_error(e)

    def log_status(self):
        with open('application_monitor.log', 'a') as log_file:
            log_file.write(f"{self.last_checked}: {self.target_url} - Status: {self.status}\n")

    def log_error(self, error):
        with open('application_monitor.log', 'a') as log_file:
            log_file.write(f"{self.last_checked}: {self.target_url} - Error: {error}\n")