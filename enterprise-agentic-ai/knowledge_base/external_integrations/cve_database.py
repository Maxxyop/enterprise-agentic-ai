import requests

class CVEDatabase:
    def __init__(self, api_url):
        self.api_url = api_url

    def get_cve_data(self, cve_id):
        response = requests.get(f"{self.api_url}/cve/{cve_id}")
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error fetching CVE data: {response.status_code}")

    def search_cves(self, query):
        response = requests.get(f"{self.api_url}/search", params={"query": query})
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error searching CVEs: {response.status_code}")

# Example usage:
# cve_db = CVEDatabase("https://cveapi.com/api")
# cve_data = cve_db.get_cve_data("CVE-2023-12345")
# print(cve_data)