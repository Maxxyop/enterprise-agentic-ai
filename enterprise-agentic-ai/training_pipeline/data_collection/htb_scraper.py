import requests
from bs4 import BeautifulSoup

class HTBScraper:
    def __init__(self):
        self.base_url = "https://www.hackthebox.com"
        self.session = requests.Session()

    def login(self, username, password):
        login_url = f"{self.base_url}/login"
        payload = {
            'username': username,
            'password': password
        }
        response = self.session.post(login_url, data=payload)
        return response.ok

    def scrape_challenges(self):
        challenges_url = f"{self.base_url}/challenges"
        response = self.session.get(challenges_url)
        if response.ok:
            soup = BeautifulSoup(response.text, 'html.parser')
            challenges = self.extract_challenges(soup)
            return challenges
        return []

    def extract_challenges(self, soup):
        challenges = []
        for challenge in soup.find_all('div', class_='challenge-card'):
            title = challenge.find('h3').text.strip()
            difficulty = challenge.find('span', class_='difficulty').text.strip()
            challenges.append({
                'title': title,
                'difficulty': difficulty
            })
        return challenges

    def close(self):
        self.session.close()

if __name__ == "__main__":
    scraper = HTBScraper()
    if scraper.login('your_username', 'your_password'):
        challenges = scraper.scrape_challenges()
        for challenge in challenges:
            print(f"Title: {challenge['title']}, Difficulty: {challenge['difficulty']}")
    scraper.close()