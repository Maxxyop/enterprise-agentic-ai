import unittest
from knowledge_base.external_integrations.cve_database import CVEDatabase

class TestCVEDatabase(unittest.TestCase):
    def test_fetch_cve(self):
        db = CVEDatabase()
        # Minimal stub: should not raise exception
        try:
            db.fetch_cve('CVE-2023-0001')
        except Exception as e:
            self.fail(f"CVEDatabase fetch_cve raised exception: {e}")
