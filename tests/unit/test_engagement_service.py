import unittest
from backend.services.engagement_service import validate_engagement_data, db_create_engagement
from unittest.mock import patch, MagicMock

class TestEngagementService(unittest.TestCase):
    def test_validate_engagement_data_valid(self):
        data = {'name': 'Test Engagement'}
        self.assertTrue(validate_engagement_data(data))

    def test_validate_engagement_data_invalid(self):
        self.assertFalse(validate_engagement_data({'name': ''}))
        self.assertFalse(validate_engagement_data({}))
        self.assertFalse(validate_engagement_data({'name': None}))

    @patch('backend.services.engagement_service.get_db_connection')
    def test_db_create_engagement(self, mock_get_db):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.lastrowid = 1
        mock_get_db.return_value = mock_conn
        data = {'name': 'Test Engagement', 'description': 'Desc'}
        result = db_create_engagement(data)
        self.assertEqual(result, {'id': 1, 'name': 'Test Engagement'})
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()

if __name__ == '__main__':
    unittest.main()
