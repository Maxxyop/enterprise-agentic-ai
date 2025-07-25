from flask import Blueprint, request, jsonify
from backend.core.database import get_db_connection
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

engagement_service = Blueprint('engagement_service', __name__)

def validate_engagement_data(data: Dict[str, Any]) -> bool:
    """Validate engagement input data."""
    return bool(data and isinstance(data.get('name'), str) and data['name'].strip())

def db_create_engagement(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new engagement in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO engagements (name, description) VALUES (?, ?)",
                   (data['name'], data.get('description', '')))
    conn.commit()
    engagement_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return {'id': engagement_id, 'name': data['name']}

@engagement_service.route('/engagements', methods=['POST'])
def create_engagement():
    """Create a new engagement (MVP-ready, robust, secure)."""
    data = request.json
    if not validate_engagement_data(data):
        logger.warning('Invalid input for engagement creation')
        return jsonify({'error': 'Invalid input'}), 400
    try:
        result = db_create_engagement(data)
        logger.info(f"Engagement created with ID {result['id']}")
        return jsonify(result), 201
    except Exception as e:
        logger.error(f'Error creating engagement: {e}')
        return jsonify({'error': 'Database error'}), 500

@engagement_service.route('/engagements/<int:engagement_id>', methods=['GET'])
def get_engagement(engagement_id: int):
    """Get engagement details by ID."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM engagements WHERE id = ?", (engagement_id,))
        engagement = cursor.fetchone()
        cursor.close()
        conn.close()
        if engagement is None:
            logger.warning(f'Engagement {engagement_id} not found')
            return jsonify({'error': 'Engagement not found'}), 404
        return jsonify({'id': engagement[0], 'name': engagement[1], 'description': engagement[2]})
    except Exception as e:
        logger.error(f'Error fetching engagement: {e}')
        return jsonify({'error': 'Database error'}), 500

@engagement_service.route('/engagements/<int:engagement_id>', methods=['PUT'])
def update_engagement(engagement_id: int):
    """Update engagement details."""
    data = request.json
    if not validate_engagement_data(data):
        logger.warning('Invalid input for engagement update')
        return jsonify({'error': 'Invalid input'}), 400
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM engagements WHERE id = ?", (engagement_id,))
        engagement = cursor.fetchone()
        if engagement is None:
            cursor.close()
            conn.close()
            logger.warning(f'Engagement {engagement_id} not found for update')
            return jsonify({'error': 'Engagement not found'}), 404
        cursor.execute("UPDATE engagements SET name = ?, description = ? WHERE id = ?",
                       (data['name'], data.get('description', ''), engagement_id))
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f'Engagement {engagement_id} updated')
        return jsonify({'id': engagement_id, 'name': data['name']}), 200
    except Exception as e:
        logger.error(f'Error updating engagement: {e}')
        return jsonify({'error': 'Database error'}), 500

@engagement_service.route('/engagements/<int:engagement_id>', methods=['DELETE'])
def delete_engagement(engagement_id: int):
    """Delete engagement by ID."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM engagements WHERE id = ?", (engagement_id,))
        engagement = cursor.fetchone()
        if engagement is None:
            cursor.close()
            conn.close()
            logger.warning(f'Engagement {engagement_id} not found for deletion')
            return jsonify({'error': 'Engagement not found'}), 404
        cursor.execute("DELETE FROM engagements WHERE id = ?", (engagement_id,))
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f'Engagement {engagement_id} deleted')
        return jsonify({'message': 'Engagement deleted'}), 204
    except Exception as e:
        logger.error(f'Error deleting engagement: {e}')
        return jsonify({'error': 'Database error'}), 500

# Placeholder for future API key integration
# def get_api_key():
#     return os.getenv('AGENTIC_API_KEY', 'YOUR_API_KEY_HERE')