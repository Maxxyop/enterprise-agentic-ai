from flask import Blueprint, request, jsonify
from backend.core.database import get_db_connection

engagement_service = Blueprint('engagement_service', __name__)

@engagement_service.route('/engagements', methods=['POST'])
def create_engagement():
    data = request.json
    # Validate input data
    if not data or 'name' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    # Create a new engagement in the database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO engagements (name, description) VALUES (?, ?)", 
                   (data['name'], data.get('description', '')))
    conn.commit()
    engagement_id = cursor.lastrowid
    cursor.close()
    conn.close()

    return jsonify({'id': engagement_id, 'name': data['name']}), 201

@engagement_service.route('/engagements/<int:engagement_id>', methods=['GET'])
def get_engagement(engagement_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM engagements WHERE id = ?", (engagement_id,))
    engagement = cursor.fetchone()
    cursor.close()
    conn.close()

    if engagement is None:
        return jsonify({'error': 'Engagement not found'}), 404

    return jsonify({'id': engagement[0], 'name': engagement[1], 'description': engagement[2]})

@engagement_service.route('/engagements/<int:engagement_id>', methods=['PUT'])
def update_engagement(engagement_id):
    data = request.json
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE engagements SET name = ?, description = ? WHERE id = ?", 
                   (data['name'], data.get('description', ''), engagement_id))
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({'id': engagement_id, 'name': data['name']}), 200

@engagement_service.route('/engagements/<int:engagement_id>', methods=['DELETE'])
def delete_engagement(engagement_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM engagements WHERE id = ?", (engagement_id,))
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({'result': 'Engagement deleted'}), 204