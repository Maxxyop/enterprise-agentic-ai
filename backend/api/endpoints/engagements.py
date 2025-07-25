from flask import Blueprint, request, jsonify
from backend.services.engagement_service import EngagementService
import logging

logger = logging.getLogger(__name__)

engagements_bp = Blueprint('engagements', __name__)
engagement_service = EngagementService()

@engagements_bp.route('/engagements', methods=['GET'])
def get_engagements():
    try:
        engagements = engagement_service.get_all_engagements()
        return jsonify(engagements), 200
    except Exception as e:
        logger.error(f'Error fetching engagements: {e}')
        return jsonify({'error': 'Database error'}), 500

@engagements_bp.route('/engagements/<int:engagement_id>', methods=['GET'])
def get_engagement(engagement_id):
    try:
        engagement = engagement_service.get_engagement_by_id(engagement_id)
        if engagement:
            return jsonify(engagement), 200
        return jsonify({'error': 'Engagement not found'}), 404
    except Exception as e:
        logger.error(f'Error fetching engagement: {e}')
        return jsonify({'error': 'Database error'}), 500

@engagements_bp.route('/engagements', methods=['POST'])
def create_engagement():
    data = request.json
    if not data or 'name' not in data:
        logger.warning('Invalid input for engagement creation')
        return jsonify({'error': 'Invalid input'}), 400
    try:
        new_engagement = engagement_service.create_engagement(data)
        return jsonify(new_engagement), 201
    except Exception as e:
        logger.error(f'Error creating engagement: {e}')
        return jsonify({'error': 'Database error'}), 500

@engagements_bp.route('/engagements/<int:engagement_id>', methods=['PUT'])
def update_engagement(engagement_id):
    data = request.json
    if not data or 'name' not in data:
        logger.warning('Invalid input for engagement update')
        return jsonify({'error': 'Invalid input'}), 400
    try:
        updated_engagement = engagement_service.update_engagement(engagement_id, data)
        if updated_engagement:
            return jsonify(updated_engagement), 200
        return jsonify({'error': 'Engagement not found'}), 404
    except Exception as e:
        logger.error(f'Error updating engagement: {e}')
        return jsonify({'error': 'Database error'}), 500

@engagements_bp.route('/engagements/<int:engagement_id>', methods=['DELETE'])
def delete_engagement(engagement_id):
    try:
        success = engagement_service.delete_engagement(engagement_id)
        if success:
            return jsonify({'message': 'Engagement deleted'}), 204
        return jsonify({'error': 'Engagement not found'}), 404
    except Exception as e:
        logger.error(f'Error deleting engagement: {e}')
        return jsonify({'error': 'Database error'}), 500