from flask import Blueprint, request, jsonify
from backend.services.engagement_service import EngagementService

engagements_bp = Blueprint('engagements', __name__)
engagement_service = EngagementService()

@engagements_bp.route('/engagements', methods=['GET'])
def get_engagements():
    engagements = engagement_service.get_all_engagements()
    return jsonify(engagements), 200

@engagements_bp.route('/engagements/<int:engagement_id>', methods=['GET'])
def get_engagement(engagement_id):
    engagement = engagement_service.get_engagement_by_id(engagement_id)
    if engagement:
        return jsonify(engagement), 200
    return jsonify({'error': 'Engagement not found'}), 404

@engagements_bp.route('/engagements', methods=['POST'])
def create_engagement():
    data = request.json
    new_engagement = engagement_service.create_engagement(data)
    return jsonify(new_engagement), 201

@engagements_bp.route('/engagements/<int:engagement_id>', methods=['PUT'])
def update_engagement(engagement_id):
    data = request.json
    updated_engagement = engagement_service.update_engagement(engagement_id, data)
    if updated_engagement:
        return jsonify(updated_engagement), 200
    return jsonify({'error': 'Engagement not found'}), 404

@engagements_bp.route('/engagements/<int:engagement_id>', methods=['DELETE'])
def delete_engagement(engagement_id):
    success = engagement_service.delete_engagement(engagement_id)
    if success:
        return jsonify({'message': 'Engagement deleted'}), 204
    return jsonify({'error': 'Engagement not found'}), 404