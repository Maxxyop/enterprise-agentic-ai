from flask import Blueprint, request, jsonify
from backend.services.engagement_service import EngagementService
from backend.services.ai_agents_service import AIAgentsService

ai_orchestration_service = Blueprint('ai_orchestration_service', __name__)

@ai_orchestration_service.route('/orchestrate', methods=['POST'])
def orchestrate():
    data = request.json
    if not data or 'task' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    task = data['task']
    engagement_service = EngagementService()
    ai_agents_service = AIAgentsService()

    # Example orchestration logic
    engagement = engagement_service.create_engagement(task)
    result = ai_agents_service.execute_task(engagement)

    return jsonify({'result': result}), 200

@ai_orchestration_service.route('/status/<engagement_id>', methods=['GET'])
def get_status(engagement_id):
    engagement_service = EngagementService()
    status = engagement_service.get_engagement_status(engagement_id)

    if status is None:
        return jsonify({'error': 'Engagement not found'}), 404

    return jsonify({'status': status}), 200