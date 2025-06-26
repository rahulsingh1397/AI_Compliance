from flask import Blueprint, jsonify
from flask_login import login_required

api_bp = Blueprint('api', __name__)

@api_bp.route('/compliance_data')
@login_required
def compliance_data():
    # Mock data - replace with real data from Data Discovery Agent
    data = {
        'sensitive_files': 12,
        'total_scans': 45,
        'risk_level': 3,
        'last_scan': '2023-06-12T14:30:00',
        'sensitive_data_types': {
            'PII': 8,
            'Financial': 3,
            'Health': 1
        },
        'compliance_status': {
            'GDPR': 85,
            'CCPA': 70,
            'HIPAA': 90
        }
    }
    return jsonify({'status': 'success', 'data': data})

@api_bp.route('/sensitive_data')
@login_required
def sensitive_data():
    # Mock data - replace with real data from Data Discovery Agent
    data = {
        'PII': 8,
        'Financial': 3,
        'Health': 1
    }
    return jsonify({'status': 'success', 'data': data})

@api_bp.route('/risk_levels')
@login_required
def risk_levels():
    # Mock data - replace with real data from Risk Assessment Agent
    data = {
        'overall_risk': 3,
        'gdpr_risk': 2,
        'ccpa_risk': 3,
        'hipaa_risk': 1
    }
    return jsonify({'status': 'success', 'data': data})

@api_bp.route('/user_preferences', methods=['GET', 'PUT'])
@login_required
def user_preferences():
    if request.method == 'PUT':
        # Handle preference updates
        pass
    # Return current preferences
    return jsonify({'status': 'success', 'data': current_user.preferences})
