import requests
import logging
import os
import sys
import json
from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required
from datetime import datetime

# Configuration for mock data behavior
class MockConfig:
    # Set to False to disable mock data fallback in production
    ENABLE_MOCK_FALLBACK = False

# Configure logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Add console handler for immediate visibility
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
log.addHandler(console)

# This blueprint is for serving data to the frontend.
api_bp = Blueprint('api', __name__)

# Default Monitoring API URL (will be overridden by environment or config when available)
DEFAULT_API_URL = "http://127.0.0.1:5001"

def format_timestamp(iso_string):
    """Formats ISO string to a more readable 'YYYY-MM-DD HH:MM:SS' format."""
    if not iso_string:
        return "N/A"
    try:
        # Handles timezone 'Z' correctly
        if iso_string.endswith('Z'):
            iso_string = iso_string[:-1] + '+00:00'
        dt_object = datetime.fromisoformat(iso_string)
        return dt_object.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
        log.warning(f"Could not parse timestamp: {iso_string}")
        return "Invalid Date"

def get_mock_data(current_time=None):
    """Generate mock data for testing"""
    if current_time is None:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return {
        'key_metrics': {
            'sensitive_files': 12,
            'total_scans': 45,
            'risk_level': 3,
            'last_scan': current_time
        },
        'sensitive_data_types': {
            'High Priority': 3,
            'Medium Priority': 12,
            'Low Priority': 27
        },
        'compliance_status': {'GDPR': 95, 'CCPA': 88, 'HIPAA': 92},
        'recent_alerts': [
            {'timestamp': current_time, 'message': 'Mock alert 1', 'severity': 'High'},
            {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'message': 'Mock alert 2', 'severity': 'Medium'}
        ],
        'is_mock_data': True,
        'mock_generated_at': current_time
    }

def transform_report_to_dashboard_view(report):
    """Transforms the raw report structure into the format expected by the dashboard frontend."""
    log.debug("Transforming raw report for dashboard view.")
    
    report_content = report.get('report_data', {}).get('content', {})
    
    # Placeholders for metrics not present in the GDPR report.
    # These would typically come from a discovery scan summary.
    sensitive_files = report.get('sensitive_files', 25)
    total_scans = report.get('total_scans', 50)
    risk_level = report.get('risk_level', 'Medium')
    
    last_scan = report.get('created_at', report.get('end_date'))

    # Summarize sensitive data types from processing activities
    sensitive_data_types = {}
    activities = report_content.get('processing_activities', [])
    if activities:
        for activity in activities:
            for category in activity.get('categories', []):
                # Normalize category names for better display
                normalized_category = category.replace("_", " ").title()
                sensitive_data_types[normalized_category] = sensitive_data_types.get(normalized_category, 0) + 1
    else:
        sensitive_data_types = {'PII': 5, 'Financial': 10, 'Health': 3}

    # Placeholder for compliance status
    compliance_status = {'GDPR': 98, 'CCPA': 75, 'HIPAA': 85}

    dashboard_data = {
        'key_metrics': {
            'sensitive_files': sensitive_files,
            'total_scans': total_scans,
            'risk_level': risk_level,
            'last_scan': last_scan
        },
        'sensitive_data_types': sensitive_data_types,
        'compliance_status': compliance_status,
        'recent_alerts': [],
        'is_mock_data': False,
        'source_report_id': report.get('report_id'),
        'source_report_type': report.get('report_type')
    }
    
    log.debug(f"Transformed data: {dashboard_data}")
    return dashboard_data

@api_bp.route('/compliance_data')
@login_required
def compliance_data():
    """Endpoint to get compliance data for the dashboard"""
    try:
        # Path to the report file is in the project root, relative to this file's location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        report_file_path = os.path.join(current_dir, '..', '..', '..', 'latest_report.json')
        
        log.info(f"Attempting to read compliance data from: {report_file_path}")
        
        if os.path.exists(report_file_path):
            with open(report_file_path, 'r') as f:
                data = json.load(f)
            log.info("Successfully loaded compliance data from file.")
            # Transform the data for the dashboard view
            dashboard_view_data = transform_report_to_dashboard_view(data)
            return jsonify(dashboard_view_data)
        else:
            log.warning(f"Report file not found at {report_file_path}. Falling back to mock data.")
            # Check if mock fallback is enabled
            if not MockConfig.ENABLE_MOCK_FALLBACK:
                return jsonify({
                    "error": "Compliance report not found",
                    "details": f"File not found at {report_file_path}",
                    "mock_fallback_enabled": False
                }), 404
            
            # Use mock data as fallback
            log.warning("Using mock data as fallback")
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            mock_data = get_mock_data(current_time)
            return jsonify(mock_data)
            
    except Exception as e:
        log.exception(f"An error occurred in compliance_data endpoint: {e}")
        print(f"CONSOLE ERROR: An error occurred in compliance_data endpoint: {e}")
        # Check if mock data is enabled
        if not MockConfig.ENABLE_MOCK_FALLBACK:
            return jsonify({"error": "An internal error occurred while processing data."}), 500
            
        # Generate current timestamp for mock data
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Return mock data to prevent frontend crash even on general errors
        mock_data = {
            'key_metrics': {
                'sensitive_files': 42,
                'total_scans': 150,
                'risk_level': 'Medium',
                'last_scan_date': current_time
            },
            'sensitive_data_types': {
                'High Priority': 3,
                'Medium Priority': 12,
                'Low Priority': 27
            },
            'compliance_status': {'GDPR': 95, 'CCPA': 88, 'HIPAA': 92},
            'recent_alerts': [
                {'timestamp': current_time, 'message': 'Mock alert 1', 'severity': 'High'},
                {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'message': 'Mock alert 2', 'severity': 'Medium'}
            ],
            'is_mock_data': True,  # Flag to indicate this is mock data
            'mock_generated_at': current_time,
            'error_type': 'general_error'  # Additional context about the error
        }
        log.info("Returning mock data due to general error")
        print(f"CONSOLE: Returning mock data due to general error at {current_time}")
        return jsonify(mock_data)
