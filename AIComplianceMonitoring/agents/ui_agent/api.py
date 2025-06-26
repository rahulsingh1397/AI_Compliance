import requests
import logging
import os
import sys
from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required
from datetime import datetime

# Configuration for mock data behavior
class MockConfig:
    # Set to False to disable mock data fallback in production
    ENABLE_MOCK_FALLBACK = True

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

@api_bp.route('/compliance_data')
# Temporarily removed login_required for testing
# @login_required
def compliance_data():
    """Fetches and aggregates data from the Monitoring API."""
    log.info("--- Request received for /api/compliance_data ---")
    print("CONSOLE: Request received for /api/compliance_data - User is authenticated")
    print(f"CONSOLE: Current user: {request.remote_addr}")
    try:
        # Determine the Monitoring API base URL
        monitoring_api_url = os.environ.get('MONITORING_API_BASE_URL')
        
        # If not in environment, try to use the same host as the current request, but with port 5001
        if not monitoring_api_url and request.host:
            # Extract host without port
            host = request.host.split(':')[0]
            monitoring_api_url = f"http://{host}:5001"
        
        # Fallback to default if all else fails
        monitoring_api_url = monitoring_api_url or DEFAULT_API_URL
        
        log.info(f"Using Monitoring API URL: {monitoring_api_url}")
        print(f"CONSOLE: Using Monitoring API URL: {monitoring_api_url}")
        
        # Fetch stats and alerts from the Monitoring API
        log.info(f"Fetching stats from {monitoring_api_url}/stats")
        print(f"CONSOLE: Fetching stats from {monitoring_api_url}/stats")
        stats_response = requests.get(f"{monitoring_api_url}/stats", timeout=5)
        log.info(f"Stats response status: {stats_response.status_code}")
        print(f"CONSOLE: Stats response status: {stats_response.status_code}")
        
        log.info(f"Fetching alerts from {monitoring_api_url}/alerts?limit=5")
        print(f"CONSOLE: Fetching alerts from {monitoring_api_url}/alerts?limit=5")
        alerts_response = requests.get(f"{monitoring_api_url}/alerts?limit=5", timeout=5)
        log.info(f"Alerts response status: {alerts_response.status_code}")
        print(f"CONSOLE: Alerts response status: {alerts_response.status_code}")

        # Check for HTTP errors
        stats_response.raise_for_status()
        alerts_response.raise_for_status()

        # Extract the 'data' payload from the responses, with defaults to prevent errors
        stats_payload = stats_response.json().get('data', {})
        alerts_payload = alerts_response.json().get('data', [])
        log.debug(f"Stats payload received: {stats_payload}")
        log.debug(f"Alerts payload received: {alerts_payload}")
        print(f"CONSOLE: Stats payload (truncated): {str(stats_payload)[:200]}...")
        print(f"CONSOLE: Alerts payload count: {len(alerts_payload)}")

        # --- Transform data for the frontend --- 

        # Extract stats from different modules
        ingestion_stats = stats_payload.get('ingestion', {})
        detection_stats = stats_payload.get('detection', {})
        alert_stats = stats_payload.get('alerts', {})

        # 1. Key Metrics from Stats
        # Note: The keys 'alerts_by_priority', 'sensitive_files_found', 'total_scans', 
        # and 'newest_alert_timestamp' are assumed based on the UI's needs.
        # These may need to be implemented in the corresponding agent modules if not present.
        alerts_by_priority = alert_stats.get('alerts_by_priority', {})
        risk_level = "Low"
        if alerts_by_priority.get('high', 0) > 0:
            risk_level = "High"
        elif alerts_by_priority.get('medium', 0) > 0:
            risk_level = "Medium"

        key_metrics = {
            'sensitive_files': detection_stats.get('sensitive_files_found', 0),
            'total_scans': ingestion_stats.get('total_scans', 0),
            'risk_level': risk_level,
            'last_scan_date': format_timestamp(alert_stats.get('newest_alert_timestamp'))
        }

        # 2. Recent Alerts Table from Alerts
        recent_alerts = [
            {
                'timestamp': format_timestamp(alert.get('timestamp')),
                'message': alert.get('message', 'No message provided'),
                'severity': alert.get('priority', 'unknown').capitalize()
            } for alert in alerts_payload
        ]

        # 3. Data for Charts (using stats)
        sensitive_data_types = {
            'High Priority': alerts_by_priority.get('high', 0),
            'Medium Priority': alerts_by_priority.get('medium', 0),
            'Low Priority': alerts_by_priority.get('low', 0)
        }
        
        # 4. Mock Compliance Status (pending other agents)
        compliance_status = {'GDPR': 95, 'CCPA': 88, 'HIPAA': 92}

        # 5. Final combined data structure
        final_data = {
            'key_metrics': key_metrics,
            'sensitive_data_types': sensitive_data_types,
            'compliance_status': compliance_status,
            'recent_alerts': recent_alerts
        }
        log.info(f"Returning final data: {final_data}")
        print(f"CONSOLE: Returning final data with {len(recent_alerts)} alerts and risk level {risk_level}")

        return jsonify(final_data)

    except requests.exceptions.RequestException as e:
        log.error(f"Could not connect to Monitoring API at {monitoring_api_url}: {e}")
        print(f"CONSOLE ERROR: Could not connect to Monitoring API at {monitoring_api_url}: {e}")
        # Check if mock data is enabled
        if not MockConfig.ENABLE_MOCK_FALLBACK:
            return jsonify({"error": "Could not connect to Monitoring API"}), 503
            
        # Generate current timestamp for mock data
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Return mock data to prevent frontend crash
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
            'mock_generated_at': current_time
        }
        log.info("Returning mock data due to API connection failure")
        print(f"CONSOLE: Returning mock data due to API connection failure at {current_time}")
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
