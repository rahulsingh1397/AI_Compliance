import os
import requests
import logging
import json
from datetime import datetime

log = logging.getLogger(__name__)

def helper_format_timestamp(iso_string):
    """Helper to parse and re-format timestamps into ISO format for the frontend."""
    if not iso_string or iso_string == 'N/A':
        return "N/A"
    try:
        # Handle different possible date formats gracefully
        if ' ' in iso_string and '+' not in iso_string and 'Z' not in iso_string.upper():
             # Attempt to parse formats like 'YYYY-MM-DD HH:MM:SS'
            dt_object = datetime.strptime(iso_string, '%Y-%m-%d %H:%M:%S')
        else:
            if iso_string.endswith('Z'):
                iso_string = iso_string[:-1] + '+00:00'
            dt_object = datetime.fromisoformat(iso_string)
        
        # Return in ISO format for JavaScript to handle
        return dt_object.isoformat()
    except (ValueError, TypeError):
        log.warning(f"Could not parse date: {iso_string}")
        return "Invalid Date"

def get_fallback_data_from_db():
    """Fetches and aggregates recent scan data from the database as a fallback."""
    from .models import ScanHistory
    print("CONSOLE: Using aggregated scan data from DB as fallback")

    recent_scans = ScanHistory.query.order_by(ScanHistory.start_time.desc()).limit(5).all()
    total_scans = ScanHistory.query.count()

    if not recent_scans:
        # Return a zeroed-out structure if no scans are found
        return {
            'key_metrics': {'sensitive_files': 0, 'total_scans': 0, 'risk_level': 'Low', 'last_scan_date': 'N/A'},
            'sensitive_data_types': {'High Priority': 0, 'Medium Priority': 0, 'Low Priority': 0},
            'compliance_status': {'GDPR': 0, 'CCPA': 0, 'HIPAA': 0},
            'recent_alerts': []
        }

    latest_scan = recent_scans[0]
    total_alerts = sum(s.sensitive_files_found for s in recent_scans)
    avg_alerts = total_alerts / len(recent_scans) if recent_scans else 0

    # Determine overall risk level based on the latest scan
    risk_level = 'Low'
    if latest_scan.sensitive_files_found > 10:
        risk_level = 'High'
    elif latest_scan.sensitive_files_found > 3:
        risk_level = 'Medium'

    # No simulated compliance scores - only return if we have actual compliance data
    compliance_status = {}  # Empty unless we have real compliance scan results

    return {
        'key_metrics': {
            'sensitive_files': latest_scan.sensitive_files_found,
            'total_scans': total_scans,
            'risk_level': risk_level,
            'last_scan_date': latest_scan.start_time.isoformat() if latest_scan.start_time else 'N/A'
        },
        'sensitive_data_types': {
            'High Priority': 0,  # Only populated from actual scan classification data
            'Medium Priority': 0,  # Only populated from actual scan classification data
            'Low Priority': 0   # Only populated from actual scan classification data
        },
        'compliance_status': compliance_status,
        'recent_alerts': []  # Only populated from actual alert/monitoring data
    }

def fetch_live_compliance_data():
    """Fetches and processes live compliance data from the monitoring API."""
    try:
        DEFAULT_API_URL = "http://127.0.0.1:5001"
        monitoring_api_url = os.environ.get('MONITORING_API_BASE_URL') or DEFAULT_API_URL
        
        try:
            stats_response = requests.get(f"{monitoring_api_url}/stats", timeout=5)
            alerts_response = requests.get(f"{monitoring_api_url}/alerts?limit=5", timeout=5)

            stats_response.raise_for_status()
            alerts_response.raise_for_status()

            ingestion_stats = stats_response.json()
            alert_stats = alerts_response.json()

            # Process data
            recent_alerts = alert_stats.get('alerts', [])
            for alert in recent_alerts:
                alert['timestamp'] = helper_format_timestamp(alert.get('timestamp'))

            alerts_by_priority = alert_stats.get('alerts_by_priority', {})
            risk_level = 'Low'
            if alerts_by_priority.get('high', 0) > 0:
                risk_level = 'High'
            elif alerts_by_priority.get('medium', 0) > 0:
                risk_level = 'Medium'

            key_metrics = {
                'sensitive_files': ingestion_stats.get('sensitive_files_found', 0),
                'total_scans': ingestion_stats.get('total_scans', 0),
                'risk_level': risk_level,
                'last_scan_date': helper_format_timestamp(alert_stats.get('newest_alert'))
            }

            sensitive_data_types = {
                'High Priority': alerts_by_priority.get('high', 0),
                'Medium Priority': alerts_by_priority.get('medium', 0),
                'Low Priority': alerts_by_priority.get('low', 0)
            }

            compliance_status = {'GDPR': 95, 'CCPA': 88, 'HIPAA': 92} # Mocked for now

            return {
                'key_metrics': key_metrics,
                'sensitive_data_types': sensitive_data_types,
                'compliance_status': compliance_status,
                'recent_alerts': recent_alerts
            }

        except requests.RequestException as e:
            log.error(f"API request failed: {type(e).__name__}: {e}")
            return get_fallback_data_from_db()
        
    except Exception as e:
        log.error(f"Unexpected error in fetch_live_compliance_data: {e}")
        return get_fallback_data_from_db()
