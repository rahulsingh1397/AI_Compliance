import os
import sys
import logging

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from AIComplianceMonitoring.agents.monitoring.alert_module import AlertModule

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def populate_test_alerts():
    """Generates and saves a few sample alerts to the database."""
    log.info("Starting database population...")

    config = {
        'database_path': os.path.join(os.path.dirname(__file__), '..', '..', '..', 'instance', 'alerts.db'),
        'deduplication_window_seconds': 3600
    }

    try:
        alert_module = AlertModule(config)
        log.info("AlertModule initialized.")

        # Clear existing alerts to ensure a clean slate
        # In a real scenario, you might not do this
        alert_module.clear_all_alerts()
        log.info("Cleared existing alerts.")

        test_alerts = [
            {'priority': 'high', 'message': 'High-risk data access from unauthorized IP', 'details': {'ip': '198.51.100.23', 'user': 'unknown'}},
            {'priority': 'medium', 'message': 'Multiple failed login attempts for admin account', 'details': {'user': 'admin', 'attempts': 15}},
            {'priority': 'low', 'message': 'Unusual login time for user `jdoe`', 'details': {'user': 'jdoe', 'time': '03:15 UTC'}},
            {'priority': 'high', 'message': 'Potential PII leak detected in outbound traffic', 'details': {'policy_id': 'PII-001', 'destination': 'external.net'}},
            {'priority': 'medium', 'message': 'Anomalous database query pattern detected', 'details': {'db_user': 'app_service', 'query_pattern': 'SELECT * FROM users'}}
        ]

        for alert_data in test_alerts:
            alert_module.create_alert(alert_data['priority'], alert_data['message'], alert_data['details'])
        
        log.info(f"Successfully inserted {len(test_alerts)} test alerts into the database.")

    except Exception as e:
        log.error("Failed to populate database.", exc_info=True)

if __name__ == '__main__':
    populate_test_alerts()
