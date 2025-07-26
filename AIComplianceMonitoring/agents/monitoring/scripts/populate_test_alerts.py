"""
Test Alert Population Script

This script populates the monitoring database with sample alerts for testing purposes.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

def setup_logging(log_level=logging.INFO):
    """Configure logging for the script."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Populate database with test alerts')
    parser.add_argument('--db-path', type=str,
                       help='Path to the alerts database file')
    parser.add_argument('--clear-existing', action='store_true',
                       help='Clear existing alerts before adding new ones')
    parser.add_argument('--count', type=int, default=5,
                       help='Number of test alerts to create (default: 5)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    return parser.parse_args()

logger = logging.getLogger(__name__)

def get_sample_alerts(count=5):
    """Generate sample alert data."""
    base_alerts = [
        {
            'priority': 'high', 
            'message': 'High-risk data access from unauthorized IP', 
            'details': {'ip': '198.51.100.23', 'user': 'unknown', 'resource': 'sensitive_data.db'}
        },
        {
            'priority': 'medium', 
            'message': 'Multiple failed login attempts for admin account', 
            'details': {'user': 'admin', 'attempts': 15, 'source_ip': '192.168.1.100'}
        },
        {
            'priority': 'low', 
            'message': 'Unusual login time for user jdoe', 
            'details': {'user': 'jdoe', 'time': '03:15 UTC', 'location': 'unusual'}
        },
        {
            'priority': 'high', 
            'message': 'Potential PII leak detected in outbound traffic', 
            'details': {'policy_id': 'PII-001', 'destination': 'external.net', 'data_type': 'SSN'}
        },
        {
            'priority': 'medium', 
            'message': 'Anomalous database query pattern detected', 
            'details': {'db_user': 'app_service', 'query_pattern': 'SELECT * FROM users', 'frequency': 'high'}
        },
        {
            'priority': 'high',
            'message': 'Unauthorized file access attempt',
            'details': {'file': '/etc/passwd', 'user': 'guest', 'action': 'read'}
        },
        {
            'priority': 'low',
            'message': 'Disk usage threshold exceeded',
            'details': {'disk': '/var/log', 'usage': '95%', 'threshold': '90%'}
        }
    ]
    
    # Return the requested number of alerts, cycling through if needed
    return (base_alerts * ((count // len(base_alerts)) + 1))[:count]

def populate_test_alerts(db_path=None, clear_existing=False, count=5):
    """Populate the database with test alerts."""
    logger.info("Starting test alert population...")
    
    # Determine database path
    if not db_path:
        db_path = Path(__file__).parent.parent.parent.parent / 'instance' / 'alerts.db'
    
    config = {
        'database_path': str(db_path),
        'deduplication_window_seconds': 3600
    }
    
    try:
        # Add project root to Python path
        project_root = Path(__file__).parent.parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        from AIComplianceMonitoring.agents.monitoring.alert_module import AlertModule
        
        alert_module = AlertModule(config)
        logger.info("AlertModule initialized successfully")
        
        if clear_existing:
            alert_module.clear_all_alerts()
            logger.info("Cleared existing alerts")
        
        test_alerts = get_sample_alerts(count)
        
        success_count = 0
        for alert_data in test_alerts:
            try:
                alert_module.create_alert(
                    alert_data['priority'], 
                    alert_data['message'], 
                    alert_data['details']
                )
                success_count += 1
            except Exception as e:
                logger.warning(f"Failed to create alert: {alert_data['message'][:50]}... Error: {e}")
        
        logger.info(f"Successfully inserted {success_count}/{len(test_alerts)} test alerts")
        
    except ImportError as e:
        logger.error(f"Failed to import AlertModule: {e}")
        logger.error("Make sure the project dependencies are properly installed")
        return False
    except Exception as e:
        logger.error(f"Failed to populate database: {e}", exc_info=True)
        return False
    
    return True

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    success = populate_test_alerts(
        db_path=args.db_path,
        clear_existing=args.clear_existing,
        count=args.count
    )
    
    if not success:
        sys.exit(1)
    
    logger.info("Test alert population completed successfully")

if __name__ == '__main__':
    main()
