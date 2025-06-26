import pandas as pd
import os
import sys
import logging
from datetime import datetime
import time
from AIComplianceMonitoring.agents.monitoring.alert_module import AlertModule

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_anomaly_dataframe():
    """Create a DataFrame with test anomalies"""
    # Create sample anomalies of different types
    anomalies = [
        {
            "type": "unauthorized_access",
            "user_id": "user123",
            "resource_id": "db_pii_table",
            "timestamp": datetime.now().isoformat(),
            "anomaly_score": 0.89,
            "priority": "high"
        },
        {
            "type": "unusual_transfer",
            "data_size": "1.2GB",
            "source": "internal_network",
            "destination": "external_ip",
            "timestamp": datetime.now().isoformat(),
            "anomaly_score": 0.75,
            "priority": "medium"
        },
        {
            "type": "suspicious_behavior",
            "user_id": "admin_user",
            "behavior": "multiple failed login attempts",
            "timestamp": datetime.now().isoformat(),
            "anomaly_score": 0.68,
            "priority": "medium"
        },
        {
            "type": "system_activity",
            "system": "payment_processor",
            "activity": "unusual API calls",
            "timestamp": datetime.now().isoformat(),
            "anomaly_score": 0.92,
            "priority": "high"
        },
        {
            "type": "compliance_violation",
            "regulation": "GDPR",
            "violation": "unencrypted PII data transfer",
            "timestamp": datetime.now().isoformat(),
            "anomaly_score": 0.81,
            "priority": "high"
        },
        {
            "type": "suspicious_behavior",
            "user_id": "guest_user",
            "behavior": "accessing sensitive files",
            "timestamp": datetime.now().isoformat(),
            "anomaly_score": 0.62,
            "priority": "low"
        }
    ]
    
    return pd.DataFrame(anomalies)

def main():
    """Main function to generate test alerts"""
    logger.info("Generating test alerts for the dashboard")
    
    # Configuration for AlertModule
    config = {
        'database_directory': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance'),
        'deduplication_window_seconds': 3600
    }
    
    # Create AlertModule instance
    alert_module = AlertModule(config)
    
    # Create test anomalies
    anomalies_df = create_test_anomaly_dataframe()
    logger.info(f"Created {len(anomalies_df)} test anomalies")
    
    # Generate alerts
    source_type = "test_data"
    source_config = {"name": "test_source"}
    
    stats = alert_module.generate_alerts(
        anomalies_df=anomalies_df,
        source_type=source_type,
        source_config=source_config
    )
    
    logger.info(f"Alert generation complete: {stats}")
    
    # Display alerts from the database
    alerts = alert_module.get_alerts()
    logger.info(f"Retrieved {len(alerts)} alerts from database")
    
    # Display alert stats
    stats = alert_module.get_stats()
    logger.info(f"Alert stats: {stats}")
    
    logger.info("Test alerts have been generated successfully!")
    logger.info("You can now refresh the dashboard to see the data")

if __name__ == "__main__":
    main()
