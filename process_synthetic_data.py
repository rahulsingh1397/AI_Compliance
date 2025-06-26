#!/usr/bin/env python
"""
Process Synthetic Data Script

This script loads synthetic data from our existing datasets and processes it through
the monitoring agent's pipeline to generate real anomalies and alerts.

The script:
1. Loads data from CSV files in the data directory
2. Processes the data through the log ingestion module
3. Allows the anomaly detection module to identify anomalies
4. Lets the alert module generate alerts for the dashboard
"""

import os
import sys
import logging
import pandas as pd
import time
from pathlib import Path

# Add the project root to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import monitoring agent components
from AIComplianceMonitoring.agents.monitoring.agent import MonitoringAgent, MonitoringAgentConfig
from AIComplianceMonitoring.agents.monitoring.log_ingestion import LogIngestionModule, SourceConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('process_synthetic_data')

def main():
    """Process synthetic data through the monitoring agent pipeline"""
    logger.info("Starting synthetic data processing")
    
    # Initialize the monitoring agent
    config = MonitoringAgentConfig(
        # Use default values but specify paths to data
        on_prem_log_paths=["AIComplianceMonitoring/data/"]  
    )
    
    agent = MonitoringAgent(config=config)
    logger.info("Monitoring agent initialized")
    
    # Create source configurations
    sources = [
        {
            "type": "on_prem",
            "name": "hr_records",
            "path": "AIComplianceMonitoring/data/5000000 HRA Records.csv",
            "format": "csv",
            "credentials": {},
            "options": {
                "sample_size": 5000  # Process a manageable amount of data
            }
        },
        {
            "type": "on_prem",
            "name": "ip_credit_email",
            "path": "AIComplianceMonitoring/data/ip-creditcard-email.csv",
            "format": "csv",
            "credentials": {},
            "options": {}
        }
    ]
    
    # Process each source
    for source_config in sources:
        try:
            logger.info(f"Processing source: {source_config['name']}")
            agent._monitor_source(source_config["type"], source_config)
            logger.info(f"Source {source_config['name']} processed successfully")
        except Exception as e:
            logger.error(f"Error processing source {source_config['name']}: {e}")
    
    # Get and print statistics
    stats = agent.get_monitoring_stats()
    logger.info(f"Processing complete. Stats: {stats}")
    
    # Get and print alerts
    alerts = agent.get_alerts(limit=10)
    logger.info(f"Alerts generated: {alerts}")
    
    logger.info("Data processing complete. Alerts have been generated and are available via the API")
    logger.info("You can now view them in the dashboard at http://127.0.0.1:5000")

if __name__ == "__main__":
    main()
