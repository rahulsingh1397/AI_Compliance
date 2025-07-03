"""
Monitoring Agent for AI-Enhanced Data Privacy and Compliance Monitoring.

Key Features:
1. Real-time log ingestion from cloud (AWS S3, Azure Blob) and on-premises sources
2. Unsupervised ML models (Isolation Forest, Autoencoders) for anomaly detection
3. Prioritized alerts (low, medium, high) for dashboard integration
"""

import os
import json
import logging
import time
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from dataclasses import field
from pydantic import BaseModel, Field, validate_arguments, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from flask import Flask, jsonify, request

# Import base agent
import sys
import os.path
# Add the root directory to sys.path to allow proper imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from AIComplianceMonitoring.agents.base_agent import BaseAgent, BaseAgentConfig

# Configure structured logging
logger = logging.getLogger(__name__)

class MonitoringAgentConfig(BaseAgentConfig):
    """Configuration for Monitoring Agent"""
    agent_name: str = "monitoring_agent"
    
    # AWS S3 configurations
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    s3_log_buckets: List[str] = Field(default_factory=list)
    
    # Azure Blob configurations
    azure_connection_string: Optional[str] = None
    azure_containers: List[str] = Field(default_factory=list)
    
    # On-premises log configurations
    on_prem_log_paths: List[str] = Field(default_factory=list)
    state_directory: str = "monitoring_state"
    max_ingestion_threads: int = 10
    device: str = "cpu"  # Device for ML models ('cpu', 'cuda')
    model_directory: str = "./models"
    database_directory: str = "./db"
    
    # Anomaly detection configurations
    anomaly_models: List[str] = Field(default_factory=lambda: ["isolation_forest", "autoencoder"])
    anomaly_threshold: float = 0.95  # 95th percentile
    false_positive_target: float = 0.05  # 5% false positive rate
    
    # Alert configurations
    alert_levels: Dict[str, float] = Field(default_factory=lambda: {
        "low": 0.75,
        "medium": 0.85,
        "high": 0.95
    })
    alert_batch_size: int = 100
    alert_db_retention_days: int = 90  # Store alerts for 90 days

    class Config:
        arbitrary_types_allowed = True


class MonitoringAgent(BaseAgent):
    """
    Monitoring Agent for real-time log ingestion and anomaly detection.
    
    Features:
    - Real-time log ingestion from multiple sources
    - ML-based anomaly detection
    - Prioritized alerts for dashboard
    """
    
    @validate_arguments
    def __init__(self, 
                 config: Optional[MonitoringAgentConfig] = None,
                 log_ingestion_module: Optional[Any] = None,
                 anomaly_detection_module: Optional[Any] = None,
                 alert_module: Optional[Any] = None):
        """
        Initialize the Monitoring Agent with dependency injection.
        
        Args:
            config: Agent configuration
            log_ingestion_module: Pre-initialized log ingestion module
            anomaly_detection_module: Pre-initialized anomaly detection module
            alert_module: Pre-initialized alert module
        """
        # Set config first, as it's used by the modules
        self.config = config or MonitoringAgentConfig()

        logger.debug("Monitoring agent initializing specialized components")

        # Initialize the component modules BEFORE calling super().__init__
        # This ensures they exist when _initialize_resources is called by the parent.
        try:
            logger.debug("Importing LogIngestionModule...")
            from AIComplianceMonitoring.agents.monitoring.log_ingestion import LogIngestionModule
            logger.debug("LogIngestionModule imported successfully")

            logger.debug("Importing AnomalyDetectionModule...")
            from AIComplianceMonitoring.agents.monitoring.anomaly_detection import AnomalyDetectionModule
            logger.debug("AnomalyDetectionModule imported successfully")

            logger.debug("Importing AlertModule...")
            from AIComplianceMonitoring.agents.monitoring.alert_module import AlertModule
            logger.debug("AlertModule imported successfully")

            # Initialize components with dependency injection
            self.log_ingestion_module = log_ingestion_module or LogIngestionModule(self.config)
            self.anomaly_detection_module = anomaly_detection_module or AnomalyDetectionModule(self.config)
            self.alert_module = alert_module or AlertModule(self.config)

        except ImportError as e:
            logger.error(f"Failed to import required modules: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error initializing Monitoring Agent components: {str(e)}")
            raise

        # Now, initialize the base agent. This will call _initialize_resources.
        super().__init__(config=self.config)

        logger.info("Monitoring agent and its components initialized successfully")
    
    def _initialize_resources(self):
        """Initialize resources needed by the monitoring agent"""
        super()._initialize_resources()
        logger.debug("Initializing monitoring-specific resources")
        
        # Initialize connections to data sources
        self.log_ingestion_module.initialize_connections()
        
        # Initialize ML models
        self.anomaly_detection_module.initialize_models()
        
        # Initialize alert database
        self.alert_module.initialize_alert_db()
        
        # Initialize Flask app
        self.app = self._create_flask_app()
    
    def _cleanup_resources(self):
        """Clean up resources used by the monitoring agent"""
        logger.debug("Cleaning up monitoring-specific resources")
        
        # Close connections to data sources
        if hasattr(self, 'log_ingestion_module') and self.log_ingestion_module:
            self.log_ingestion_module.close_connections()
        
        super()._cleanup_resources()
        logger.info(f"{self.config.agent_name} has been shut down.")
    
    def start_monitoring(self, sources: Optional[List[str]] = None):
        """
        Start monitoring logs from specified sources.
        If no sources are specified, monitor all configured sources.
        
        Args:
            sources: List of source identifiers to monitor
        """
        logger.info(f"Starting log monitoring for sources: {sources or 'all'}")
        
        try:
            # Start the monitoring process
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                
                # Submit log ingestion tasks
                for source_type, source_configs in self.log_ingestion_module.get_available_sources().items():
                    if sources and source_type not in sources:
                        continue
                    
                    for source_config in source_configs:
                        futures.append(
                            executor.submit(
                                self._monitor_source, 
                                source_type, 
                                source_config
                            )
                        )
                
                # Process results as they complete
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        logger.debug(f"Monitoring task completed with result: {result}")
                    except Exception as e:
                        logger.error(f"Monitoring task failed: {str(e)}")
            
            logger.info("All monitoring tasks completed")
            return {"status": "completed", "message": "Monitoring completed successfully"}
            
        except Exception as e:
            error_msg = f"Error in start_monitoring: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    def _monitor_source(self, source_type: str, source_config: Dict[str, Any]):
        """
        Monitor a specific log source.
        
        Args:
            source_type: Type of the source (aws_s3, azure_blob, on_prem)
            source_config: Configuration for the source
        """
        logger.info(f"Monitoring source: {source_type} - {source_config.get('name', 'unnamed')}")
        
        try:
            # Ingest logs from the source
            logs_df = self._execute_with_retry(
                self.log_ingestion_module.ingest_logs,
                source_type,
                source_config
            )
            
            if logs_df is None or logs_df.empty:
                logger.info(f"No new logs from source: {source_type} - {source_config.get('name', 'unnamed')}")
                return {"status": "success", "message": "No new logs", "alert_count": 0}
            
            # Detect anomalies in the logs
            anomalies_df = self._execute_with_retry(
                self.anomaly_detection_module.detect_anomalies,
                logs_df
            )
            
            if anomalies_df is None or anomalies_df.empty:
                logger.info(f"No anomalies detected in logs from: {source_type} - {source_config.get('name', 'unnamed')}")
                return {"status": "success", "message": "No anomalies detected", "alert_count": 0}
            
            # Generate alerts for the detected anomalies
            alert_results = self._execute_with_retry(
                self.alert_module.generate_alerts,
                anomalies_df,
                source_type,
                source_config
            )
            
            logger.info(f"Generated {alert_results.get('alert_count', 0)} alerts from source: "
                       f"{source_type} - {source_config.get('name', 'unnamed')}")
            
            return {
                "status": "success",
                "message": "Monitoring completed",
                "alert_count": alert_results.get("alert_count", 0),
                "source_type": source_type,
                "source_name": source_config.get("name", "unnamed")
            }
            
        except Exception as e:
            error_msg = f"Error monitoring source {source_type}: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    def get_alerts(self, 
                   priority: Optional[str] = None, 
                   limit: int = 100, 
                   offset: int = 0) -> Dict[str, Any]:
        """
        Get alerts with optional filtering.
        
        Args:
            priority: Filter by priority (low, medium, high)
            limit: Maximum number of alerts to return
            offset: Pagination offset
        
        Returns:
            Dictionary with alerts and metadata
        """
        try:
            alerts = self.alert_module.get_alerts(priority, limit, offset)
            return {
                "status": "success",
                "data": alerts,
                "count": len(alerts),
                "limit": limit,
                "offset": offset
            }
        except Exception as e:
            error_msg = f"Error retrieving alerts: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics for dashboard display"""
        try:
            # Get statistics from each module
            ingestion_stats = self.log_ingestion_module.get_stats()
            detection_stats = self.anomaly_detection_module.get_stats()
            alert_stats = self.alert_module.get_stats()
            
            return {
                "status": "success",
                "data": {
                    "ingestion": ingestion_stats,
                    "detection": detection_stats,
                    "alerts": alert_stats,
                    "timestamp": time.time()
                }
            }
        except Exception as e:
            error_msg = f"Error retrieving monitoring stats: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

    def _create_flask_app(self):
        app = Flask(self.config.agent_name)

        @app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint."""
            health_status = self.health_check()
            return jsonify(health_status)

        @app.route('/stats', methods=['GET'])
        def get_stats():
            """Endpoint to get statistics from all modules."""
            stats = self.get_monitoring_stats()
            return jsonify(stats)

        @app.route('/alerts', methods=['GET'])
        def get_alerts():
            """Endpoint to get recent alerts."""
            limit = request.args.get('limit', 100, type=int)
            priority = request.args.get('priority', None, type=str)
            offset = request.args.get('offset', 0, type=int)
            alerts = self.get_alerts(priority=priority, limit=limit, offset=offset)
            return jsonify(alerts)

        return app

    def run(self, host='0.0.0.0', port=5001):
        """Starts the agent's API server."""
        logger.info(f"Starting {self.config.agent_name} API server on {host}:{port}")
        # The agent serves data on demand via API endpoints.
        # Continuous monitoring would be handled by a separate scheduled task or trigger.
        self.app.run(host=host, port=port, debug=False)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    logger.info("Initializing Monitoring Agent as a service...")
    
    try:
        config = MonitoringAgentConfig()
        agent = MonitoringAgent(config=config)
        agent.run(host='0.0.0.0', port=5001)
    except Exception as e:
        logger.error(f"Failed to start Monitoring Agent service: {e}", exc_info=True)
        sys.exit(1)
