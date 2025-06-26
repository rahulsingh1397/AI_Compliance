"""
Alert Module for the Monitoring Agent.

This module handles the generation and management of prioritized alerts:
- Low, medium, and high priority alerts (FR2.4)
- Alert database management
- Dashboard integration

Features:
- Prioritization based on configured thresholds
- Persistence in database
- Retrieval API for dashboard integration
"""

import os
from pathlib import Path
import logging
import time
import pandas as pd
import json
import sqlite3
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# Import RL feedback components


# Configure logging
logger = logging.getLogger(__name__)

class AlertModule:
    """
    Handles the generation and management of prioritized alerts.
    """
    
    def __init__(self, config, feedback_manager=None):
        """
        Initialize the alert module.
        
        Args:
            config: Configuration object with necessary parameters
            feedback_manager: Optional RLFeedbackManager instance for anomaly validation
        """
        logger.debug("Initializing AlertModule")
        self.config = config
        self.db_path = Path(self.config.database_directory) / 'alerts.db'
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        self.db_connection = None
        self.feedback_manager = feedback_manager
        self.stats = {
            "alerts_generated": {
                "low": 0,
                "medium": 0,
                "high": 0
            },
            "alerts_active": 0,
            "validation_enhanced": 0,
            "oldest_alert": None,
            "newest_alert": None,
            "false_positives_prevented": 0
        }
        self.initialize_alert_db()
    
    def initialize_alert_db(self):
        """Initialize the alert database and create tables if they don't exist."""
        try:
            self.db_connection = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.db_connection.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    message TEXT NOT NULL,
                    source_type TEXT,
                    status TEXT DEFAULT 'new',
                    details TEXT,
                    is_validated BOOLEAN DEFAULT FALSE
                )
            ''')
            self.db_connection.commit()
            logger.info(f"Alert database initialized at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}", exc_info=True)
            self.db_connection = None
    
    def generate_alerts(self, 
                       anomalies_df: pd.DataFrame, 
                       source_type: str, 
                       source_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate alerts from detected anomalies.
        
        Args:
            anomalies_df: DataFrame containing anomalous logs
            source_type: Type of the source (aws_s3, azure_blob, on_prem)
            source_config: Configuration for the source
            
        Returns:
            Dictionary with alert generation statistics
        """
        from .rl_feedback_manager import RLFeedbackManager # Lazy import

        if not self.db_connection:
            logger.error("Cannot generate alerts: Database connection not available.")
            return {}

        logger.info(f"Generating alerts from {len(anomalies_df)} anomalies")
        
        alert_counts = {"low": 0, "medium": 0, "high": 0}
        alerts_created = 0
        false_positives_filtered = 0
        validated_count = 0
        
        for _, anomaly_row in anomalies_df.iterrows():
            anomaly_data = anomaly_row.to_dict()
            alert_id = f"anom-{int(time.time() * 1000)}-{_}"
            anomaly_data["id"] = alert_id
            anomaly_data["source_type"] = source_type
            anomaly_data["detected_at"] = datetime.now().isoformat()

            score = anomaly_data.get("anomaly_score", 0.5)
            if score > self.config.get('high_sev_threshold', 0.8):
                priority = "high"
            elif score > self.config.get('medium_sev_threshold', 0.5):
                priority = "medium"
            else:
                priority = "low"
            anomaly_data["severity"] = priority

            should_alert, is_validated = True, False
            if self.feedback_manager:
                try:
                    validation_result = self.feedback_manager.process_anomaly(anomaly_data)
                    validated_count += 1
                    if validation_result.get("action") == "suppress":
                        should_alert = False
                        false_positives_filtered += 1
                    is_validated = validation_result.get("is_true_positive", False)
                except Exception as e:
                    logger.error(f"Error during RL feedback processing: {e}", exc_info=True)

            if should_alert:
                if self._is_duplicate(anomaly_data):
                    logger.debug(f"Duplicate anomaly detected, skipping alert generation for {alert_id}")
                    continue

                alert_message = self._generate_alert_message(anomaly_data)
                details_json = json.dumps({k: str(v) for k, v in anomaly_data.items()})

                try:
                    cursor = self.db_connection.cursor()
                    cursor.execute(
                        "INSERT INTO alerts (id, timestamp, priority, message, source_type, details, is_validated) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (alert_id, anomaly_data["detected_at"], priority, alert_message, source_type, details_json, is_validated)
                    )
                    self.db_connection.commit()
                    alerts_created += 1
                    alert_counts[priority] += 1
                except sqlite3.Error as e:
                    logger.error(f"Failed to insert alert into database: {e}", exc_info=True)

        for p, count in alert_counts.items():
            self.stats["alerts_generated"][p] += count
        self.stats["false_positives_prevented"] += false_positives_filtered
        
        logger.info(f"Alert generation complete. Created: {alerts_created}, Filtered: {false_positives_filtered}")
        
        return {
            "alerts_created": alerts_created,
            "false_positives_filtered": false_positives_filtered,
            "validated_count": validated_count,
            "priority_counts": alert_counts
        }
    
    def get_alerts(self, 
                  priority: Optional[str] = None, 
                  limit: int = 100, 
                  offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get alerts with optional filtering.
        
        Args:
            priority: Filter by priority (low, medium, high)
            limit: Maximum number of alerts to return
            offset: Pagination offset
            
        Returns:
            List of alert dictionaries
        """
        if not self.db_connection:
            logger.error("Database connection not available.")
            return []

        try:
            self.db_connection.row_factory = sqlite3.Row
            cursor = self.db_connection.cursor()
            
            query = "SELECT * FROM alerts"
            params = []
            if priority:
                query += " WHERE priority = ?"
                params.append(priority)
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            alerts = [dict(row) for row in cursor.fetchall()]
            for alert in alerts:
                if 'details' in alert and isinstance(alert['details'], str):
                    alert['details'] = json.loads(alert['details'])
            return alerts
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve alerts: {e}", exc_info=True)
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about alerts"""
        if self.db_connection:
            try:
                cursor = self.db_connection.cursor()
                cursor.execute("SELECT COUNT(*) FROM alerts WHERE status = 'new'")
                self.stats['alerts_active'] = cursor.fetchone()[0]
                cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM alerts")
                oldest, newest = cursor.fetchone()
                self.stats['oldest_alert'] = oldest
                self.stats['newest_alert'] = newest
            except sqlite3.Error as e:
                logger.error(f"Could not update stats from DB: {e}")

        return {
            **self.stats,
            "timestamp": datetime.now().isoformat()
        }

    def _is_duplicate(self, anomaly_data: Dict[str, Any], time_window_minutes: int = 60) -> bool:
        """Check if a similar alert has been generated recently."""
        if not self.db_connection:
            return False

        message = self._generate_alert_message(anomaly_data)
        time_threshold = (datetime.now() - timedelta(minutes=time_window_minutes)).isoformat()

        try:
            cursor = self.db_connection.cursor()
            cursor.execute(
                "SELECT 1 FROM alerts WHERE message = ? AND timestamp >= ?",
                (message, time_threshold)
            )
            return cursor.fetchone() is not None
        except sqlite3.Error as e:
            logger.error(f"Failed to check for duplicate alerts: {e}")
            return False
    
    def _generate_alert_message(self, anomaly_data: Dict[str, Any]) -> str:
        """Generate a human-readable alert message from anomaly data
        
        Args:
            anomaly_data: Dictionary with anomaly information
            
        Returns:
            Human-readable alert message
        """
        # Determine anomaly type
        anomaly_type = anomaly_data.get("type", "unknown")
        
        # Messages by type
        if "unauthorized_access" in anomaly_type.lower():
            user_id = anomaly_data.get("user_id", "unknown user")
            resource = anomaly_data.get("resource_id", "sensitive resource")
            return f"Unauthorized access attempt by {user_id} to {resource}"
            
        elif "unusual_transfer" in anomaly_type.lower():
            size = anomaly_data.get("data_size", "unknown size")
            source = anomaly_data.get("source", "unknown source")
            destination = anomaly_data.get("destination", "unknown destination")
            return f"Unusual data transfer of {size} from {source} to {destination}"
            
        elif "suspicious_behavior" in anomaly_type.lower():
            user_id = anomaly_data.get("user_id", "unknown user")
            behavior = anomaly_data.get("behavior", "suspicious activity")
            return f"Suspicious behavior detected: {user_id} - {behavior}"
            
        elif "system_activity" in anomaly_type.lower():
            system = anomaly_data.get("system", "unknown system")
            activity = anomaly_data.get("activity", "anomalous activity")
            return f"Anomalous system activity detected in {system}: {activity}"
            
        elif "compliance_violation" in anomaly_type.lower():
            regulation = anomaly_data.get("regulation", "compliance policy")
            violation = anomaly_data.get("violation", "policy violation")
            return f"{regulation} violation detected: {violation}"
            
        else:
            # Generic message for other types
            score = anomaly_data.get("anomaly_score", 0)
            return f"Anomaly detected with confidence score {score:.2f}"
    
    def _extract_affected_resources(self, anomaly_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract information about affected resources from anomaly data
        
        Args:
            anomaly_data: Dictionary with anomaly information
            
        Returns:
            List of affected resources with their attributes
        """
        resources = []
        
        # Add user if present
        if "user_id" in anomaly_data:
            resources.append({
                "type": "user",
                "id": str(anomaly_data["user_id"]),
                "name": anomaly_data.get("user_name", "Unknown User")
            })
        
        # Add data resource if present
        if "resource_id" in anomaly_data:
            resources.append({
                "type": "data_resource",
                "id": str(anomaly_data["resource_id"]),
                "name": anomaly_data.get("resource_name", "Unknown Resource")
            })
        
        # Add system if present
        if "system" in anomaly_data:
            resources.append({
                "type": "system",
                "id": str(anomaly_data["system"]),
                "name": anomaly_data.get("system_name", "Unknown System")
            })
        
        return resources

    def clear_all_alerts(self):
        """Deletes all records from the alerts table."""
        if not self.db_connection:
            logger.error("Cannot clear alerts: Database connection not available.")
            return

        try:
            cursor = self.db_connection.cursor()
            cursor.execute("DELETE FROM alerts")
            self.db_connection.commit()
            logger.info("All alerts have been cleared from the database.")
        except sqlite3.Error as e:
            logger.error(f"Failed to clear alerts: {e}", exc_info=True)

    def create_alert(self, priority: str, message: str, details: Dict[str, Any], source_type: str = "manual"):
        """Creates a single alert and saves it to the database."""
        if not self.db_connection:
            logger.error("Cannot create alert: Database connection not available.")
            return None

        alert_id = f"manual-{int(time.time() * 1000)}"
        timestamp = datetime.now().isoformat()
        details_json = json.dumps(details)
        status = "new"
        is_validated = False

        try:
            cursor = self.db_connection.cursor()
            cursor.execute(
                """
                INSERT INTO alerts (id, timestamp, priority, message, source_type, status, details, is_validated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (alert_id, timestamp, priority, message, source_type, status, details_json, is_validated)
            )
            self.db_connection.commit()
            logger.info(f"Manual alert created with ID: {alert_id}")
            return alert_id
        except sqlite3.Error as e:
            logger.error(f"Failed to create manual alert: {e}", exc_info=True)
            return None
