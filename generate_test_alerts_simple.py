import sqlite3
import os
import sys
import logging
from datetime import datetime
import time
import json
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define database path
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'alerts.db')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def initialize_db():
    """Initialize the database if it doesn't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create alerts table if it doesn't exist
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
    conn.commit()
    logger.info(f"Alert database initialized at {DB_PATH}")
    return conn

def create_test_alerts(conn):
    """Create test alerts directly in the database"""
    cursor = conn.cursor()
    
    # Sample alerts of different types
    test_alerts = [
        {
            "id": f"anom-{uuid.uuid4()}",
            "timestamp": datetime.now().isoformat(),
            "priority": "high",
            "message": "Unauthorized access attempt by user123 to db_pii_table",
            "source_type": "test_data",
            "status": "new",
            "details": json.dumps({
                "type": "unauthorized_access",
                "user_id": "user123",
                "resource_id": "db_pii_table",
                "anomaly_score": 0.89
            }),
            "is_validated": True
        },
        {
            "id": f"anom-{uuid.uuid4()}",
            "timestamp": datetime.now().isoformat(),
            "priority": "medium",
            "message": "Unusual data transfer of 1.2GB from internal_network to external_ip",
            "source_type": "test_data",
            "status": "new",
            "details": json.dumps({
                "type": "unusual_transfer",
                "data_size": "1.2GB",
                "source": "internal_network",
                "destination": "external_ip",
                "anomaly_score": 0.75
            }),
            "is_validated": True
        },
        {
            "id": f"anom-{uuid.uuid4()}",
            "timestamp": datetime.now().isoformat(),
            "priority": "medium",
            "message": "Suspicious behavior detected: admin_user - multiple failed login attempts",
            "source_type": "test_data",
            "status": "new",
            "details": json.dumps({
                "type": "suspicious_behavior", 
                "user_id": "admin_user",
                "behavior": "multiple failed login attempts",
                "anomaly_score": 0.68
            }),
            "is_validated": True
        },
        {
            "id": f"anom-{uuid.uuid4()}",
            "timestamp": datetime.now().isoformat(),
            "priority": "high",
            "message": "Anomalous system activity detected in payment_processor: unusual API calls",
            "source_type": "test_data",
            "status": "new",
            "details": json.dumps({
                "type": "system_activity",
                "system": "payment_processor",
                "activity": "unusual API calls",
                "anomaly_score": 0.92
            }),
            "is_validated": True
        },
        {
            "id": f"anom-{uuid.uuid4()}",
            "timestamp": datetime.now().isoformat(),
            "priority": "high",
            "message": "GDPR violation detected: unencrypted PII data transfer",
            "source_type": "test_data",
            "status": "new",
            "details": json.dumps({
                "type": "compliance_violation",
                "regulation": "GDPR",
                "violation": "unencrypted PII data transfer",
                "anomaly_score": 0.81
            }),
            "is_validated": True
        },
        {
            "id": f"anom-{uuid.uuid4()}",
            "timestamp": datetime.now().isoformat(),
            "priority": "low",
            "message": "Suspicious behavior detected: guest_user - accessing sensitive files",
            "source_type": "test_data",
            "status": "new",
            "details": json.dumps({
                "type": "suspicious_behavior",
                "user_id": "guest_user",
                "behavior": "accessing sensitive files",
                "anomaly_score": 0.62
            }),
            "is_validated": True
        }
    ]
    
    # Insert alerts into the database
    for alert in test_alerts:
        cursor.execute(
            """
            INSERT INTO alerts (id, timestamp, priority, message, source_type, status, details, is_validated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alert["id"], 
                alert["timestamp"], 
                alert["priority"], 
                alert["message"],
                alert["source_type"],
                alert["status"],
                alert["details"],
                alert["is_validated"]
            )
        )
    
    conn.commit()
    logger.info(f"Added {len(test_alerts)} test alerts to the database")

def update_stats(conn):
    """Update the stats in the database directly"""
    cursor = conn.cursor()
    
    # Count alerts by priority
    cursor.execute("SELECT COUNT(*) FROM alerts WHERE priority = 'high'")
    high_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM alerts WHERE priority = 'medium'")
    medium_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM alerts WHERE priority = 'low'")
    low_count = cursor.fetchone()[0]
    
    # Get oldest and newest alerts
    cursor.execute("SELECT timestamp FROM alerts ORDER BY timestamp ASC LIMIT 1")
    oldest_result = cursor.fetchone()
    oldest = oldest_result[0] if oldest_result else None
    
    cursor.execute("SELECT timestamp FROM alerts ORDER BY timestamp DESC LIMIT 1")
    newest_result = cursor.fetchone()
    newest = newest_result[0] if newest_result else None
    
    # Calculate total alerts
    total_alerts = high_count + medium_count + low_count
    
    # Print stats
    stats = {
        "alerts_generated": {
            "high": high_count,
            "medium": medium_count,
            "low": low_count
        },
        "alerts_active": total_alerts,
        "oldest_alert": oldest,
        "newest_alert": newest,
        "timestamp": datetime.now().isoformat(),
        "false_positives_prevented": 0
    }
    
    logger.info(f"Database stats: {stats}")
    return stats

def main():
    """Main function to generate test alerts"""
    logger.info("Generating test alerts for the dashboard (simplified version)")
    
    # Initialize database
    conn = initialize_db()
    
    try:
        # Create test alerts
        create_test_alerts(conn)
        
        # Update and print stats
        stats = update_stats(conn)
        
        logger.info("Test alerts have been generated successfully!")
        logger.info("You can now refresh the dashboard to see the data")
        
    finally:
        conn.close()

if __name__ == "__main__":
    main()
