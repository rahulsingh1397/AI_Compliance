import unittest
import os
import sqlite3
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
import time

# Add project root to path to allow absolute imports
import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from AIComplianceMonitoring.agents.monitoring.alert_module import AlertModule

class TestAlertModule(unittest.TestCase):

    def setUp(self):
        """Set up a temporary database and test data for each test."""
        self.test_dir = Path('./test_temp_data')
        self.db_dir = self.test_dir / 'db'
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_dir / 'test_alerts.db'

        self.config = {
            'database_directory': self.db_dir,
            'high_sev_threshold': 0.8,
            'medium_sev_threshold': 0.5
        }

        # Mock RL Feedback Manager
        self.mock_feedback_manager = MagicMock()

        # Initialize AlertModule
        self.alert_module = AlertModule(self.config, self.mock_feedback_manager)
        self.alert_module.db_path = self.db_path # Override db path for testing
        self.alert_module.initialize_alert_db()

        # Sample anomaly data
        self.anomalies_df = pd.DataFrame([
            {'anomaly_score': 0.9, 'type': 'unauthorized_access', 'user_id': 'user1'},
            {'anomaly_score': 0.6, 'type': 'unusual_transfer', 'data_size': '10GB'},
            {'anomaly_score': 0.4, 'type': 'suspicious_behavior', 'user_id': 'user2'},
            {'anomaly_score': 0.95, 'type': 'unauthorized_access', 'user_id': 'user3'}
        ])

    def tearDown(self):
        """Clean up the temporary database after each test."""
        if self.alert_module.db_connection:
            self.alert_module.db_connection.close()
        if self.db_path.exists():
            os.remove(self.db_path)
        if self.test_dir.exists():
            import shutil
            shutil.rmtree(self.test_dir)

    def test_01_database_initialization(self):
        """Test if the database and alerts table are created correctly."""
        self.assertTrue(self.db_path.exists())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alerts';")
        self.assertIsNotNone(cursor.fetchone(), "'alerts' table should exist.")
        conn.close()

    def test_02_generate_alerts(self):
        """Test the generation of alerts from anomalies."""
        self.mock_feedback_manager.process_anomaly.return_value = {'action': 'alert', 'is_true_positive': True}
        
        stats = self.alert_module.generate_alerts(self.anomalies_df, 'test_source', {})
        
        self.assertEqual(stats['alerts_created'], 4)
        self.assertEqual(stats['priority_counts']['high'], 2)
        self.assertEqual(stats['priority_counts']['medium'], 1)
        self.assertEqual(stats['priority_counts']['low'], 1)

        alerts = self.alert_module.get_alerts()
        self.assertEqual(len(alerts), 4)
        self.assertEqual(alerts[0]['priority'], 'high') # Ordered by timestamp DESC

    def test_03_get_alerts_with_filter(self):
        """Test retrieving alerts with a priority filter."""
        self.mock_feedback_manager.process_anomaly.return_value = {'action': 'alert'}
        self.alert_module.generate_alerts(self.anomalies_df, 'test_source', {})
        
        high_priority_alerts = self.alert_module.get_alerts(priority='high')
        self.assertEqual(len(high_priority_alerts), 2)

        medium_priority_alerts = self.alert_module.get_alerts(priority='medium')
        self.assertEqual(len(medium_priority_alerts), 1)

    def test_04_alert_deduplication(self):
        """Test that duplicate alerts are not generated within the time window."""
        self.mock_feedback_manager.process_anomaly.return_value = {'action': 'alert'}
        
        # First run
        self.alert_module.generate_alerts(self.anomalies_df, 'test_source', {})
        self.assertEqual(len(self.alert_module.get_alerts()), 4)

        # Second run immediately after - should generate 0 new alerts
        stats = self.alert_module.generate_alerts(self.anomalies_df, 'test_source', {})
        self.assertEqual(stats['alerts_created'], 0)
        self.assertEqual(len(self.alert_module.get_alerts()), 4)

    def test_05_rl_feedback_suppression(self):
        """Test that alerts are suppressed based on RL feedback."""
        # Mock feedback to suppress one anomaly
        def side_effect(anomaly_data):
            if anomaly_data['user_id'] == 'user1':
                return {'action': 'suppress'}
            return {'action': 'alert'}
        self.mock_feedback_manager.process_anomaly.side_effect = side_effect

        stats = self.alert_module.generate_alerts(self.anomalies_df, 'test_source', {})
        
        self.assertEqual(stats['alerts_created'], 3)
        self.assertEqual(stats['false_positives_filtered'], 1)
        self.assertEqual(len(self.alert_module.get_alerts()), 3)

    def test_06_get_stats(self):
        """Test the statistics retrieval method."""
        self.mock_feedback_manager.process_anomaly.return_value = {'action': 'alert'}
        self.alert_module.generate_alerts(self.anomalies_df, 'test_source', {})

        stats = self.alert_module.get_stats()

        self.assertEqual(stats['alerts_generated']['high'], 2)
        self.assertEqual(stats['alerts_generated']['medium'], 1)
        self.assertEqual(stats['alerts_generated']['low'], 1)
        self.assertEqual(stats['alerts_active'], 4)
        self.assertIsNotNone(stats['newest_alert'])

if __name__ == '__main__':
    unittest.main()
