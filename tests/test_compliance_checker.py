import unittest
import pandas as pd
import logging
import os
import sys

# Add project root to path
# Add project root to allow imports from AIComplianceMonitoring
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AIComplianceMonitoring.agents.monitoring.compliance_checker import ComplianceChecker
from AIComplianceMonitoring.agents.monitoring.agent import MonitoringAgent, MonitoringAgentConfig

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestComplianceChecker(unittest.TestCase):
    """
    Test suite for the ComplianceChecker and its integration with the MonitoringAgent.
    """

    def setUp(self):
        """Set up the test environment before each test."""
        logger.info("Setting up test case...")
        self.config = {}
        self.compliance_checker = ComplianceChecker(self.config)

        # Mock data for testing
        self.mock_logs = pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-10-26 10:00:00', '2023-10-26 10:01:00', '2023-10-26 10:02:00', '2023-10-26 10:03:00']),
            'user': ['normal_user', 'Evil Corp', 'another_user', 'third_user'],
            'resource': ['server-1', 'sensitive-db', 'Questionable Tech', 'server-2'],
            'action': ['login', 'access', 'transfer', 'logout'],
            'message': ['User logged in.', 'Accessed sensitive data.', 'Data transfer initiated.', 'User logged out.']
        })

    def test_ofac_breach_detection(self):
        """Test that OFAC breaches are correctly detected."""
        logger.info("Running test: test_ofac_breach_detection")
        result_df = self.compliance_checker.check_compliance(self.mock_logs.copy())
        self.assertTrue('ofac_match' in result_df.columns)
        self.assertTrue(result_df.loc[1, 'ofac_match'])
        self.assertFalse(result_df.loc[0, 'ofac_match'])
        logger.info("OFAC breach detection test passed.")

    def test_bis_breach_detection(self):
        """Test that BIS breaches are correctly detected."""
        logger.info("Running test: test_bis_breach_detection")
        result_df = self.compliance_checker.check_compliance(self.mock_logs.copy())
        self.assertTrue('bis_match' in result_df.columns)
        self.assertTrue(result_df.loc[2, 'bis_match'])
        self.assertFalse(result_df.loc[0, 'bis_match'])
        logger.info("BIS breach detection test passed.")

    def test_compliance_breach_flag(self):
        """Test that the overall compliance_breach flag is set correctly."""
        logger.info("Running test: test_compliance_breach_flag")
        result_df = self.compliance_checker.check_compliance(self.mock_logs.copy())
        self.assertTrue('compliance_breach' in result_df.columns)
        self.assertTrue(result_df.loc[1, 'compliance_breach'])
        self.assertTrue(result_df.loc[2, 'compliance_breach'])
        self.assertFalse(result_df.loc[0, 'compliance_breach'])
        self.assertFalse(result_df.loc[3, 'compliance_breach'])
        logger.info("Compliance breach flag test passed.")

    def test_no_breaches(self):
        """Test that no breaches are detected in clean data."""
        logger.info("Running test: test_no_breaches")
        clean_logs = pd.DataFrame({
            'user': ['user1', 'user2'],
            'resource': ['server1', 'server2']
        })
        result_df = self.compliance_checker.check_compliance(clean_logs)
        self.assertFalse(result_df['compliance_breach'].any())
        logger.info("No breaches test passed.")

class MockModule:
    """A generic mock module for testing the MonitoringAgent."""
    def __init__(self, *args, **kwargs):
        pass
    def get_stats(self):
        return {}

class MockLogIngestionModule(MockModule):
    def ingest_logs(self, source_type, source_config):
        return TestComplianceChecker.static_mock_logs

    def initialize_connections(self):
        pass

class MockAnomalyDetectionModule(MockModule):
    def detect_anomalies(self, logs_df):
        # Return an empty DataFrame to isolate compliance alerts
        return pd.DataFrame()

    def initialize_models(self):
        pass

class MockAlertModule(MockModule):
    def __init__(self, config):
        self.alerts = []

    def generate_alerts(self, alerts_df, source_type, source_config):
        self.alerts.append(alerts_df)
        return {'alert_count': len(alerts_df)}

    def initialize_alert_db(self):
        pass

class TestAgentIntegration(unittest.TestCase):
    """
    Test the integration of the ComplianceChecker with the MonitoringAgent.
    """
    def setUp(self):
        """Set up a MonitoringAgent with mocked dependencies."""
        logger.info("Setting up agent integration test case...")
        config = MonitoringAgentConfig()
        TestComplianceChecker.static_mock_logs = pd.DataFrame({
            'user': ['normal_user', 'Evil Corp'],
            'resource': ['server-1', 'sensitive-db']
        })

        self.alert_module = MockAlertModule(config)
        self.agent = MonitoringAgent(
            config=config,
            log_ingestion_module=MockLogIngestionModule(),
            anomaly_detection_module=MockAnomalyDetectionModule(),
            alert_module=self.alert_module,
            compliance_checker_module=ComplianceChecker(config)
        )

    def test_compliance_breach_creates_alert(self):
        """Verify that a compliance breach results in a high-priority alert."""
        logger.info("Running test: test_compliance_breach_creates_alert")
        self.agent._monitor_source('mock_source', {'name': 'test'})
        
        self.assertEqual(len(self.alert_module.alerts), 1)
        alerts_df = self.alert_module.alerts[0]
        self.assertEqual(len(alerts_df), 1)
        
        alert = alerts_df.iloc[0]
        self.assertEqual(alert['anomaly_model'], 'compliance_checker')
        self.assertEqual(alert['anomaly_score'], 1.0)
        self.assertEqual(alert['user'], 'Evil Corp')
        logger.info("Compliance breach alert test passed.")

if __name__ == '__main__':
    unittest.main()
