"""
Tests for the Hybrid Monitoring System in AIComplianceMonitoring.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime
import tempfile
import shutil

# Add the project root to the path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Redirect all output files to the tests/test_results directory
TEST_RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'test_results')
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

from AIComplianceMonitoring.agents.monitoring.hybrid_anomaly_detector import HybridAnomalyDetector
from AIComplianceMonitoring.agents.remediation.ai_remediation_manager import AIEnhancedRemediationManager, AIResponseEngine
from AIComplianceMonitoring.agents.monitoring.hybrid_monitor import HybridMonitoringAgent, HybridMonitoringConfig


class TestHybridAnomalyDetector(unittest.TestCase):
    """Test the hybrid anomaly detection system."""
    
    def setUp(self):
        """Set up test environment."""
        self.detector = HybridAnomalyDetector(lstm_units=32, ae_units=16)
        
        # Create test data
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        # Normal data
        self.normal_data = pd.DataFrame(
            np.random.normal(0, 1, (n_samples, n_features)),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Anomalous data
        self.anomalous_data = pd.DataFrame(
            np.random.normal(5, 2, (20, n_features)),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Combined dataset for training
        self.training_data = pd.concat([self.normal_data, self.anomalous_data])
    
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.lstm_units, 32)
        self.assertEqual(self.detector.ae_units, 16)
        self.assertFalse(self.detector.is_trained)
    
    @unittest.skip("Skipping TensorFlow-dependent test")
    def test_model_training(self):
        """Test model training with synthetic data."""
        # Requires TensorFlow, so we'll skip for basic testing
        self.detector.train(self.normal_data, epochs=2, batch_size=16)
        self.assertTrue(self.detector.is_trained)
        self.assertIsNotNone(self.detector.lstm_model)
        self.assertIsNotNone(self.detector.ae_model)
        self.assertIsNotNone(self.detector.combined_model)
    
    @unittest.skip("Skipping TensorFlow-dependent test")
    def test_anomaly_detection(self):
        """Test anomaly detection on test data."""
        # Mock training since we're skipping actual training
        self.detector.is_trained = True
        self.detector.lstm_model = MagicMock()
        self.detector.ae_model = MagicMock()
        self.detector.combined_model = MagicMock()
        
        # Mock model predictions
        self.detector.lstm_model.predict.return_value = np.array([[0.1], [0.9]])
        self.detector.ae_model.predict.return_value = np.array([[0.1, 0.2], [0.8, 0.9]])
        self.detector.combined_model.predict.return_value = np.array([[0.2], [0.8]])
        
        # Create test data
        test_data = pd.DataFrame({
            'feature_0': [0, 5],
            'feature_1': [0, 5]
        })
        
        # Detect anomalies
        results = self.detector.detect_anomalies(test_data)
        
        # Check results structure
        self.assertIn('anomalies', results)
        self.assertIn('scores', results)
        self.assertIn('confidence', results)
        
        # Check individual model results
        self.assertIn('lstm', results['anomalies'])
        self.assertIn('ae', results['anomalies'])
        self.assertIn('combined', results['anomalies'])


class TestAIRemediationManager(unittest.TestCase):
    """Test the AI-enhanced remediation manager."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'rules': [
                {
                    'name': 'Test Rule',
                    'description': 'Test rule for breaches',
                    'actions': ['logging'],
                    'conditions': {'breach_type': 'test'},
                    'severity': 5
                }
            ]
        }
        self.ai_manager = AIEnhancedRemediationManager(self.config)
    
    def test_initialization(self):
        """Test that the AI remediation manager initializes correctly."""
        self.assertIsNotNone(self.ai_manager)
        self.assertIsNotNone(self.ai_manager.ai_engine)
        self.assertFalse(self.ai_manager.is_trained)
    
    def test_breach_handling_untrained(self):
        """Test handling of breaches when AI is not trained."""
        # Mock AI engine methods
        self.ai_manager.ai_engine.predict_breach_severity = MagicMock()
        self.ai_manager.ai_engine.predict_breach_severity.side_effect = ValueError("Model not trained")
        
        # Mock rule-based actions
        self.ai_manager._get_rule_based_actions = MagicMock()
        self.ai_manager._get_rule_based_actions.return_value = ['logging']
        
        # Test breach data
        breach_data = {
            'breach_type': 'test',
            'user': 'test_user',
            'resource': 'test_resource'
        }
        
        # Should raise ValueError when not trained
        with self.assertRaises(ValueError):
            self.ai_manager.handle_breach(breach_data)
    
    def test_confidence_calculation(self):
        """Test confidence calculation for recommendations."""
        ai_actions = ['email_alert', 'logging']
        rule_actions = ['logging', 'quarantine']
        
        # Calculate confidence
        confidence = self.ai_manager._calculate_confidence(ai_actions, rule_actions)
        
        # Should be 0.5 as there's 1 matching action out of 2 in the larger list
        self.assertEqual(confidence, 0.5)


class TestHybridMonitoringAgent(unittest.TestCase):
    """Test the hybrid monitoring agent."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp(dir=TEST_RESULTS_DIR)
        
        # Create config
        self.config = HybridMonitoringConfig(
            monitoring_interval=60,
            log_sources=[],
            alert_config={},
            remediation_config={
                "enabled": True,
                "config": {}
            },
            use_hybrid_anomaly_detection=True,
            use_ai_remediation=True,
            fallback_to_traditional=True
        )
        
        # Mock components
        self.mock_log_ingestion = MagicMock()
        self.mock_alert_module = MagicMock()
        self.mock_compliance_module = MagicMock()
        self.mock_remediation_module = MagicMock()
        
        # Initialize agent with mocks
        with patch('AIComplianceMonitoring.agents.monitoring.agent.LogIngestionModule', return_value=self.mock_log_ingestion), \
             patch('AIComplianceMonitoring.agents.monitoring.agent.AlertModule', return_value=self.mock_alert_module), \
             patch('AIComplianceMonitoring.agents.monitoring.agent.ComplianceChecker', return_value=self.mock_compliance_module), \
             patch('AIComplianceMonitoring.agents.monitoring.agent.RemediationIntegration', return_value=self.mock_remediation_module):
            self.agent = HybridMonitoringAgent(self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test that the hybrid monitoring agent initializes correctly."""
        self.assertIsNotNone(self.agent)
        self.assertIsNotNone(self.agent.hybrid_config)
        self.assertEqual(self.agent.hybrid_config.use_hybrid_anomaly_detection, True)
        self.assertEqual(self.agent.hybrid_config.use_ai_remediation, True)
    
    def test_monitor_source(self):
        """Test monitoring a source."""
        # Mock log ingestion to return test logs
        test_logs = pd.DataFrame({
            'user': ['user1', 'user2'],
            'resource': ['res1', 'res2'],
            'action': ['read', 'write'],
            'timestamp': [datetime.now().isoformat(), datetime.now().isoformat()]
        })
        self.mock_log_ingestion.ingest_logs.return_value = test_logs
        
        # Mock anomaly detection
        with patch.object(self.agent, '_detect_anomalies') as mock_detect:
            mock_detect.return_value = (test_logs, [
                {
                    'source': 'test',
                    'user': 'user1',
                    'resource': 'res1',
                    'anomaly_score': 0.9,
                    'detection_method': 'hybrid_ai'
                }
            ])
            
            # Mock compliance checking
            self.mock_compliance_module.check_compliance.return_value = {
                'breaches': [
                    {
                        'breach_type': 'ofac',
                        'user': 'user1',
                        'resource': 'res1',
                        'match_score': 0.95
                    }
                ]
            }
            
            # Mock remediation handling
            with patch.object(self.agent, '_handle_compliance_breach') as mock_handle:
                mock_handle.return_value = [
                    {
                        'action': 'logging',
                        'status': 'success',
                        'breach_type': 'ofac',
                        'method': 'ai_enhanced'
                    }
                ]
                
                # Call monitor_source
                result = self.agent._monitor_source('test_source', {'name': 'test'})
                
                # Verify result
                self.assertEqual(result['status'], 'success')
                self.assertEqual(result['anomaly_count'], 2)
                self.assertEqual(result['alert_count'], 1)
                self.assertEqual(result['using_hybrid_detection'], True)
                self.assertEqual(result['using_ai_remediation'], True)
    
    def test_get_stats(self):
        """Test getting agent stats."""
        stats = self.agent.get_stats()
        
        # Verify AI stats are included
        self.assertIn('ai_components', stats)
        self.assertIn('hybrid_anomaly_detection', stats['ai_components'])
        self.assertIn('ai_remediation', stats['ai_components'])


if __name__ == '__main__':
    unittest.main()
