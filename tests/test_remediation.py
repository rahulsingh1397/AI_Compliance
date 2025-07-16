"""
Tests for the Remediation system in AIComplianceMonitoring.
"""

import sys
import os
import unittest
import json
import tempfile
import shutil
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime

# Add the project root to the path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Redirect all output files to the tests/test_results directory
TEST_RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'test_results')
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

from AIComplianceMonitoring.agents.remediation.manager import RemediationManager, RemediationRule
from AIComplianceMonitoring.agents.remediation.actions import (
    RemediationAction, EmailAlertAction, LoggingAction, 
    QuarantineAction, AccessRevocationAction
)
from AIComplianceMonitoring.agents.remediation.integration import (
    RemediationIntegration, RemediationIntegrationConfig
)


class TestRemediationSystem(unittest.TestCase):
    """Test the remediation system components."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp(dir=TEST_RESULTS_DIR)
        self.logs_dir = os.path.join(self.temp_dir, 'logs')
        self.quarantine_dir = os.path.join(self.temp_dir, 'quarantine')
        
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.quarantine_dir, exist_ok=True)
        
        # Test config with test mode email
        self.test_config = {
            'rules': [
                {
                    'name': 'Test OFAC Rule',
                    'description': 'Test rule for OFAC breaches',
                    'actions': ['email_alert', 'logging'],
                    'conditions': {'breach_type': 'ofac'},
                    'severity': 10
                },
                {
                    'name': 'Test BIS Rule',
                    'description': 'Test rule for BIS breaches',
                    'actions': ['logging', 'quarantine'],
                    'conditions': {'breach_type': 'bis'},
                    'severity': 8
                },
                {
                    'name': 'Test Default Rule',
                    'description': 'Default rule for any breach',
                    'actions': ['logging'],
                    'conditions': {},
                    'severity': 1
                }
            ],
            'email_alert': {
                'test_mode': True,
                'smtp_server': 'localhost',
                'smtp_port': 25,
                'sender': 'test@example.com',
                'default_recipients': ['compliance@example.com']
            },
            'logging': {
                'log_dir': self.logs_dir,
                'log_file': 'test_audit.log'
            },
            'quarantine': {
                'quarantine_dir': self.quarantine_dir
            },
            'access_revocation': {
                'revocation_log': os.path.join(self.logs_dir, 'revocations.log')
            }
        }
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_remediation_rule_matching(self):
        """Test the rule matching logic."""
        # Create test rules
        rule1 = RemediationRule(
            name="Test Rule 1",
            description="Test rule for specific breach type",
            actions=["email_alert"],
            conditions={"breach_type": "ofac", "confidence": "high"},
            severity=10
        )
        
        rule2 = RemediationRule(
            name="Test Rule 2",
            description="Test rule with multiple condition options",
            actions=["logging"],
            conditions={"breach_type": ["ofac", "bis"]},
            severity=5
        )
        
        # Test matching
        self.assertTrue(rule1.matches({
            "breach_type": "ofac", 
            "confidence": "high"
        }))
        
        self.assertFalse(rule1.matches({
            "breach_type": "ofac", 
            "confidence": "medium"
        }))
        
        self.assertTrue(rule2.matches({
            "breach_type": "ofac"
        }))
        
        self.assertTrue(rule2.matches({
            "breach_type": "bis"
        }))
        
        self.assertFalse(rule2.matches({
            "breach_type": "other"
        }))
        
    def test_remediation_manager_initialization(self):
        """Test that the remediation manager initializes correctly."""
        manager = RemediationManager(self.test_config)
        
        # Check that rules were loaded
        self.assertEqual(len(manager.rules), 3)
        
        # Check rule names
        rule_names = [rule.name for rule in manager.rules]
        self.assertIn('Test OFAC Rule', rule_names)
        self.assertIn('Test BIS Rule', rule_names)
        self.assertIn('Test Default Rule', rule_names)
        
    def test_action_registration(self):
        """Test action handler registration."""
        manager = RemediationManager(self.test_config)
        
        # Register mock actions
        action1 = MagicMock(return_value={"status": "success"})
        action2 = MagicMock(return_value={"status": "success"})
        
        manager.register_action("test_action_1", action1)
        manager.register_action("test_action_2", action2)
        
        self.assertIn("test_action_1", manager.actions)
        self.assertIn("test_action_2", manager.actions)
        
    def test_breach_handling(self):
        """Test handling of compliance breaches."""
        manager = RemediationManager(self.test_config)
        
        # Register mock actions
        email_action = MagicMock(return_value={"status": "success"})
        logging_action = MagicMock(return_value={"status": "success"})
        
        manager.register_action("email_alert", email_action)
        manager.register_action("logging", logging_action)
        
        # Test OFAC breach
        breach_data = {
            "breach_type": "ofac",
            "user": "test_user",
            "resource": "test_resource",
            "confidence": "high"
        }
        
        results = manager.handle_breach(breach_data)
        
        # Verify actions were called
        email_action.assert_called_once()
        logging_action.assert_called_once()
        
        # Check results
        self.assertEqual(len(results), 2)
        action_names = [r["action"] for r in results]
        self.assertIn("email_alert", action_names)
        self.assertIn("logging", action_names)
        
    def test_email_alert_action(self):
        """Test the email alert action in test mode."""
        email_config = self.test_config["email_alert"]
        action = EmailAlertAction(email_config)
        
        # Test breach data
        breach_data = {
            "breach_type": "ofac",
            "user": "suspicious_user",
            "resource": "sensitive_document.pdf",
            "timestamp": datetime.now().isoformat()
        }
        
        # Test rule data
        rule_data = {
            "name": "OFAC High Risk",
            "severity": 10
        }
        
        # Execute action in test mode
        result = action.execute(breach_data, rule_data)
        
        # Verify result
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["test_mode"])
        self.assertEqual(result["recipients"], email_config["default_recipients"])
        
        # Check that the email was stored
        self.assertEqual(len(action.sent_emails), 1)
        self.assertIn("COMPLIANCE ALERT", action.sent_emails[0]["subject"])
        
    def test_logging_action(self):
        """Test the logging action."""
        logging_config = self.test_config["logging"]
        action = LoggingAction(logging_config)
        
        # Test breach data
        breach_data = {
            "breach_type": "bis",
            "user": "test_user",
            "resource": "test_resource",
            "timestamp": datetime.now().isoformat()
        }
        
        # Test rule data
        rule_data = {
            "name": "BIS Medium Risk",
            "severity": 7
        }
        
        # Execute action
        result = action.execute(breach_data, rule_data)
        
        # Verify result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["log_path"], os.path.join(logging_config["log_dir"], logging_config["log_file"]))
        
        # Check that the log file exists and contains the breach data
        log_file_path = result["log_path"]
        self.assertTrue(os.path.exists(log_file_path))
        
        # Read the log file and verify it contains the breach data
        with open(log_file_path, 'r') as f:
            log_content = f.read()
            self.assertIn("BIS Medium Risk", log_content)
            self.assertIn("test_user", log_content)
            self.assertIn("test_resource", log_content)
            
    def test_integration_with_monitor(self):
        """Test the integration with the monitoring system."""
        # Create the integration module with test config
        integration_config = RemediationIntegrationConfig(
            enabled=True,
            config=self.test_config
        )
        integration = RemediationIntegration(config=integration_config)
        
        # Mock the remediation manager's handle_breach method
        integration.remediation_manager.handle_breach = MagicMock(
            return_value=[{"action": "logging", "status": "success"}]
        )
        
        # Test breach from compliance checker
        breach_data = {
            "breach_type": "ofac",
            "user": "sanctioned_entity",
            "resource": "sensitive_file.pdf",
            "match_score": 0.95
        }
        
        # Handle the breach
        results = integration.handle_compliance_breach(breach_data)
        
        # Verify the breach was handled
        integration.remediation_manager.handle_breach.assert_called_once()
        
        # Check that the breach was transformed correctly
        call_args = integration.remediation_manager.handle_breach.call_args[0][0]
        self.assertEqual(call_args["breach_type"], "ofac")
        self.assertEqual(call_args["user"], "sanctioned_entity")
        self.assertEqual(call_args["confidence"], "high")  # High confidence due to 0.95 match score
        
        # Check stats
        stats = integration.get_stats()
        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["status"], "active")
    
    def test_remediation_disabled(self):
        """Test behavior when remediation is disabled."""
        # Create the integration with remediation disabled
        integration_config = RemediationIntegrationConfig(
            enabled=False,
            config=self.test_config
        )
        integration = RemediationIntegration(config=integration_config)
        
        # Verify that the remediation manager wasn't initialized
        self.assertIsNone(integration.remediation_manager)
        
        # Test handling a breach
        breach_data = {
            "breach_type": "ofac",
            "user": "test_user",
            "resource": "test_resource"
        }
        
        # Should return empty results when disabled
        results = integration.handle_compliance_breach(breach_data)
        self.assertEqual(results, [])
        
        # Check stats when disabled
        stats = integration.get_stats()
        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["status"], "disabled")


if __name__ == "__main__":
    unittest.main()
