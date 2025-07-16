"""
Integration module for connecting the RemediationManager to the existing monitoring workflow.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field

from AIComplianceMonitoring.agents.monitoring.compliance_checker import ComplianceChecker
from AIComplianceMonitoring.agents.remediation.manager import RemediationManager
from AIComplianceMonitoring.agents.remediation.actions import (
    RemediationAction, EmailAlertAction, LoggingAction, 
    QuarantineAction, AccessRevocationAction
)
from AIComplianceMonitoring.agents.remediation.default_config import DEFAULT_REMEDIATION_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class RemediationIntegrationConfig:
    """Configuration for remediation integration."""
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=lambda: DEFAULT_REMEDIATION_CONFIG.copy())


class RemediationIntegration:
    """
    Integration class that connects the ComplianceChecker with the RemediationManager.
    
    This class:
    1. Subscribes to compliance breach events
    2. Transforms breach data into remediation-compatible format
    3. Triggers appropriate remediation actions
    4. Reports remediation results
    """
    
    def __init__(self, config: Optional[RemediationIntegrationConfig] = None):
        """
        Initialize the remediation integration.
        
        Args:
            config: Configuration for the integration
        """
        self.config = config or RemediationIntegrationConfig()
        
        if not self.config.enabled:
            logger.info("Remediation integration is disabled")
            self.remediation_manager = None
            return
        
        # Initialize the remediation manager
        self.remediation_manager = RemediationManager(self.config.config)
        
        # Register default action handlers
        self._register_default_actions()
        
        logger.info("Remediation integration initialized")
    
    def _register_default_actions(self):
        """Register the default action handlers with the remediation manager."""
        if not self.remediation_manager:
            return
            
        # Initialize and register email alerts
        email_config = self.config.config.get('email_alert', {})
        email_action = EmailAlertAction(email_config)
        self.remediation_manager.register_action('email_alert', email_action.execute)
        
        # Initialize and register logging action
        logging_config = self.config.config.get('logging', {})
        logging_action = LoggingAction(logging_config)
        self.remediation_manager.register_action('logging', logging_action.execute)
        
        # Initialize and register quarantine action
        quarantine_config = self.config.config.get('quarantine', {})
        quarantine_action = QuarantineAction(quarantine_config)
        self.remediation_manager.register_action('quarantine', quarantine_action.execute)
        
        # Initialize and register access revocation action
        access_config = self.config.config.get('access_revocation', {})
        access_action = AccessRevocationAction(access_config)
        self.remediation_manager.register_action('access_revocation', access_action.execute)
    
    def register_custom_action(self, action_name: str, action_handler: Callable):
        """
        Register a custom action handler with the remediation manager.
        
        Args:
            action_name: Unique identifier for the action
            action_handler: Callable that implements the action
        """
        if not self.remediation_manager:
            logger.warning("Remediation integration is disabled, cannot register action")
            return
            
        self.remediation_manager.register_action(action_name, action_handler)
        logger.info(f"Registered custom remediation action: {action_name}")
    
    def handle_compliance_breach(self, breach: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Handle a compliance breach by triggering appropriate remediation actions.
        
        Args:
            breach: Dictionary containing breach details from ComplianceChecker
            
        Returns:
            List of executed actions and their results
        """
        if not self.remediation_manager or not self.config.enabled:
            logger.info("Remediation is disabled, skipping breach handling")
            return []
            
        # Transform the breach data into remediation-compatible format
        remediation_data = self._transform_breach_data(breach)
        
        # Trigger remediation
        logger.info(f"Handling compliance breach: {breach.get('breach_type', 'unknown')}")
        results = self.remediation_manager.handle_breach(remediation_data)
        
        # Log a summary of actions taken
        action_summary = ', '.join([r['action'] for r in results if r['status'] == 'success'])
        if action_summary:
            logger.info(f"Remediation complete. Actions taken: {action_summary}")
        else:
            logger.warning("No remediation actions were successfully executed")
            
        return results
    
    def _transform_breach_data(self, breach: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform compliance breach data into the format expected by the remediation system.
        
        Args:
            breach: The breach data from ComplianceChecker
            
        Returns:
            Transformed data for the RemediationManager
        """
        # Default confidence level based on breach data
        confidence = "medium"  # Default confidence
        
        # Extract match score if available and determine confidence
        if 'match_score' in breach:
            match_score = breach['match_score']
            if match_score > 0.9:
                confidence = "high"
            elif match_score > 0.7:
                confidence = "medium"
            else:
                confidence = "low"
                
        # Create remediation-compatible data format
        return {
            'breach_type': breach.get('breach_type', 'unknown'),
            'user': breach.get('user', 'unknown'),
            'resource': breach.get('resource', 'unknown'),
            'resource_path': breach.get('resource_path', None),
            'confidence': confidence,
            'timestamp': breach.get('timestamp', pd.Timestamp.now().isoformat()),
            'details': breach.get('details', {}),
            'original_breach': breach  # Include the original breach data for reference
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the remediation integration."""
        if not self.remediation_manager:
            return {
                'enabled': False,
                'status': 'disabled'
            }
            
        manager_stats = self.remediation_manager.get_stats()
        return {
            'enabled': True,
            'status': 'active',
            'rules_count': manager_stats['rules_count'],
            'actions_count': manager_stats['actions_count'],
            'breaches_handled': manager_stats['breaches_handled']
        }
