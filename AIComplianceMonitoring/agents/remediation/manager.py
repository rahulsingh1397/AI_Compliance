"""
RemediationManager for orchestrating automated responses to compliance breaches.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class RemediationRule:
    """
    A rule that defines when to trigger remediation actions.
    """
    name: str
    description: str
    # List of actions to execute when this rule triggers
    actions: List[str] = field(default_factory=list)
    # Conditions that must be met (e.g., breach_type == 'ofac')
    conditions: Dict[str, Any] = field(default_factory=dict)
    # Severity level (higher means more severe)
    severity: int = 1
    
    def matches(self, breach_data: Dict[str, Any]) -> bool:
        """Check if this rule matches the breach data."""
        for key, expected_value in self.conditions.items():
            if key not in breach_data:
                return False
            
            actual_value = breach_data[key]
            
            # Handle various comparison types
            if isinstance(expected_value, list):
                if actual_value not in expected_value:
                    return False
            elif actual_value != expected_value:
                return False
                
        return True


class RemediationManager:
    """
    Orchestrates automated remediation responses to compliance breaches.
    
    This manager:
    1. Receives breach notifications
    2. Evaluates which rules match
    3. Executes the appropriate actions
    4. Maintains an audit trail
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the remediation manager.
        
        Args:
            config: Configuration dictionary with settings like rules and actions.
        """
        self.config = config
        self.rules: List[RemediationRule] = []
        self.actions: Dict[str, Callable] = {}
        self._breach_history: List[Dict[str, Any]] = []
        
        # Load rules from config
        self._load_rules()
        
        logger.info(f"RemediationManager initialized with {len(self.rules)} rules.")
        
    def _load_rules(self):
        """Load remediation rules from the configuration."""
        rules_config = self.config.get('rules', [])
        for rule_config in rules_config:
            rule = RemediationRule(
                name=rule_config.get('name', 'Unnamed Rule'),
                description=rule_config.get('description', ''),
                actions=rule_config.get('actions', []),
                conditions=rule_config.get('conditions', {}),
                severity=rule_config.get('severity', 1)
            )
            self.rules.append(rule)
    
    def register_action(self, action_name: str, action_handler: Callable):
        """
        Register an action handler with the manager.
        
        Args:
            action_name: Unique identifier for the action
            action_handler: Callable that implements the action
        """
        if action_name in self.actions:
            logger.warning(f"Overwriting existing action handler for '{action_name}'")
        
        self.actions[action_name] = action_handler
        logger.debug(f"Registered action handler for '{action_name}'")
    
    def handle_breach(self, breach_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a compliance breach and execute appropriate remediation actions.
        
        Args:
            breach_data: Dictionary containing breach details
            
        Returns:
            List of executed actions and their results
        """
        if not breach_data:
            logger.warning("Received empty breach data, skipping remediation")
            return []
        
        # Add timestamp to breach data
        breach_data['timestamp'] = pd.Timestamp.now().isoformat()
        
        # Store in history
        self._breach_history.append(breach_data.copy())
        
        # Find matching rules
        matching_rules = []
        for rule in self.rules:
            if rule.matches(breach_data):
                matching_rules.append(rule)
        
        if not matching_rules:
            logger.info("No remediation rules matched for this breach")
            return []
        
        # Sort rules by severity (highest first)
        matching_rules.sort(key=lambda r: r.severity, reverse=True)
        
        # Execute actions for matching rules
        results = []
        executed_actions: Set[str] = set()
        
        for rule in matching_rules:
            logger.info(f"Applying remediation rule: {rule.name}")
            
            for action_name in rule.actions:
                # Skip if already executed (to prevent duplicate actions)
                if action_name in executed_actions:
                    continue
                
                if action_name not in self.actions:
                    logger.error(f"Action '{action_name}' not found")
                    continue
                
                try:
                    logger.debug(f"Executing action: {action_name}")
                    result = self.actions[action_name](breach_data, rule)
                    results.append({
                        'action': action_name,
                        'rule': rule.name,
                        'status': 'success',
                        'result': result
                    })
                    executed_actions.add(action_name)
                except Exception as e:
                    logger.exception(f"Action '{action_name}' failed")
                    results.append({
                        'action': action_name,
                        'rule': rule.name,
                        'status': 'error',
                        'error': str(e)
                    })
        
        logger.info(f"Executed {len(results)} remediation actions")
        return results
    
    def get_breach_history(self) -> List[Dict[str, Any]]:
        """Get the history of processed breaches."""
        return self._breach_history.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the remediation manager."""
        return {
            'rules_count': len(self.rules),
            'actions_count': len(self.actions),
            'breaches_handled': len(self._breach_history)
        }
