"""
Privacy Preserving Agent Module

This module implements privacy-preserving techniques for AI compliance monitoring,
including Zero-Knowledge Machine Learning (ZKML) and Federated Learning components.
"""

__version__ = "0.1.0"

# Import core components
from .data_protection import DataProtectionManager, ProtectedData, ProtectionLevel
from .federated_learning import FederatedLearningManager, ModelUpdate
from .federated_client import FederatedClient, ClientConfig, TrainingMetrics
from .zkml_manager import ZKMLManager, ZKProof
from .audit_log import AuditLogger, AuditEvent, EventType
from .anonymization import AdvancedAnonymizer, AnonymizationConfig, AnonymizationMethod

__all__ = [
    'DataProtectionManager',
    'ProtectedData', 
    'ProtectionLevel',
    'FederatedLearningManager',
    'ModelUpdate',
    'FederatedClient',
    'ClientConfig',
    'TrainingMetrics',
    'ZKMLManager',
    'ZKProof',
    'AuditLogger',
    'AuditEvent',
    'EventType'
]
