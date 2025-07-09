"""
Privacy Preserving Agent Module

This module implements privacy-preserving techniques for AI compliance monitoring,
including Zero-Knowledge Machine Learning (ZKML) and Federated Learning components.
"""

__version__ = "0.1.0"

# Import core components
from .zkml_manager import ZKMLManager
from .federated_learning import FederatedLearningManager
from .audit_log import SecureAuditLog
from .data_protection import DataProtectionManager

__all__ = [
    'ZKMLManager',
    'FederatedLearningManager',
    'SecureAuditLog',
    'DataProtectionManager'
]
