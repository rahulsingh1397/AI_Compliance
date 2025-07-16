"""
Remediation module for automated compliance breach handling.
"""

from .manager import RemediationManager
from .actions import (
    RemediationAction,
    EmailAlertAction,
    LoggingAction,
    QuarantineAction,
    AccessRevocationAction
)
