"""
Secure Audit Log

This module provides a secure, tamper-evident audit logging system for tracking
privacy-preserving operations in the AI compliance monitoring system.
"""

import json
import hashlib
import hmac
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum

class AuditLogEntryType(str, Enum):
    """Types of audit log entries."""
    MODEL_TRAIN = "model_train"
    MODEL_PREDICT = "model_predict"
    DATA_ACCESS = "data_access"
    ZK_PROOF_GENERATED = "zk_proof_generated"
    ZK_PROOF_VERIFIED = "zk_proof_verified"
    FEDERATED_UPDATE = "federated_update"
    FEDERATED_AGGREGATION = "federated_aggregation"
    PRIVACY_CHECK = "privacy_check"
    SECURITY_EVENT = "security_event"

@dataclass
class AuditLogEntry:
    """Represents a single audit log entry."""
    entry_type: AuditLogEntryType
    timestamp: str
    user_id: str
    operation: str
    details: Dict[str, Any]
    previous_hash: Optional[str] = None
    signature: Optional[str] = None
    entry_id: str = field(default_factory=lambda: base64.urlsafe_b64encode(
        hashlib.sha256(datetime.utcnow().isoformat().encode()).digest()
    ).decode('utf-8')[:16])

    def to_dict(self) -> Dict[str, Any]:
        """Convert the audit log entry to a dictionary."""
        return {
            'entry_id': self.entry_id,
            'entry_type': self.entry_type.value,
            'timestamp': self.timestamp,
            'user_id': self.user_id,
            'operation': self.operation,
            'details': self.details,
            'previous_hash': self.previous_hash,
            'signature': self.signature
        }
    
    def to_json(self) -> str:
        """Convert the audit log entry to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

class SecureAuditLog:
    """
    A secure, tamper-evident audit logging system that maintains an immutable
    chain of log entries with cryptographic verification.
    """
    
    def __init__(self, secret_key: Optional[bytes] = None):
        """
        Initialize the secure audit log.
        
        Args:
            secret_key: Optional secret key for HMAC signing. If not provided,
                       a default key will be used (not recommended for production).
        """
        self.entries: List[AuditLogEntry] = []
        self.secret_key = secret_key or b'default-insecure-key-change-me'
        
    def add_entry(
        self,
        entry_type: AuditLogEntryType,
        user_id: str,
        operation: str,
        details: Dict[str, Any],
        timestamp: Optional[str] = None
    ) -> AuditLogEntry:
        """
        Add a new entry to the audit log.
        
        Args:
            entry_type: Type of the audit log entry
            user_id: ID of the user performing the action
            operation: Description of the operation
            details: Additional details about the operation
            timestamp: Optional timestamp (defaults to current time)
            
        Returns:
            The created AuditLogEntry
        """
        if not timestamp:
            timestamp = datetime.utcnow().isoformat()
            
        previous_hash = self._get_latest_hash()
        
        entry = AuditLogEntry(
            entry_type=entry_type,
            timestamp=timestamp,
            user_id=user_id,
            operation=operation,
            details=details,
            previous_hash=previous_hash
        )
        
        # Sign the entry
        entry.signature = self._sign_entry(entry)
        
        self.entries.append(entry)
        return entry
    
    def verify_log_integrity(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Verify the integrity of the entire audit log.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not self.entries:
            return True, []
            
        # Check first entry
        first_entry = self.entries[0]
        if first_entry.previous_hash is not None:
            issues.append({
                'entry_id': first_entry.entry_id,
                'issue': 'First entry should have no previous_hash',
                'severity': 'high'
            })
        
        # Verify signatures and hashes
        for i in range(1, len(self.entries)):
            current = self.entries[i]
            previous = self.entries[i-1]
            
            # Verify previous hash matches
            expected_previous_hash = self._calculate_hash(previous)
            if current.previous_hash != expected_previous_hash:
                issues.append({
                    'entry_id': current.entry_id,
                    'issue': f'Hash mismatch: expected {expected_previous_hash}, got {current.previous_hash}',
                    'severity': 'critical'
                })
            
            # Verify signature
            if not self._verify_signature(current):
                issues.append({
                    'entry_id': current.entry_id,
                    'issue': 'Invalid signature',
                    'severity': 'critical'
                })
        
        return len(issues) == 0, issues
    
    def _calculate_hash(self, entry: AuditLogEntry) -> str:
        """Calculate the hash of an entry."""
        data = {
            'entry_id': entry.entry_id,
            'entry_type': entry.entry_type.value,
            'timestamp': entry.timestamp,
            'user_id': entry.user_id,
            'operation': entry.operation,
            'details': entry.details,
            'previous_hash': entry.previous_hash
        }
        
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def _sign_entry(self, entry: AuditLogEntry) -> str:
        """Generate an HMAC signature for an entry."""
        data = self._calculate_hash(entry)
        signature = hmac.new(
            self.secret_key,
            data.encode(),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')
    
    def _verify_signature(self, entry: AuditLogEntry) -> bool:
        """Verify the HMAC signature of an entry."""
        if not entry.signature:
            return False
            
        expected_signature = self._sign_entry(entry)
        return hmac.compare_digest(entry.signature, expected_signature)
    
    def _get_latest_hash(self) -> Optional[str]:
        """Get the hash of the most recent entry, or None if empty."""
        if not self.entries:
            return None
        return self._calculate_hash(self.entries[-1])
    
    def export_log(self) -> str:
        """Export the audit log as a JSON string."""
        return json.dumps([entry.to_dict() for entry in self.entries], indent=2)
    
    @classmethod
    def import_log(cls, log_data: str, secret_key: Optional[bytes] = None) -> 'SecureAuditLog':
        """
        Import an audit log from a JSON string.
        
        Args:
            log_data: JSON string containing the audit log
            secret_key: Secret key for signature verification
            
        Returns:
            A new SecureAuditLog instance
        """
        log_entries = json.loads(log_data)
        audit_log = cls(secret_key=secret_key)
        
        for entry_data in log_entries:
            entry = AuditLogEntry(
                entry_type=AuditLogEntryType(entry_data['entry_type']),
                timestamp=entry_data['timestamp'],
                user_id=entry_data['user_id'],
                operation=entry_data['operation'],
                details=entry_data['details'],
                previous_hash=entry_data['previous_hash'],
                signature=entry_data['signature'],
                entry_id=entry_data['entry_id']
            )
            audit_log.entries.append(entry)
            
        return audit_log
