"""
Default configuration for the RemediationManager.

This file defines the default remediation rules and action configurations.
"""

# Default remediation rules
DEFAULT_REMEDIATION_RULES = [
    {
        "name": "OFAC High-Risk",
        "description": "Critical action for OFAC sanctions list matches",
        "actions": ["email_alert", "logging", "access_revocation"],
        "conditions": {
            "breach_type": "ofac",
            "confidence": "high"
        },
        "severity": 10
    },
    {
        "name": "BIS High-Risk",
        "description": "Critical action for BIS entity list matches",
        "actions": ["email_alert", "logging", "access_revocation"],
        "conditions": {
            "breach_type": "bis",
            "confidence": "high" 
        },
        "severity": 9
    },
    {
        "name": "OFAC Medium-Risk",
        "description": "Moderate action for potential OFAC matches",
        "actions": ["email_alert", "logging"],
        "conditions": {
            "breach_type": "ofac",
            "confidence": "medium"
        },
        "severity": 7
    },
    {
        "name": "BIS Medium-Risk",
        "description": "Moderate action for potential BIS matches",
        "actions": ["email_alert", "logging"],
        "conditions": {
            "breach_type": "bis", 
            "confidence": "medium"
        },
        "severity": 6
    },
    {
        "name": "OFAC Low-Risk",
        "description": "Minimal action for low-confidence OFAC matches",
        "actions": ["logging"],
        "conditions": {
            "breach_type": "ofac",
            "confidence": "low"
        },
        "severity": 4
    },
    {
        "name": "BIS Low-Risk",
        "description": "Minimal action for low-confidence BIS matches",
        "actions": ["logging"],
        "conditions": {
            "breach_type": "bis",
            "confidence": "low" 
        },
        "severity": 3
    },
    {
        "name": "Generic Compliance Breach",
        "description": "Default action for any compliance breach",
        "actions": ["logging"],
        "conditions": {},  # Empty means match all
        "severity": 1
    }
]

# Default email configuration (test mode enabled by default)
DEFAULT_EMAIL_CONFIG = {
    "test_mode": True,  # Don't actually send emails in test mode
    "smtp_server": "smtp.example.com",
    "smtp_port": 587,
    "use_ssl": False,
    "username": "",
    "password": "",
    "sender": "compliance-alerts@example.com",
    "default_recipients": ["compliance-team@example.com"]
}

# Default logging configuration
DEFAULT_LOGGING_CONFIG = {
    "log_dir": "logs",
    "log_file": "compliance_audit.log"
}

# Default quarantine configuration
DEFAULT_QUARANTINE_CONFIG = {
    "quarantine_dir": "quarantine"
}

# Default access revocation configuration
DEFAULT_ACCESS_REVOCATION_CONFIG = {
    "revocation_log": "logs/access_revocations.log"
}

# Complete default configuration
DEFAULT_REMEDIATION_CONFIG = {
    "rules": DEFAULT_REMEDIATION_RULES,
    "email_alert": DEFAULT_EMAIL_CONFIG,
    "logging": DEFAULT_LOGGING_CONFIG,
    "quarantine": DEFAULT_QUARANTINE_CONFIG,
    "access_revocation": DEFAULT_ACCESS_REVOCATION_CONFIG
}
