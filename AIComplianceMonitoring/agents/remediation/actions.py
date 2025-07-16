"""
Remediation action handlers for automated compliance breach responses.
"""

import logging
import smtplib
import json
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, Optional, Protocol, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class RemediationAction(ABC):
    """
    Abstract base class for all remediation actions.
    """
    
    @abstractmethod
    def execute(self, breach_data: Dict[str, Any], rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute this remediation action.
        
        Args:
            breach_data: The compliance breach data
            rule_data: Information about the rule that triggered this action
            
        Returns:
            Dictionary with results of the action
        """
        pass
        

class EmailAlertAction(RemediationAction):
    """
    Send an email alert about a compliance breach.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the email alerter.
        
        Args:
            config: Configuration with SMTP settings and recipient lists
        """
        self.config = config
        self.smtp_server = config.get('smtp_server', 'localhost')
        self.smtp_port = config.get('smtp_port', 25)
        self.use_ssl = config.get('use_ssl', False)
        self.username = config.get('username')
        self.password = config.get('password')
        self.sender = config.get('sender', 'compliance-alerts@example.com')
        
        # Default recipients if not specified per alert
        self.default_recipients = config.get('default_recipients', [])
        
        # For testing/development without sending actual emails
        self.test_mode = config.get('test_mode', False)
        self.sent_emails: List[Dict[str, Any]] = []
    
    def _format_email_body(self, breach_data: Dict[str, Any], rule_data: Dict[str, Any]) -> str:
        """Format an HTML email body with breach details."""
        timestamp = breach_data.get('timestamp', datetime.now().isoformat())
        user = breach_data.get('user', 'Unknown')
        resource = breach_data.get('resource', 'N/A')
        breach_type = breach_data.get('breach_type', 'compliance')
        
        # Create HTML content
        html = f"""
        <html>
            <head>
                <style>
                    .breach-alert {{ color: #d9534f; font-weight: bold; }}
                    .breach-details {{ background-color: #f8f9fa; padding: 10px; border: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                <h2 class="breach-alert">🚨 Compliance Breach Alert</h2>
                <p>A compliance breach was detected by the AI Compliance Monitoring system.</p>
                
                <div class="breach-details">
                    <p><strong>Time:</strong> {timestamp}</p>
                    <p><strong>Rule:</strong> {rule_data.get('name', 'Unknown')}</p>
                    <p><strong>Type:</strong> {breach_type}</p>
                    <p><strong>User:</strong> {user}</p>
                    <p><strong>Resource:</strong> {resource}</p>
                </div>
                
                <p>Please investigate this breach immediately and take appropriate action.</p>
                <p>This is an automated message. Do not reply.</p>
            </body>
        </html>
        """
        return html
    
    def execute(self, breach_data: Dict[str, Any], rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send an email alert about the breach.
        
        Args:
            breach_data: Data about the compliance breach
            rule_data: Information about the rule that triggered this action
            
        Returns:
            Dictionary with the result of the email sending attempt
        """
        # Determine recipients (rule-specific or default)
        recipients = breach_data.get('recipients', self.default_recipients)
        
        if not recipients:
            logger.warning("No recipients specified for email alert")
            return {'status': 'skipped', 'reason': 'no_recipients'}
            
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"COMPLIANCE ALERT: {breach_data.get('breach_type', 'Compliance')} Breach Detected"
        msg['From'] = self.sender
        msg['To'] = ', '.join(recipients)
        
        # Add HTML body
        html_body = self._format_email_body(breach_data, rule_data)
        msg.attach(MIMEText(html_body, 'html'))
        
        # In test mode, don't actually send emails
        if self.test_mode:
            logger.info(f"TEST MODE: Would send email to {recipients}")
            self.sent_emails.append({
                'to': recipients,
                'subject': msg['Subject'],
                'body': html_body,
                'time': datetime.now().isoformat()
            })
            return {
                'status': 'success',
                'test_mode': True,
                'recipients': recipients
            }
            
        # Send the actual email
        try:
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            else:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                
            if self.username and self.password:
                server.login(self.username, self.password)
                
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Sent compliance alert email to {len(recipients)} recipients")
            return {
                'status': 'success',
                'recipients': recipients
            }
        except Exception as e:
            logger.exception("Failed to send email alert")
            return {
                'status': 'error',
                'error': str(e),
                'recipients': recipients
            }


class LoggingAction(RemediationAction):
    """
    Write detailed breach information to a secure audit log.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the audit logger.
        
        Args:
            config: Configuration with log paths and formats
        """
        self.config = config
        self.log_dir = config.get('log_dir', 'logs')
        self.log_file = config.get('log_file', 'compliance_audit.log')
        
        # Create log directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.log_path = os.path.join(self.log_dir, self.log_file)
        
    def execute(self, breach_data: Dict[str, Any], rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log the breach information to the audit log.
        
        Args:
            breach_data: Data about the compliance breach
            rule_data: Information about the rule that triggered this action
            
        Returns:
            Dictionary with the result of the logging operation
        """
        log_entry = {
            'timestamp': breach_data.get('timestamp', datetime.now().isoformat()),
            'rule': rule_data.get('name', 'Unknown Rule'),
            'severity': rule_data.get('severity', 1),
            'breach_data': breach_data,
            'remediation_action': 'logging'
        }
        
        try:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
            logger.info(f"Breach logged to audit file: {self.log_path}")
            return {
                'status': 'success',
                'log_path': self.log_path
            }
        except Exception as e:
            logger.exception("Failed to write to audit log")
            return {
                'status': 'error',
                'error': str(e)
            }


class QuarantineAction(RemediationAction):
    """
    Quarantine a resource by moving it to a secure location and restricting access.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the quarantine handler.
        
        Args:
            config: Configuration with quarantine settings
        """
        self.config = config
        self.quarantine_dir = config.get('quarantine_dir', 'quarantine')
        
        # Create quarantine directory if it doesn't exist
        if not os.path.exists(self.quarantine_dir):
            os.makedirs(self.quarantine_dir)
            
    def execute(self, breach_data: Dict[str, Any], rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Quarantine the resource involved in the breach.
        
        Args:
            breach_data: Data about the compliance breach
            rule_data: Information about the rule that triggered this action
            
        Returns:
            Dictionary with the result of the quarantine operation
        """
        # Extract resource information
        resource_path = breach_data.get('resource_path')
        
        if not resource_path or not os.path.exists(resource_path):
            logger.warning(f"Resource not found for quarantine: {resource_path}")
            return {
                'status': 'skipped',
                'reason': 'resource_not_found',
                'resource': resource_path
            }
            
        try:
            # Generate unique quarantine filename to avoid collisions
            filename = os.path.basename(resource_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quarantine_filename = f"{timestamp}_{filename}"
            quarantine_path = os.path.join(self.quarantine_dir, quarantine_filename)
            
            # Create metadata file with breach information
            metadata_path = f"{quarantine_path}.meta.json"
            
            # In a real system, this would be an atomic move operation
            # Here we'll simulate with a copy (a real implementation should use shutil)
            logger.info(f"Quarantining {resource_path} -> {quarantine_path}")
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump({
                    'original_path': resource_path,
                    'quarantine_time': datetime.now().isoformat(),
                    'breach_data': breach_data,
                    'rule_data': {
                        'name': rule_data.get('name'),
                        'severity': rule_data.get('severity'),
                    }
                }, f, indent=2)
            
            # Return success (in a real implementation, this would actually move the file)
            return {
                'status': 'success',
                'original_path': resource_path,
                'quarantine_path': quarantine_path,
                'metadata_path': metadata_path
            }
        except Exception as e:
            logger.exception(f"Failed to quarantine resource: {resource_path}")
            return {
                'status': 'error',
                'error': str(e),
                'resource': resource_path
            }


class AccessRevocationAction(RemediationAction):
    """
    Revoke access permissions for a user involved in a compliance breach.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the access revocation handler.
        
        Args:
            config: Configuration with access control settings
        """
        self.config = config
        # In a real system, this would integrate with your IAM system
        self.revocation_log = config.get('revocation_log', 'access_revocations.log')
        
    def execute(self, breach_data: Dict[str, Any], rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Revoke access for the user involved in the breach.
        
        Args:
            breach_data: Data about the compliance breach
            rule_data: Information about the rule that triggered this action
            
        Returns:
            Dictionary with the result of the access revocation
        """
        user = breach_data.get('user')
        
        if not user:
            logger.warning("No user specified for access revocation")
            return {
                'status': 'skipped',
                'reason': 'no_user_specified'
            }
            
        try:
            # In a real system, this would call your IAM API
            # For demonstration, we'll just log the revocation
            
            revocation_entry = {
                'timestamp': datetime.now().isoformat(),
                'user': user,
                'reason': f"Compliance breach: {breach_data.get('breach_type', 'Unknown')}",
                'rule': rule_data.get('name', 'Unknown Rule'),
                'temporary': True
            }
            
            log_dir = os.path.dirname(self.revocation_log)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            with open(self.revocation_log, 'a') as f:
                f.write(json.dumps(revocation_entry) + '\n')
                
            logger.info(f"Access revoked for user: {user}")
            return {
                'status': 'success',
                'user': user,
                'temporary': True
            }
        except Exception as e:
            logger.exception(f"Failed to revoke access for user: {user}")
            return {
                'status': 'error',
                'error': str(e),
                'user': user
            }
