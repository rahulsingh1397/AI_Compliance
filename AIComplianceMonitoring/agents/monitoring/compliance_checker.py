import logging
import json
from typing import Dict, Any, Optional
from flask import Flask

# Assuming models and db are accessible via the Flask app context

logger = logging.getLogger(__name__)

class ComplianceChecker:
    """
    Handles compliance checks by fetching the latest scan results from the database.
    """
    def __init__(self, app: Optional[Flask] = None):
        """
        Initialize the compliance checker.

        Args:
            app: The Flask application instance to access the database.
        """
        self.app = app
        if self.app:
            logger.info("ComplianceChecker initialized with Flask app context.")
        else:
            logger.warning("ComplianceChecker initialized without Flask app context. Stats will be unavailable.")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the compliance state from the last scan.
        """
        if not self.app:
            return {"error": "Flask app context not available."}

        with self.app.app_context():
            from AIComplianceMonitoring.agents.ui_agent.models import ScanHistory

            latest_scan = ScanHistory.query.order_by(ScanHistory.start_time.desc()).first()

            if not latest_scan:
                return {
                    'sensitive_files_found': 0,
                    'total_files_scanned': 0,
                    'risk_level': 'Low',
                    'last_scan_date': 'N/A',
                    'total_scans': ScanHistory.query.count(),
                    'pii_types_found': {}
                }

            risk_level = 'Low'
            if latest_scan.sensitive_files_found > 10:
                risk_level = 'High'
            elif latest_scan.sensitive_files_found > 3:
                risk_level = 'Medium'

            pii_types_found = {}
            if latest_scan.results and 'pii_types_found' in latest_scan.results:
                pii_types_found = latest_scan.results['pii_types_found']

            return {
                'sensitive_files_found': latest_scan.sensitive_files_found,
                'total_files_scanned': latest_scan.total_files_scanned,
                'risk_level': risk_level,
                'last_scan_date': latest_scan.start_time.isoformat() if latest_scan.start_time else 'N/A',
                'total_scans': ScanHistory.query.count(),
                'pii_types_found': pii_types_found
            }
