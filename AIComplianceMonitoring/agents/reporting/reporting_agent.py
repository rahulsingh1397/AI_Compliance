import json
import os
from datetime import datetime

class ReportingAgent:
    def __init__(self, report_dir):
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)

    def create_report(self, scan_results):
        """Generates a report from scan results and saves it."""
        report_id = f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        report_path = os.path.join(self.report_dir, f"{report_id}.json")

        report_data = {
            'report_id': report_id,
            'generation_date': datetime.now().isoformat(),
            'scan_summary': {
                'total_files_scanned': scan_results.get('total_files_scanned', 0),
                'sensitive_files_found': scan_results.get('sensitive_files_found', 0),
            },
            'sensitive_files': scan_results.get('sensitive_files', [])
        }

        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=4)

        return report_path
