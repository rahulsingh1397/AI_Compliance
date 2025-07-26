import json
import os
from datetime import datetime, timedelta
import random

# --- Configuration ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'AIComplianceMonitoring', 'instance')
OUTPUT_FILENAME = 'latest_report.json'

# --- Mock Data Generation ---

def generate_mock_data():
    """Generates a comprehensive mock report with realistic data."""
    now = datetime.now()

    key_metrics = {
        'sensitive_files': random.randint(50, 200),
        'total_scans': random.randint(100, 500),
        'risk_level': random.choice(['Low', 'Medium', 'High']),
        'last_scan': (now - timedelta(hours=random.randint(1, 24))).isoformat()
    }

    sensitive_data_types = {
        'labels': ['PII', 'Financial', 'Health', 'Proprietary'],
        'data': [random.randint(100, 500) for _ in range(4)]
    }

    compliance_status = {
        'labels': ['GDPR', 'HIPAA', 'CCPA', 'PCI-DSS'],
        'data': [random.randint(80, 100) for _ in range(4)]
    }

    recent_alerts = [
        {
            'id': i,
            'description': random.choice([
                'Unencrypted PII discovered in S3 bucket',
                'Anomalous access to financial records',
                'Data retention policy violation for GDPR',
                'New sensitive file detected in public share'
            ]),
            'severity': random.choice(['Low', 'Medium', 'High']),
            'timestamp': (now - timedelta(days=random.randint(1, 10))).strftime('%Y-%m-%d %H:%M:%S'),
            'source': random.choice(['AWS S3', 'SharePoint', 'On-Prem Server']),
            'status': random.choice(['New', 'In Progress', 'Resolved'])
        } for i in range(1, 6)
    ]

    sensitive_files_data = [
        {
            'name': f'document_{i}.pdf',
            'path': f'/data/department_{random.choice(["finance", "hr", "legal"])}/',
            'source': random.choice(['SharePoint', 'Google Drive', 'Local Network']),
            'risk_level': random.choice(['Low', 'Medium', 'High']),
            'last_accessed': (now - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
            'size': f'{random.uniform(0.5, 20.0):.1f} MB'
        } for i in range(1, 15)
    ]

    return {
        'key_metrics': key_metrics,
        'sensitive_data_types': sensitive_data_types,
        'compliance_status': compliance_status,
        'recent_alerts': recent_alerts,
        'sensitive_files': sensitive_files_data
    }

# --- Main Execution ---

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    report_data = generate_mock_data()
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=4)

    print(f'Successfully generated report at: {output_path}')
