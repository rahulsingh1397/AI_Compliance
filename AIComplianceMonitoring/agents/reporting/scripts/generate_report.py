"""
Mock Compliance Report Generator

This script generates sample compliance reports for testing and demonstration purposes.
"""

import json
import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import random

def setup_logging(log_level=logging.INFO):
    """Configure logging for the script."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate mock compliance reports')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for reports (default: instance/)')
    parser.add_argument('--filename', type=str, default='latest_report.json',
                       help='Output filename (default: latest_report.json)')
    parser.add_argument('--alerts-count', type=int, default=5,
                       help='Number of mock alerts to generate (default: 5)')
    parser.add_argument('--files-count', type=int, default=15,
                       help='Number of mock sensitive files to generate (default: 15)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    return parser.parse_args()

logger = logging.getLogger(__name__)

def generate_key_metrics() -> Dict[str, Any]:
    """Generate mock key metrics data."""
    now = datetime.now()
    return {
        'sensitive_files': random.randint(50, 200),
        'total_scans': random.randint(100, 500),
        'risk_level': random.choice(['Low', 'Medium', 'High']),
        'last_scan': (now - timedelta(hours=random.randint(1, 24))).isoformat(),
        'compliance_score': random.randint(75, 98),
        'data_sources_monitored': random.randint(5, 25)
    }

def generate_sensitive_data_types() -> Dict[str, List]:
    """Generate mock sensitive data types distribution."""
    return {
        'labels': ['PII', 'Financial', 'Health', 'Proprietary', 'Legal'],
        'data': [random.randint(50, 300) for _ in range(5)]
    }

def generate_compliance_status() -> Dict[str, List]:
    """Generate mock compliance status data."""
    return {
        'labels': ['GDPR', 'HIPAA', 'CCPA', 'PCI-DSS', 'SOX'],
        'data': [random.randint(80, 100) for _ in range(5)]
    }

def generate_recent_alerts(count: int = 5) -> List[Dict[str, Any]]:
    """Generate mock recent alerts data."""
    now = datetime.now()
    alert_templates = [
        'Unencrypted PII discovered in S3 bucket',
        'Anomalous access to financial records',
        'Data retention policy violation for GDPR',
        'New sensitive file detected in public share',
        'Unauthorized data export attempt detected',
        'Failed compliance scan on database server',
        'Suspicious file access pattern identified',
        'Data classification mismatch found'
    ]
    
    sources = ['AWS S3', 'SharePoint', 'On-Prem Server', 'Azure Blob', 'Google Drive', 'Database']
    severities = ['Low', 'Medium', 'High', 'Critical']
    statuses = ['New', 'In Progress', 'Resolved', 'Acknowledged']
    
    return [
        {
            'id': i,
            'description': random.choice(alert_templates),
            'severity': random.choice(severities),
            'timestamp': (now - timedelta(days=random.randint(1, 10))).strftime('%Y-%m-%d %H:%M:%S'),
            'source': random.choice(sources),
            'status': random.choice(statuses),
            'affected_records': random.randint(1, 1000) if random.random() > 0.3 else None
        } for i in range(1, count + 1)
    ]

def generate_sensitive_files_data(count: int = 15) -> List[Dict[str, Any]]:
    """Generate mock sensitive files data."""
    now = datetime.now()
    departments = ['finance', 'hr', 'legal', 'marketing', 'operations', 'it']
    sources = ['SharePoint', 'Google Drive', 'Local Network', 'AWS S3', 'OneDrive']
    risk_levels = ['Low', 'Medium', 'High', 'Critical']
    file_types = ['.pdf', '.docx', '.xlsx', '.csv', '.txt']
    
    return [
        {
            'name': f'document_{i}{random.choice(file_types)}',
            'path': f'/data/department_{random.choice(departments)}/',
            'source': random.choice(sources),
            'risk_level': random.choice(risk_levels),
            'last_accessed': (now - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
            'size': f'{random.uniform(0.1, 50.0):.1f} MB',
            'data_types': random.sample(['PII', 'Financial', 'Health', 'Proprietary'], k=random.randint(1, 3))
        } for i in range(1, count + 1)
    ]

def generate_mock_data(alerts_count: int = 5, files_count: int = 15) -> Dict[str, Any]:
    """Generate a comprehensive mock report with realistic data."""
    logger.info(f"Generating mock report with {alerts_count} alerts and {files_count} files")
    
    return {
        'report_metadata': {
            'generated_at': datetime.now().isoformat(),
            'report_type': 'compliance_summary',
            'version': '1.0'
        },
        'key_metrics': generate_key_metrics(),
        'sensitive_data_types': generate_sensitive_data_types(),
        'compliance_status': generate_compliance_status(),
        'recent_alerts': generate_recent_alerts(alerts_count),
        'sensitive_files': generate_sensitive_files_data(files_count)
    }
def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    try:
        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            # Default to instance directory relative to project root
            project_root = Path(__file__).parent.parent.parent.parent
            output_dir = project_root / 'instance'
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Generate mock data
        report_data = generate_mock_data(
            alerts_count=args.alerts_count,
            files_count=args.files_count
        )
        
        # Write report to file
        output_path = output_dir / args.filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Successfully generated mock report at: {output_path}")
        logger.info(f"Report contains:")
        logger.info(f"  - {len(report_data['recent_alerts'])} alerts")
        logger.info(f"  - {len(report_data['sensitive_files'])} sensitive files")
        logger.info(f"  - {report_data['key_metrics']['sensitive_files']} total sensitive files tracked")
        logger.info(f"  - Risk level: {report_data['key_metrics']['risk_level']}")
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
