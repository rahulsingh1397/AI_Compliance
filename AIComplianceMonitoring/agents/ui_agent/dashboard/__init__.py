

import os
import json
import threading
from datetime import datetime
from pathlib import Path

from flask import Blueprint, render_template, session, current_app, jsonify
from flask_login import login_required, current_user
from flask_babel import _

# Use absolute imports from the project root
from AIComplianceMonitoring.agents.ui_agent.extensions import db
from AIComplianceMonitoring.agents.ui_agent.models import ScanHistory
from AIComplianceMonitoring.agents.reporting.reporting_agent import ReportingAgent

# For DataDiscoveryAgent, we'll import it directly when needed since it's outside ui_agent

# For DataDiscoveryAgent, we'll import it directly when needed since it's outside ui_agent

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard')
@login_required
def index():
    lang = session.get('lang', current_user.language if current_user else 'en')
    
    return render_template(
        'dashboard.html', 
        lang=lang,
        _=_  # Explicitly pass translation function
    )

@dashboard_bp.route('/risk')
@login_required
def risk():
    lang = session.get('lang', current_user.language if current_user else 'en')
    return render_template('dashboard/risk.html', lang=lang)

@dashboard_bp.route('/alerts')
@login_required
def alerts():
    lang = session.get('lang', current_user.language if current_user else 'en')
    return render_template('dashboard/alerts.html', lang=lang)

@dashboard_bp.route('/files')
@login_required
def sensitive_files():
    lang = session.get('lang', current_user.language if current_user else 'en')
    report_path = os.path.join(current_app.instance_path, 'latest_report.json')
    files_data = []
    try:
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)
                files_data = report.get('sensitive_files', [])
    except Exception as e:
        current_app.logger.error(f"Could not load or parse sensitive files data: {e}")

    return render_template('dashboard/sensitive_files.html', title=_('Sensitive Files'), files=files_data, lang=lang, _=_)

@dashboard_bp.route('/file-details/<path:file_path>')
@login_required
def sensitive_file_details(file_path):
    lang = session.get('lang', current_user.language if current_user else 'en')
    report_path = os.path.join(current_app.instance_path, 'latest_report.json')
    file_details = None
    try:
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)
                for f in report.get('sensitive_files', []):
                    if f['path'] == file_path:
                        file_details = f
                        break
    except Exception as e:
        current_app.logger.error(f"Could not load or parse sensitive files data: {e}")

    if not file_details:
        return "File not found", 404

    return render_template('dashboard/sensitive_file_details.html', title=_('File Details'), file=file_details, lang=lang, _=_)

@dashboard_bp.route('/risk-overview')
@login_required
def risk_overview():
    lang = session.get('lang', current_user.language if current_user else 'en')
    # Fetch live data using the new utility function
    from AIComplianceMonitoring.agents.ui_agent.utils import fetch_live_compliance_data
    live_data = fetch_live_compliance_data()

    # Transform live data into the format expected by the template
    key_metrics = live_data.get('key_metrics', {})
    risk_level = key_metrics.get('risk_level', 'Low')

    # --- Granular Risk Scoring Logic ---

    def calculate_score(value, thresholds):
        if value >= thresholds['high']:
            return 85
        if value >= thresholds['medium']:
            return 60
        if value > 0:
            return 30
        return 0

    # 1. Data Exposure Score
    sensitive_files_count = key_metrics.get('sensitive_files', 0)
    data_exposure_score = calculate_score(sensitive_files_count, {'medium': 10, 'high': 50})

    # 2. Access Control Score
    recent_alerts_count = len(live_data.get('recent_alerts', []))
    access_control_score = calculate_score(recent_alerts_count, {'medium': 5, 'high': 15})

    # 3. Compliance Gaps Score - only calculate if we have actual compliance data
    compliance_statuses = live_data.get('compliance_status', {})
    if compliance_statuses:
        avg_compliance = sum(compliance_statuses.values()) / len(compliance_statuses)
        compliance_gap = 100 - avg_compliance
        compliance_gaps_score = calculate_score(compliance_gap, {'medium': 15, 'high': 30})
    else:
        compliance_gaps_score = 0  # No score without actual compliance data

    # 4. Overall Risk Score (only from categories with actual data)
    category_scores = [score for score in [data_exposure_score, access_control_score, compliance_gaps_score] if score > 0]
    risk_score = int(sum(category_scores) / len(category_scores)) if category_scores else 0

    # Determine overall risk level from the average score
    if risk_score >= 75:
        risk_level = 'High'
    elif risk_score >= 45:
        risk_level = 'Medium'
    else:
        risk_level = 'Low'

    # Fetch historical scan data for the trend chart
    scan_history = ScanHistory.query.order_by(ScanHistory.start_time.asc()).limit(12).all()
    
    risk_trend_labels = [scan.start_time.strftime('%b %d') for scan in scan_history]
    risk_trend_data = [scan.sensitive_files_found for scan in scan_history] # Using alert count as a proxy for risk score

    # Generate description based only on actual data
    sensitive_files = key_metrics.get('sensitive_files', 0)
    alerts_count = len(live_data.get('recent_alerts', []))
    
    if sensitive_files == 0 and alerts_count == 0:
        description = "No risk assessment available - no scan data found."
    else:
        description = f"Risk assessment based on {sensitive_files} sensitive files found in latest scan."
        if alerts_count > 0:
            description += f" {alerts_count} recent alerts detected."

    # Create the final data structure for the template
    risk_data = {
        'overall_risk': {
            'score': risk_score,
            'level': risk_level,
            'description': description
        },
        'risk_trend': {
            'labels': risk_trend_labels,
            'data': risk_trend_data
        },
        'risk_categories': [
            {
                'name': 'Data Exposure',
                'score': data_exposure_score,
                'level': 'High' if data_exposure_score >= 75 else ('Medium' if data_exposure_score >= 45 else 'Low'),
                'description': 'Risks related to sensitive data being accessible or stored insecurely.',
                'factors': [f"{key_metrics.get('sensitive_files', 0)} sensitive files found"]
            },
            {
                'name': 'Access Control',
                'score': access_control_score,
                'level': 'High' if access_control_score >= 75 else ('Medium' if access_control_score >= 45 else 'Low'),
                'description': 'Risks from improper user permissions and access patterns.',
                'factors': [f"{len(live_data.get('recent_alerts', []))} recent alerts"]
            },
            {
                'name': 'Compliance Gaps',
                'score': compliance_gaps_score,
                'level': 'High' if compliance_gaps_score >= 75 else ('Medium' if compliance_gaps_score >= 45 else 'Low'),
                'description': 'Risks associated with failing to meet regulatory requirements.',
                'factors': [f"{framework.upper()}: {score}%" for framework, score in live_data.get('compliance_status', {}).items()] if live_data.get('compliance_status') else ['No compliance data available']
            }
        ]
    }

    return render_template('dashboard/risk_overview.html', title=_('Risk Overview'), **risk_data, lang=lang, _=_)

@dashboard_bp.route('/scans')
@login_required
def scans():
    
    lang = session.get('lang', current_user.language if current_user else 'en')
    
    # Get scan history from database, ordered by most recent first
    scan_records = ScanHistory.query.order_by(ScanHistory.start_time.desc()).all()
    
    # Transform database records to template format
    scans_data = []
    for scan in scan_records:
        scans_data.append({
            'id': scan.scan_id,
            'start_time': scan.start_time,  # Pass datetime object
            'end_time': scan.end_time,      # Pass datetime object
            'status': scan.status,
            'files_scanned': scan.files_scanned,
            'alerts_found': scan.sensitive_files_found  # Using sensitive_files_found as alerts_found
        })
    
    return render_template('dashboard/scans.html', title=_('Scan History'), scans=scans_data, lang=lang, _=_)

def run_scan_in_background(app, scan_id):
    
    with app.app_context():
        # Create scan history record
        scan_record = ScanHistory(scan_id=scan_id, status='Running')
        db.session.add(scan_record)
        db.session.commit()
        
        try:
            from AIComplianceMonitoring.agents.data_discovery.data_discovery_agent import DataDiscoveryAgent, AgentConfig

            app.logger.info("Background Scan: Initializing DataDiscoveryAgent...")
            config = AgentConfig()
            with DataDiscoveryAgent(config=config) as agent:
                app.logger.info("Background Scan: DataDiscoveryAgent initialized.")
                project_root = Path(app.root_path).parent.parent.parent
                data_dir = project_root / "AIComplianceMonitoring" / "data"
                app.logger.info(f"Background Scan: Starting scan of directory: {data_dir}")

                file_extensions = list(agent.config.default_file_extensions) + ['.xlsx']
                results = agent.batch_scan_files(
                    directory_path=str(data_dir),
                    file_extensions=file_extensions,
                    max_workers=agent.config.max_workers
                )

                app.logger.info("Background Scan: Scan complete. Transforming results for UI.")

                # Transform results into the format expected by the UI
                sensitive_files_for_ui = []
                for file_detail in results.get('sensitive_files_details', []):
                    risk_level = "Low"
                    if len(file_detail.get('scan_details', {}).get('sensitive_data_types', [])) > 5:
                        risk_level = "High"
                    elif len(file_detail.get('scan_details', {}).get('sensitive_data_types', [])) > 2:
                        risk_level = "Medium"
                    
                    sensitive_files_for_ui.append({
                        'name': Path(file_detail['file_path']).name,
                        'path': file_detail['file_path'],
                        'size': file_detail['file_size'],
                        'last_accessed': file_detail['last_modified'],
                        'risk_level': risk_level,
                        'source': 'Local Scan',
                        'scan_details': file_detail.get('scan_details', {})
                    })

                report_data = {
                    'scan_id': f"scan_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    'timestamp': datetime.now().isoformat(),
                    'total_files_scanned': results.get('total_files_scanned', 0),
                    'sensitive_files_found': results.get('sensitive_files_found', 0),
                    'sensitive_files': sensitive_files_for_ui
                }

                report_path = os.path.join(app.instance_path, 'latest_report.json')
                with open(report_path, 'w') as f:
                    json.dump(report_data, f, indent=4)
                
                # Update scan record with completion
                scan_record.end_time = datetime.now()
                scan_record.status = 'Completed'
                scan_record.files_scanned = results.get('total_files_scanned', 0)
                scan_record.sensitive_files_found = results.get('sensitive_files_found', 0)
                scan_record.results_path = report_path
                db.session.commit()
                
                app.logger.info(f"Background Scan: Results saved to {report_path}")

        except Exception as e:
            app.logger.error(f"Error during background scan: {e}", exc_info=True)
            # Update scan record with failure
            scan_record.end_time = datetime.now()
            scan_record.status = 'Failed'
            db.session.commit()

@dashboard_bp.route('/scans/run', methods=['POST'])
@login_required
def run_scan():
    app = current_app._get_current_object()
    scan_id = f"scan_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    scan_thread = threading.Thread(target=run_scan_in_background, args=(app, scan_id))
    scan_thread.start()
    return jsonify({'status': 'success', 'message': 'Scan initiated successfully. Results will be available shortly.', 'scan_id': scan_id})


@dashboard_bp.route('/reports/create', methods=['POST'])
@login_required
def create_report():
    """Creates a new compliance report from the latest scan data."""
    app = current_app._get_current_object()
    
    # Path to the latest scan results
    latest_scan_path = os.path.join(app.instance_path, 'latest_report.json')
    
    if not os.path.exists(latest_scan_path):
        return jsonify({'status': 'error', 'message': 'No scan data found. Please run a scan first.'}), 404

    try:
        with open(latest_scan_path, 'r') as f:
            scan_data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        app.logger.error(f"Failed to read or parse scan data: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to load scan data.'}), 500

    # Directory to save reports
    report_dir = os.path.join(app.instance_path, 'reports')
    
    # Create and run the reporting agent
    try:
        agent = ReportingAgent(report_dir=report_dir)
        report_path = agent.create_report(scan_results=scan_data)
        
        # Make the report path relative for the user
        relative_report_path = os.path.relpath(report_path, app.instance_path)
        
        app.logger.info(f"Report created successfully at {report_path}")
        return jsonify({
            'status': 'success', 
            'message': 'Report created successfully.',
            'report_path': relative_report_path
        })
    except Exception as e:
        app.logger.error(f"Failed to create report: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'An unexpected error occurred while creating the report.'}), 500

def format_file_size(size_bytes):
    """Converts file size in bytes to a human-readable format."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def run_discovery_scan_job():
    """Function to run the data discovery scan in the background."""
    try:
        from AIComplianceMonitoring.agents.data_discovery.data_discovery_agent import DataDiscoveryAgent
    except ImportError:
        current_app.logger.error("Could not import DataDiscoveryAgent. Using mock agent as fallback.")
        class MockDataDiscoveryAgent:
            def scan_directory(self, directory):
                current_app.logger.info(f"Mock scanning directory: {directory}")
                return {'status': 'completed', 'files_scanned': 150, 'sensitive_files': 12}
        DataDiscoveryAgent = MockDataDiscoveryAgent

    with current_app.app_context():
        try:
            current_app.logger.info("Starting background data discovery scan...")
            with DataDiscoveryAgent() as agent:
                scan_directory = os.path.abspath(os.path.join(current_app.root_path, '..', '..', 'data'))
                current_app.logger.info(f"Scanning directory: {scan_directory}")
                
                if not os.path.isdir(scan_directory):
                    current_app.logger.error(f"Scan directory not found: {scan_directory}")
                    return

                result = agent.batch_scan_files(directory_path=scan_directory)
                current_app.logger.info(f"Data discovery scan completed.")

                # Process and save the results for the UI
                if result and result.get('successful_scans') > 0:
                    report_path = os.path.join(current_app.instance_path, 'latest_report.json')
                    sensitive_files_data = []
                    for file_path, scan_details in result.get('scan_details', {}).items():
                        if scan_details.get('sensitive_data_found'):
                            risk_score = scan_details.get('risk_score', 0)
                            if risk_score > 0.7:
                                risk_level = 'High'
                            elif risk_score > 0.4:
                                risk_level = 'Medium'
                            else:
                                risk_level = 'Low'
                            
                            try:
                                file_stat = os.stat(file_path)
                                last_accessed_formatted = datetime.fromtimestamp(file_stat.st_atime).strftime('%Y-%m-%d %H:%M:%S')
                                size_formatted = format_file_size(file_stat.st_size)

                                sensitive_files_data.append({
                                    'name': os.path.basename(file_path),
                                    'path': file_path,
                                    'source': 'Local Filesystem',
                                    'risk_level': risk_level,
                                    'last_accessed': last_accessed_formatted,
                                    'size': size_formatted,
                                    'scan_details': scan_details
                                })
                            except FileNotFoundError:
                                current_app.logger.warning(f"Could not stat file, it may have been moved or deleted: {file_path}")

                    report = {'sensitive_files': sensitive_files_data}
                    with open(report_path, 'w') as f:
                        json.dump(report, f, indent=4)
                    current_app.logger.info(f"Scan report saved to {report_path} with {len(sensitive_files_data)} sensitive files.")

        except Exception as e:
            current_app.logger.error(f"Error during background data discovery scan: {e}", exc_info=True)

# Removed duplicate run_scan function

@dashboard_bp.route('/data-types')
@login_required
def data_types():
    lang = session.get('lang', current_user.language if current_user else 'en')
    mock_data = {
        'summary': {
            'total_types': 8,
            'most_common': 'Credit Card Number',
            'highest_risk': 'Social Security Number'
        },
        'data_types': [
            {'name': 'Credit Card Number', 'risk_level': 'High', 'files_found': 1250, 'regulation': 'PCI-DSS'},
            {'name': 'Social Security Number', 'risk_level': 'High', 'files_found': 830, 'regulation': 'PII'},
            {'name': 'Email Address', 'risk_level': 'Medium', 'files_found': 15200, 'regulation': 'PII'},
            {'name': 'Phone Number', 'risk_level': 'Medium', 'files_found': 9800, 'regulation': 'PII'},
            {'name': 'Driver\'s License', 'risk_level': 'Medium', 'files_found': 450, 'regulation': 'PII'},
            {'name': 'Medical Record Number', 'risk_level': 'High', 'files_found': 210, 'regulation': 'HIPAA'},
            {'name': 'IP Address', 'risk_level': 'Low', 'files_found': 58000, 'regulation': 'CCPA'},
            {'name': 'Bank Account Number', 'risk_level': 'High', 'files_found': 670, 'regulation': 'PCI-DSS'}
        ]
    }
    return render_template('dashboard/data_types.html', title=_('Sensitive Data Types'), **mock_data, lang=lang, _=_)

@dashboard_bp.route('/compliance-status')
@login_required
def compliance_status():
    lang = session.get('lang', current_user.language if current_user else 'en')
    mock_regulations = [
        {'name': 'GDPR', 'region': 'European Union', 'status': 'At Risk', 'controls_assessed': 120, 'controls_passing': 110, 'compliance_percent': 91},
        {'name': 'HIPAA', 'region': 'United States', 'status': 'Compliant', 'controls_assessed': 85, 'controls_passing': 83, 'compliance_percent': 97},
        {'name': 'CCPA', 'region': 'California (USA)', 'status': 'Compliant', 'controls_assessed': 50, 'controls_passing': 50, 'compliance_percent': 100},
        {'name': 'PCI-DSS', 'region': 'Global', 'status': 'Non-Compliant', 'controls_assessed': 250, 'controls_passing': 210, 'compliance_percent': 84},
        {'name': 'SOX', 'region': 'United States', 'status': 'Compliant', 'controls_assessed': 75, 'controls_passing': 75, 'compliance_percent': 100},
    ]
    return render_template('dashboard/compliance_status.html', title=_('Compliance Status'), regulations=mock_regulations, lang=lang, _=_)

@dashboard_bp.route('/alert/<int:id>')
@login_required
def alert_details(id):
    # In a real app, you'd fetch alert details from the database based on the ID
    lang = session.get('lang', current_user.language if current_user else 'en')
    # For now, we'll just find the mock alert to display something.
    mock_alerts = [
        {'id': 1, 'date': '2023-06-10', 'message': 'Sensitive data detected in shared folder', 'severity': 'high', 'details': 'Detailed information about the sensitive data found in the shared folder...'},
        {'id': 2, 'date': '2023-06-09', 'message': 'Compliance check overdue for GDPR', 'severity': 'medium', 'details': 'The scheduled GDPR compliance check has not been completed on time.'},
        {'id': 3, 'date': '2023-06-08', 'message': 'New data source added', 'severity': 'low', 'details': 'A new data source has been connected and is pending initial scan.'}
    ]
    alert = next((alert for alert in mock_alerts if alert['id'] == id), None)
    
    if alert is None:
        # A simple way to handle not found, could redirect to dashboard with a flash message
        return "Alert not found", 404
        
    return render_template('dashboard/alert_details.html', alert=alert, lang=lang)
