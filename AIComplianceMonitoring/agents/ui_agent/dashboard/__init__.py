from flask import Blueprint, render_template, session
from flask_login import login_required, current_user
from flask_babel import _
import json

from ..models import db

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
