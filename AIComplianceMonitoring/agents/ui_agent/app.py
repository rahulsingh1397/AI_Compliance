import logging
import traceback
from datetime import timedelta

from flask import Flask, session, request, redirect, url_for
from flask_login import current_user
from flask_migrate import Migrate

# Use relative imports to fix module paths
from . import config
from .extensions import db, login_manager, babel # Import extensions from central file

# No top-level model imports or user_loader definitions here

# 1. Application Factory
def create_app():
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)
    app.config.from_object(config.Config)

    # 2. Initialize extensions with the app instance
    db.init_app(app)
    login_manager.init_app(app)
    babel.init_app(app, locale_selector=get_locale)
    Migrate(app, db, directory=app.config['MIGRATIONS_DIR'])

    # 3. Configure LoginManager and User Loader
    login_manager.login_view = 'auth.login'
    login_manager.session_protection = 'strong'

    @login_manager.user_loader
    def load_user(user_id):
        # Import model locally to avoid circular dependency
        from .models import User
        return User.query.get(int(user_id))

    # 4. Import and register blueprints with relative import paths
    from .main import main_bp
    from .auth import auth_bp
    from .dashboard import dashboard_bp
    from .settings import settings_bp
    from .api import api_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(api_bp, url_prefix='/api')

    # 5. Add root redirect
    @app.route('/', endpoint='index')
    def index_redirect():
        return redirect(url_for('dashboard.index'))
        
    # TEMPORARY: Direct test endpoint to debug dashboard data issues
    @app.route('/test_compliance_data')
    def test_compliance_data():
        """Direct test endpoint for compliance data without auth requirement."""
        from flask import jsonify
        import requests
        import logging
        import os
        import traceback
        import json
        from datetime import datetime
        
        log = logging.getLogger(__name__)
        print("\n\nCONSOLE: Test compliance data endpoint accessed")
        
        # Use mock data as fallback if API calls fail
        def get_mock_data():
            print("CONSOLE: Using mock data as fallback")
            return {
                'key_metrics': {
                    'sensitive_files': 42,
                    'total_scans': 150,
                    'risk_level': 'Medium',
                    'last_scan_date': '2025-06-25 15:30:00'
                },
                'sensitive_data_types': {
                    'High Priority': 3,
                    'Medium Priority': 12,
                    'Low Priority': 27
                },
                'compliance_status': {'GDPR': 95, 'CCPA': 88, 'HIPAA': 92},
                'recent_alerts': [
                    {'timestamp': '2025-06-25 15:30:00', 'message': 'Mock alert 1', 'severity': 'High'},
                    {'timestamp': '2025-06-25 14:25:00', 'message': 'Mock alert 2', 'severity': 'Medium'}
                ]
            }
        
        try:
            DEFAULT_API_URL = "http://127.0.0.1:5001"
            monitoring_api_url = os.environ.get('MONITORING_API_BASE_URL') or DEFAULT_API_URL
            print(f"CONSOLE: Using Monitoring API URL: {monitoring_api_url}")
            
            try:
                # Fetch stats and alerts from the Monitoring API
                print(f"CONSOLE: Sending request to {monitoring_api_url}/stats")
                stats_response = requests.get(f"{monitoring_api_url}/stats", timeout=5)
                print(f"CONSOLE: Stats response status: {stats_response.status_code}")
                
                print(f"CONSOLE: Sending request to {monitoring_api_url}/alerts")
                alerts_response = requests.get(f"{monitoring_api_url}/alerts?limit=5", timeout=5)
                print(f"CONSOLE: Alerts response status: {alerts_response.status_code}")
                
                # Check for HTTP errors
                stats_response.raise_for_status()
                alerts_response.raise_for_status()
                
                # Extract the 'data' payload from the responses
                stats_payload = stats_response.json().get('data', {})
                alerts_payload = alerts_response.json().get('data', [])
                
                print(f"CONSOLE: Stats payload: {json.dumps(stats_payload, indent=2)}")
                print(f"CONSOLE: Alerts payload: {json.dumps(alerts_payload[:2], indent=2)}")
                
                # Extract stats from different modules
                ingestion_stats = stats_payload.get('ingestion', {})
                detection_stats = stats_payload.get('detection', {})
                alert_stats = stats_payload.get('alerts', {})
                
                # Format alerts for the frontend
                recent_alerts = [{
                    'timestamp': format_timestamp(alert.get('timestamp')),
                    'message': alert.get('message', 'No message provided'),
                    'severity': alert.get('priority', 'unknown').capitalize()
                } for alert in alerts_payload]
                
                # Key metrics
                alerts_by_priority = alert_stats.get('alerts_by_priority', {})
                risk_level = "Low"
                if alerts_by_priority.get('high', 0) > 0:
                    risk_level = "High"
                elif alerts_by_priority.get('medium', 0) > 0:
                    risk_level = "Medium"
                    
                key_metrics = {
                    'sensitive_files': detection_stats.get('sensitive_files_found', 0),
                    'total_scans': ingestion_stats.get('total_scans', 0),
                    'risk_level': risk_level,
                    'last_scan_date': format_timestamp(alert_stats.get('newest_alert'))
                }
                
                # Data types chart data
                sensitive_data_types = {
                    'High Priority': alerts_by_priority.get('high', 0),
                    'Medium Priority': alerts_by_priority.get('medium', 0),
                    'Low Priority': alerts_by_priority.get('low', 0)
                }
                
                # Mock compliance status for now
                compliance_status = {'GDPR': 95, 'CCPA': 88, 'HIPAA': 92}
                
                # Final data structure
                final_data = {
                    'key_metrics': key_metrics,
                    'sensitive_data_types': sensitive_data_types,
                    'compliance_status': compliance_status,
                    'recent_alerts': recent_alerts
                }
                
                print(f"CONSOLE: Returning live API data with metrics: {key_metrics}")
                return jsonify(final_data)
                
            except requests.RequestException as e:
                print(f"CONSOLE ERROR: API request failed: {type(e).__name__}: {e}")
                # Use mock data if API calls fail
                return jsonify(get_mock_data())
            
        except Exception as e:
            # Comprehensive error logging
            print(f"CONSOLE ERROR: Unexpected error: {type(e).__name__}: {e}")
            print(f"CONSOLE ERROR: Traceback: {traceback.format_exc()}")
            
            # Still return mock data even after unexpected errors
            try:
                return jsonify(get_mock_data())
            except:
                return jsonify({"error": "Critical error generating response"}), 500
    
    def format_timestamp(iso_string):
        """Helper to format timestamps"""
        if not iso_string:
            return "N/A"
        try:
            if iso_string.endswith('Z'):
                iso_string = iso_string[:-1] + '+00:00'
            dt_object = datetime.fromisoformat(iso_string)
            return dt_object.strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            return "Invalid Date"

    # 6. Register error handlers and request hooks
    register_handlers(app)

    return app

# 7. Helper functions (Locale Selector, Handlers)
def get_locale():
    """Selects language for the current request."""
    if current_user.is_authenticated and hasattr(current_user, 'language') and current_user.language:
        return current_user.language
    return session.get('lang', 'en')

def register_handlers(app):
    """Registers request/response handlers and error handlers."""
    @app.errorhandler(500)
    def handle_500_error(e):
        app.logger.error(f"500 Error: {str(e)}")
        app.logger.error("Traceback:\n" + traceback.format_exc())
        return "Internal Server Error", 500

    @app.before_request
    def log_request_info():
        if request.path.startswith('/static/'):
            return
        # Log the specific endpoint that was matched
        endpoint = request.endpoint or 'No endpoint matched'
        app.logger.debug(f"Request: {request.method} {request.url} -> Matched endpoint: [{endpoint}]")

    @app.after_request
    def log_response_info(response):
        if request.path.startswith('/static/'):
            return response
        app.logger.debug(f"Response: {response.status}")
        return response

# 8. Main execution
if __name__ == '__main__':
    app = create_app()
    # Note: Setting debug=True is not recommended for production
    app.run(host='0.0.0.0', port=5000, debug=True)
