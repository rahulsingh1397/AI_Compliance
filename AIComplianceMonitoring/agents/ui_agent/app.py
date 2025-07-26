import logging
import os
import traceback
from datetime import timedelta, datetime
import json

from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from flask_login import current_user, login_required
from flask_migrate import Migrate
import threading
from AIComplianceMonitoring.agents.monitoring.agent import create_monitoring_agent_service

from AIComplianceMonitoring.agents.ui_agent import config
from AIComplianceMonitoring.agents.ui_agent.extensions import db, login_manager, babel # Import extensions from central file
from AIComplianceMonitoring.agents.ui_agent.formatters import format_timestamp as fmt_timestamp, format_file_size

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
        from AIComplianceMonitoring.agents.ui_agent import models
        return models.User.query.get(int(user_id))

    # 4. Register custom template filters
    app.jinja_env.filters['datetime'] = fmt_timestamp
    app.jinja_env.filters['filesize'] = format_file_size

    # 5. Register blueprints
    register_blueprints(app)

    # 6. Add root redirect
    @app.route('/', endpoint='index')
    def index_redirect():
        return redirect(url_for('dashboard.index'))
        
    # TEMPORARY: Direct test endpoint to debug dashboard data issues
    @app.route('/test_compliance_data')
    @login_required
    def test_compliance_data():
        from .utils import fetch_live_compliance_data
        live_data = fetch_live_compliance_data()
        return jsonify(live_data)

    @app.route('/run_scan', methods=['POST'])
    @login_required
    def run_scan():
        """Triggers a new data scan in a background thread."""
        from threading import Thread
        import traceback
        from datetime import datetime
        from .models import ScanHistory, db

        def scan_in_background():
            scan_id = f"scan_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            print(f"BACKGROUND SCAN [{scan_id}]: Starting comprehensive data discovery scan...")
            
            # Use app context for database operations in background thread
            with app.app_context():
                try:
                    # Create scan history record
                    scan_record = ScanHistory(
                        scan_id=scan_id,
                        start_time=datetime.utcnow(),
                        status='running',
                        files_scanned=0,
                        sensitive_files_found=0
                    )
                    
                    db.session.add(scan_record)
                    db.session.commit()
                    print(f"BACKGROUND SCAN [{scan_id}]: Scan record created")
                    
                    # Get data directory
                    data_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
                    print(f"BACKGROUND SCAN [{scan_id}]: Target directory: {data_directory}")
                    
                    if not os.path.exists(data_directory):
                        raise FileNotFoundError(f"Data directory not found: {data_directory}")
                    
                    # Initialize Data Discovery Agent
                    print(f"BACKGROUND SCAN [{scan_id}]: Initializing Data Discovery Agent...")
                    from AIComplianceMonitoring.agents.data_discovery.agent import DataDiscoveryAgent, AgentConfig
                    
                    # Configure the agent for this scan
                    config = AgentConfig(
                        max_workers=4,  # Limit workers to avoid overwhelming the system
                        sample_size=500  # Reasonable sample size for performance
                    )
                    
                    # Use context manager for proper resource management
                    with DataDiscoveryAgent(config=config) as agent:
                        print(f"BACKGROUND SCAN [{scan_id}]: Running batch file scan...")
                        
                        # Run comprehensive scan using Data Discovery Agent
                        scan_results = agent.batch_scan_files(
                            directory_path=data_directory,
                            file_extensions=['.csv', '.txt', '.json', '.xlsx', '.xls'],
                            max_workers=4
                        )
                        
                        print(f"BACKGROUND SCAN [{scan_id}]: Data Discovery Agent scan completed")
                        print(f"BACKGROUND SCAN [{scan_id}]: Results: {scan_results.get('total_files_scanned', 0)} files, {scan_results.get('sensitive_files_found', 0)} sensitive files")
                        
                        # Update scan record with actual results
                        scan_record.files_scanned = scan_results.get('total_files_scanned', 0)
                        scan_record.sensitive_files_found = scan_results.get('sensitive_files_found', 0)
                        scan_record.end_time = datetime.utcnow()
                        scan_record.status = 'completed'
                        
                        # Store detailed results path if available
                        if scan_results.get('scan_details'):
                            results_filename = f"scan_results_{scan_id}.json"
                            results_path = os.path.join(data_directory, '..', 'results', results_filename)
                            os.makedirs(os.path.dirname(results_path), exist_ok=True)
                            
                            with open(results_path, 'w') as f:
                                json.dump(scan_results, f, indent=2, default=str)
                            
                            scan_record.results_path = results_path
                            print(f"BACKGROUND SCAN [{scan_id}]: Detailed results saved to {results_path}")
                    
                    db.session.commit()
                    print(f"BACKGROUND SCAN [{scan_id}]: Comprehensive scan completed successfully")
                    
                except Exception as e:
                    print(f"BACKGROUND SCAN [{scan_id}]: Error during scan: {e}")
                    print(f"BACKGROUND SCAN [{scan_id}]: Traceback: {traceback.format_exc()}")
                    
                    # Update scan record with error
                    try:
                        scan_record.end_time = datetime.utcnow()
                        scan_record.status = 'failed'
                        db.session.commit()
                    except Exception as db_error:
                        print(f"BACKGROUND SCAN [{scan_id}]: Failed to update scan record: {db_error}")
                finally:
                    db.session.close()

        # Run the scan in a background thread to avoid blocking the UI
        thread = Thread(target=scan_in_background)
        thread.daemon = True
        thread.start()

        return jsonify({'status': 'success', 'message': 'Scan initiated successfully. Results will be available upon completion.'})

    @app.route('/test_scan', methods=['GET'])
    @login_required
    def test_scan():
        """Test scan functionality without threading for debugging."""
        try:
            from .models import ScanHistory, db
            from datetime import datetime
            
            # Get data directory
            data_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
            
            if not os.path.exists(data_directory):
                return jsonify({'error': f'Data directory not found: {data_directory}'})
            
            # List files in directory
            files_found = []
            for root, dirs, files in os.walk(data_directory):
                for file in files:
                    if file.lower().endswith(('.csv', '.txt', '.json', '.xlsx', '.xls')):
                        files_found.append(os.path.join(root, file))
            
            # Test database connection
            scan_count = ScanHistory.query.count()
            
            return jsonify({
                'status': 'success',
                'data_directory': data_directory,
                'files_found': len(files_found),
                'file_list': files_found[:5],  # First 5 files
                'existing_scans': scan_count,
                'database_working': True
            })
            
        except Exception as e:
            import traceback
            return jsonify({
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            })

    # 7. Start Monitoring Agent Service in a background thread
    # The check prevents the thread from starting twice in debug mode
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        monitoring_agent = create_monitoring_agent_service(app)
        monitor_thread = threading.Thread(target=monitoring_agent.run, kwargs={'host': '0.0.0.0', 'port': 5001})
        monitor_thread.daemon = True
        monitor_thread.start()
        app.logger.info("Monitoring agent service started in background thread.")

    # 8. Register error handlers and request hooks
    register_handlers(app)

    return app

# 9. Helper functions (Locale Selector, Handlers)
def register_blueprints(app):
    """Register all blueprints for the application."""
    from .scripts.main import main_bp
    from . import auth_bp, dashboard_bp, settings_bp, api_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(api_bp, url_prefix='/api')

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

