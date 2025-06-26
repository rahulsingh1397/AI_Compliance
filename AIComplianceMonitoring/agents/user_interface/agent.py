"""
User Interface Agent for AI-Enhanced Data Privacy and Compliance Monitoring.

This module implements the User Interface Agent that manages the web-based dashboard
with role-based access control (RBAC) and multi-language support.
"""

import os
import json
import logging
import pytz
from typing import Dict, Any, List, Optional, Union, Tuple
from flask import (
    Flask, request, jsonify, render_template, 
    session, redirect, url_for, flash
)
from flask_login import (
    LoginManager, UserMixin, login_user, 
    logout_user, login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta
from functools import wraps
from dataclasses import dataclass, asdict
from typing import TypedDict

# Define types for settings
class UserSettings(TypedDict):
    theme: str
    timezone: str
    email_notifications: bool
    in_app_notifications: bool
    items_per_page: int
    notification_types: Dict[str, bool]

@dataclass
class UserSession:
    id: str
    device_info: str
    ip_address: str
    last_activity: datetime
    is_current: bool = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class User(UserMixin):
    """User class for Flask-Login."""
    
    def __init__(self, user_id, username, email, role, language="en"):
        self.id = user_id
        self.username = username
        self.email = email
        self.role = role
        self.language = language
    
    def has_permission(self, permission):
        """Check if user has a specific permission."""
        role_permissions = {
            'admin': ['view_dashboard', 'view_reports', 'manage_users', 'configure_system', 'export_data'],
            'compliance_officer': ['view_dashboard', 'view_reports', 'export_data'],
            'data_officer': ['view_dashboard', 'view_reports', 'export_data'],
            'security_analyst': ['view_dashboard', 'view_reports'],
            'auditor': ['view_reports', 'export_data'],
            'viewer': ['view_dashboard']
        }
        
        return permission in role_permissions.get(self.role, [])

class UserInterfaceAgent:
    """
    User Interface Agent for managing the web-based dashboard.
    
    This agent provides a Flask-based web interface with role-based access control
    and multi-language support for the AI-Enhanced Data Privacy and Compliance
    Monitoring system.
    """
    
    def __init__(self, 
                config_path: str = None,
                host: str = "0.0.0.0",
                port: int = 5000,
                debug: bool = False):
        """
        Initialize the User Interface Agent.
        
        Args:
            config_path: Path to configuration file
            host: Host to run the Flask server on
            port: Port to run the Flask server on
            debug: Whether to run Flask in debug mode
        """
        logger.info("Initializing User Interface Agent")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize Flask app
        self.app = Flask(__name__, 
                        static_folder='static',
                        template_folder='templates')
        self.app.secret_key = self.config.get('secret_key', os.urandom(24))
        self.app.config['JWT_SECRET_KEY'] = self.config.get('jwt_secret_key', os.urandom(24))
        self.app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
        
        # Initialize Flask-Login
        self.login_manager = LoginManager()
        self.login_manager.init_app(self.app)
        self.login_manager.login_view = 'login'
        
        # Mock user database (would be replaced with actual database in production)
        self.users = {
            '1': {
                'id': '1',
                'username': 'admin',
                'email': 'admin@example.com',
                'password': generate_password_hash('admin123'),
                'role': 'admin',
                'language': 'en'
            },
            '2': {
                'id': '2',
                'username': 'compliance',
                'email': 'compliance@example.com',
                'password': generate_password_hash('compliance123'),
                'role': 'compliance_officer',
                'language': 'en'
            },
            '3': {
                'id': '3',
                'username': 'security',
                'email': 'security@example.com',
                'password': generate_password_hash('security123'),
                'role': 'security_analyst',
                'language': 'en'
            }
        }
        
        # Available languages
        self.languages = {
            'en': 'English',
            'es': 'Español',
            'fr': 'Français',
            'de': 'Deutsch'
        }
        
        # Register routes
        self._register_routes()
        
        # Server settings
        self.host = host
        self.port = port
        self.debug = debug
        
        logger.info("User Interface Agent initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary with configuration
        """
        default_config = {
            'secret_key': os.urandom(24).hex(),
            'jwt_secret_key': os.urandom(24).hex(),
            'session_timeout': 3600,  # 1 hour
            'default_language': 'en'
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return {**default_config, **config}
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
        
        return default_config
    
    def _register_routes(self):
        """Register Flask routes."""
        
        @self.login_manager.user_loader
        def load_user(user_id):
            """Load user by ID."""
            if user_id in self.users:
                user_data = self.users[user_id]
                return User(
                    user_id=user_data['id'],
                    username=user_data['username'],
                    email=user_data['email'],
                    role=user_data['role'],
                    language=user_data['language']
                )
            return None
        
        def role_required(roles):
            """Decorator to require specific roles for a route."""
            def decorator(f):
                @wraps(f)
                def decorated_function(*args, **kwargs):
                    if current_user.role not in roles:
                        return jsonify({'error': 'Unauthorized'}), 403
                    return f(*args, **kwargs)
                return decorated_function
            return decorator
        
        @self.app.route('/')
        def index():
            """Render the index page."""
            if current_user.is_authenticated:
                return redirect(url_for('dashboard'))
            return redirect(url_for('login'))
        
        @self.app.route('/login', methods=['GET', 'POST'])
        def login():
            """Handle user login."""
            if current_user.is_authenticated:
                return redirect(url_for('dashboard'))
            
            if request.method == 'POST':
                username = request.form.get('username')
                password = request.form.get('password')
                remember = 'remember' in request.form
                
                # Find user by username
                user_id = None
                for uid, user_data in self.users.items():
                    if user_data['username'] == username:
                        user_id = uid
                        break
                
                if user_id and check_password_hash(self.users[user_id]['password'], password):
                    user = User(
                        user_id=self.users[user_id]['id'],
                        username=self.users[user_id]['username'],
                        email=self.users[user_id]['email'],
                        role=self.users[user_id]['role'],
                        language=self.users[user_id]['language']
                    )
                    login_user(user, remember=remember)
                    return redirect(url_for('dashboard'))
                
                return render_template('login.html', error='Invalid username or password')
            
            return render_template('login.html')
        
        @self.app.route('/logout')
        @login_required
        def logout():
            """Handle user logout."""
            logout_user()
            return redirect(url_for('login'))
            
        @self.app.route('/settings', methods=['GET', 'POST'])
        @login_required
        def settings():
            """Render and handle the settings page."""
            # Get all available timezones
            timezones = pytz.all_timezones
            
            # Mock user settings (in a real app, this would come from a database)
            user_settings = {
                'theme': 'light',
                'timezone': 'UTC',
                'email_notifications': True,
                'in_app_notifications': True,
                'items_per_page': 25,
                'notification_types': {
                    'critical': True,
                    'warning': True,
                    'info': True
                }
            }
            
            # Mock active sessions (in a real app, this would come from a database)
            active_sessions = [
                {
                    'id': '1',
                    'device_info': 'Windows 10, Chrome',
                    'ip_address': '192.168.1.1',
                    'last_activity': datetime.utcnow(),
                    'is_current': True
                },
                {
                    'id': '2',
                    'device_info': 'Mac OS, Safari',
                    'ip_address': '192.168.1.2',
                    'last_activity': datetime.utcnow() - timedelta(days=1),
                    'is_current': False
                }
            ]
            
            # Handle form submissions
            if request.method == 'POST':
                form_type = request.form.get('form_type')
                
                if form_type == 'profile':
                    # Handle profile update
                    # In a real app, this would update the user in the database
                    flash('Profile updated successfully!', 'success')
                    return redirect(url_for('settings'))
                    
                elif form_type == 'password':
                    # Handle password change
                    current_password = request.form.get('current_password')
                    new_password = request.form.get('new_password')
                    confirm_password = request.form.get('confirm_password')
                    
                    # In a real app, verify current password and update
                    if new_password == confirm_password and len(new_password) >= 8:
                        flash('Password updated successfully!', 'success')
                    else:
                        flash('Failed to update password. Please check your input.', 'danger')
                    return redirect(url_for('settings'))
                    
                elif form_type == 'notifications':
                    # Handle notification preferences
                    flash('Notification preferences saved!', 'success')
                    return redirect(url_for('settings'))
                    
                elif form_type == 'system' and current_user.role in ['admin', 'system_admin']:
                    # Handle system settings (admin only)
                    flash('System settings updated!', 'success')
                    return redirect(url_for('settings'))
            
            # Render the settings template
            return render_template(
                'settings.html',
                user_settings=user_settings,
                active_sessions=active_sessions,
                timezones=timezones[:50]  # Limit to first 50 timezones for the dropdown
            )
            
        @self.app.route('/api/settings/update', methods=['POST'])
        @login_required
        def update_settings():
            """Update user settings via API."""
            data = request.get_json()
            setting_type = data.get('type')
            settings = data.get('settings', {})
            
            # In a real app, this would update the database
            # For now, we'll just return a success message
            return jsonify({
                'status': 'success',
                'message': f'{setting_type} settings updated successfully'
            })
            
        @self.app.route('/api/sessions/revoke/<session_id>', methods=['POST'])
        @login_required
        def revoke_session(session_id):
            """Revoke a user session."""
            # In a real app, this would remove the session from the database
            # For now, we'll just return a success message
            return jsonify({
                'status': 'success',
                'message': f'Session {session_id} has been revoked'
            })
        
        # Mock data for dashboard (would be replaced with actual API calls in production)
        MOCK_DATA = {
            'sensitive_data_summary': {
                'total_records': 1250,
                'structured_data': 850,
                'unstructured_data': 400,
                'high_risk': 320,
                'medium_risk': 580,
                'low_risk': 350
            },
            'data_types': {
                'PII': 620,
                'Financial': 280,
                'Health': 150,
                'Other': 200
            },
            'compliance_status': {
                'GDPR': {
                    'compliant': 85,
                    'non_compliant': 15,
                    'issues': [
                        {'severity': 'high', 'count': 5, 'description': 'Missing consent records'},
                        {'severity': 'medium', 'count': 10, 'description': 'Incomplete data processing records'}
                    ]
                },
                'CCPA': {
                    'compliant': 90,
                    'non_compliant': 10,
                    'issues': [
                        {'severity': 'medium', 'count': 7, 'description': 'Incomplete data inventory'},
                        {'severity': 'low', 'count': 3, 'description': 'Missing privacy notices'}
                    ]
                },
                'HIPAA': {
                    'compliant': 92,
                    'non_compliant': 8,
                    'issues': [
                        {'severity': 'high', 'count': 3, 'description': 'Unencrypted PHI'},
                        {'severity': 'low', 'count': 5, 'description': 'Incomplete access logs'}
                    ]
                }
            },
            'recent_alerts': [
                {
                    'id': 'alert-001',
                    'timestamp': '2025-06-08T14:35:22Z',
                    'severity': 'high',
                    'description': 'Unauthorized access attempt to PII data',
                    'source': 'Database server DB-PROD-01',
                    'status': 'open'
                },
                {
                    'id': 'alert-002',
                    'timestamp': '2025-06-08T12:15:45Z',
                    'severity': 'medium',
                    'description': 'Unusual data transfer pattern detected',
                    'source': 'File server FS-PROD-03',
                    'status': 'investigating'
                },
                {
                    'id': 'alert-003',
                    'timestamp': '2025-06-08T09:22:18Z',
                    'severity': 'low',
                    'description': 'New sensitive data identified',
                    'source': 'Email server MAIL-01',
                    'status': 'resolved'
                }
            ],
            'discovery_metrics': {
                'last_scan': '2025-06-08T00:00:00Z',
                'files_scanned': 15420,
                'databases_scanned': 8,
                'new_sensitive_data_found': 142,
                'classification_accuracy': 96.5
            }
        }
        
        @self.app.route('/reports')
        @login_required
        @role_required(['admin', 'compliance_officer', 'data_officer', 'security_analyst', 'auditor'])
        def reports():
            """Render the reports page."""
            return render_template(
                'reports.html',
                user=current_user,
                page_title='Reports',
                reports_data=MOCK_DATA
            )
        
        @self.app.route('/settings', methods=['GET', 'POST'])
        @login_required
        @role_required(['admin'])
        def settings():
            """Render and handle the settings page."""
            if request.method == 'POST':
                # Handle form submission
                language = request.form.get('language', 'en')
                timezone = request.form.get('timezone', 'UTC')
                date_format = request.form.get('date_format', 'MM/DD/YYYY')
                
                # Update user preferences (in a real app, this would update the database)
                user_id = current_user.id
                if user_id in self.users:
                    self.users[user_id]['language'] = language
                    current_user.language = language
                    
                flash('Settings saved successfully!', 'success')
                return redirect(url_for('settings'))
            
            return render_template(
                'settings.html',
                user=current_user,
                page_title='Settings',
                languages=self.languages,
                timezone=current_user.timezone if hasattr(current_user, 'timezone') else 'UTC',
                date_format=current_user.date_format if hasattr(current_user, 'date_format') else 'MM/DD/YYYY'
            )
        
        # API Routes for Dashboard Data
        @self.app.route('/api/dashboard/summary')
        @login_required
        def dashboard_summary():
            """API endpoint for dashboard summary data."""
            return jsonify(MOCK_DATA['sensitive_data_summary'])
        
        @self.app.route('/api/dashboard/data-types')
        @login_required
        def data_types():
            """API endpoint for data types breakdown."""
            return jsonify(MOCK_DATA['data_types'])
        
        @self.app.route('/api/dashboard/compliance')
        @login_required
        def compliance_status():
            """API endpoint for compliance status."""
            return jsonify(MOCK_DATA['compliance_status'])
        
        @self.app.route('/api/dashboard/alerts')
        @login_required
        def recent_alerts():
            """API endpoint for recent alerts."""
            return jsonify(MOCK_DATA['recent_alerts'])
        
        @self.app.route('/api/dashboard/discovery')
        @login_required
        def discovery_metrics():
            """API endpoint for discovery metrics."""
            return jsonify(MOCK_DATA['discovery_metrics'])
        
        @self.app.route('/api/set-language', methods=['POST'])
        @login_required
        def set_language():
            """Set user language preference."""
            data = request.get_json()
            language = data.get('language')
            
            if language in self.languages:
                # Update user language in database
                user_id = current_user.id
                if user_id in self.users:
                    self.users[user_id]['language'] = language
                    current_user.language = language
                    return jsonify({'success': True})
            
            return jsonify({'error': 'Invalid language'}), 400
        
        @self.app.route('/api/user-info')
        @login_required
        def user_info():
            """Get current user information."""
            return jsonify({
                'id': current_user.id,
                'username': current_user.username,
                'email': current_user.email,
                'role': current_user.role,
                'language': current_user.language,
                'permissions': [
                    perm for perm in [
                        'view_dashboard', 'view_reports', 'manage_users',
                        'configure_system', 'export_data'
                    ] if current_user.has_permission(perm)
                ]
            })
    
    def run(self):
        """Run the Flask application."""
        logger.info(f"Starting User Interface Agent on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=self.debug)
    
    def generate_token(self, user_id: str) -> str:
        """
        Generate JWT token for API authentication.
        
        Args:
            user_id: User ID to generate token for
            
        Returns:
            JWT token string
        """
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=1)
        }
        return jwt.encode(payload, self.app.config['JWT_SECRET_KEY'], algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[str]:
        """
        Verify JWT token and return user ID if valid.
        
        Args:
            token: JWT token to verify
            
        Returns:
            User ID if token is valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
            return payload['user_id']
        except:
            return None
