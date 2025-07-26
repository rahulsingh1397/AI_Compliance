from .extensions import db
from flask_login import UserMixin
from datetime import datetime

class LoginAttempt(db.Model):
    __tablename__ = 'login_attempts'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    ip_address = db.Column(db.String(45), nullable=False)
    user_agent = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    successful = db.Column(db.Boolean, default=False)

    __table_args__ = {'extend_existing': True}
    
    def __repr__(self):
        return f"<LoginAttempt {self.username} - {'Success' if self.successful else 'Failed'}>"

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), default='user')
    language = db.Column(db.String(5), default='en')
    timezone = db.Column(db.String(50), default='UTC')
    email_notifications = db.Column(db.Boolean, default=True)
    in_app_notifications = db.Column(db.Boolean, default=True)

    __table_args__ = {'extend_existing': True}
    
    def __repr__(self):
        return f'<User {self.username}>'

class ScanHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(db.String(100), unique=True, nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    status = db.Column(db.String(50), default='Running')
    files_scanned = db.Column(db.Integer, default=0)
    sensitive_files_found = db.Column(db.Integer, default=0)
    results_path = db.Column(db.String(255))

    __table_args__ = {'extend_existing': True}

    def __repr__(self):
        return f'<ScanHistory {self.scan_id}>'
