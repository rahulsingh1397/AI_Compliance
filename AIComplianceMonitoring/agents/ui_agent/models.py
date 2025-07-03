from AIComplianceMonitoring.agents.ui_agent.extensions import db
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
    
    def __repr__(self):
        return f'<User {self.username}>'
