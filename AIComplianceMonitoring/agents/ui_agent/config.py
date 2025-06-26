import os

# Define the base directory for the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
INSTANCE_DIR = os.path.join(BASE_DIR, 'instance')

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(INSTANCE_DIR, 'app.db')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'
    
    # Directory for database migrations
    MIGRATIONS_DIR = os.path.join(BASE_DIR, 'migrations')
    
    # Supported languages
    LANGUAGES = ['en', 'es', 'fr', 'de']
    
    # Translation configuration
    TRANSLATION_DIR = os.path.join(os.path.dirname(__file__), 'translations')
    
    # Security settings
    SESSION_COOKIE_SECURE = False # Should be True in production over HTTPS
    REMEMBER_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # API settings
    API_RATE_LIMIT = "200 per day;50 per hour"
    
    # Agent settings
    AGENT_NAME = "UI Agent"
    AGENT_VERSION = "1.0.0"
