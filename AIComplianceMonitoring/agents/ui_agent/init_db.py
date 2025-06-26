# To run this script, navigate to the project root (AI_Compliance) and run:
# python -m AIComplianceMonitoring.agents.ui_agent.init_db

from .app import app
from .extensions import db
from .models import User, LoginAttempt  # Import all models

def init_db():
    """Creates all database tables."""
    with app.app_context():
        print("Creating all database tables...")
        db.create_all()
        print("Database tables created successfully!")

if __name__ == '__main__':
    init_db()
