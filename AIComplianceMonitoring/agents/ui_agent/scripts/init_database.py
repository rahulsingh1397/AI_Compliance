# To run this script, navigate to the project root (AI_Compliance) and run:
# python -m AIComplianceMonitoring.agents.ui_agent.init_db

from AIComplianceMonitoring.agents.ui_agent.app import create_app
from AIComplianceMonitoring.agents.ui_agent.extensions import db
from AIComplianceMonitoring.agents.ui_agent.models import User, LoginAttempt  # Import all models

def init_db():
    """Creates all database tables."""
    app = create_app()
    with app.app_context():
        print("Creating all database tables...")
        db.create_all()
        print("Database tables created successfully!")

if __name__ == '__main__':
    init_db()
