import os
import sys
from werkzeug.security import generate_password_hash
from sqlalchemy.exc import IntegrityError

# Adjust the import path according to your project structure
# This assumes app, db are initialized in AIComplianceMonitoring.agents.ui_agent.app
# and User model is in AIComplianceMonitoring.agents.ui_agent.models
from AIComplianceMonitoring.agents.ui_agent.app import app, db
from AIComplianceMonitoring.agents.ui_agent.models import User

def init_db():
    with app.app_context():
        try:
            # Create database tables if they don't exist
            db.create_all()
            print("Database tables checked/created.")

            # Create a superuser if it doesn't exist
            admin_username = 'admin'
            if not User.query.filter_by(username=admin_username).first():
                hashed_password = generate_password_hash('admin123', method='pbkdf2:sha256')
                admin = User(
                    username=admin_username,
                    email='admin@example.com',
                    password=hashed_password,
                    role='admin',  # Ensure 'admin' is a valid role in your User model
                    language='en',
                    timezone='UTC',
                    email_notifications=True,
                    in_app_notifications=True
                )
                db.session.add(admin)
                db.session.commit()
                print(f"Admin user '{admin_username}' created successfully!")
                print(f"Username: {admin_username}")
                print("Password: admin123")
            else:
                print(f"Admin user '{admin_username}' already exists.")
            
            print("Database initialization process completed.")

        except IntegrityError as e:
            db.session.rollback()
            print(f"Database integrity error: {e}")
            print("This might be due to a unique constraint violation (e.g., user already exists with different casing, or email is already taken).")
        except Exception as e:
            db.session.rollback()
            print(f"An error occurred during database initialization: {e}")

if __name__ == '__main__':
    # Ensure the script is run in the context of your project's root for correct imports
    # You might need to adjust PYTHONPATH if running this script directly from a different location
    # Example: PYTHONPATH=. python init_db.py
    print("Starting database initialization...")
    init_db()
