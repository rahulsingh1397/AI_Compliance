# UI Agent

The UI Agent provides a web interface for interacting with the AI Compliance Monitoring system. It's built using Flask and offers a dashboard for monitoring compliance status, managing configurations, and triggering compliance operations.

## Key Components

### 1. Core Application (app.py)
- Flask application setup
- Blueprint registration
- Authentication configuration
- Route handlers for main application endpoints

### 2. API Module (api.py)
- RESTful API endpoints for compliance data
- Integration with other agents
- JSON data exchange format
- Authentication and access control for API endpoints

### 3. Authentication (auth/)
- User authentication and authorization
- Login/logout functionality
- Permission management
- Security controls

### 4. Dashboard (dashboard/)
- Compliance monitoring dashboard
- Data visualization components
- Real-time compliance status updates
- Interactive controls for compliance operations

### 5. Database Models (models.py)
- SQLAlchemy ORM models
- User account management
- Configuration storage
- Compliance data persistence

### 6. Database Initialization (init_db.py)
- Database schema setup
- Initial data population
- Migration management

### 7. Templates and Static Assets
- HTML templates for web pages
- CSS styles and JavaScript for interactive elements
- Images and other static resources

## Configuration

The UI Agent can be configured through:
- `config.py` - Core configuration settings
- Environment variables
- Settings UI in the web application

## Running the UI Agent

```bash
# Initialize the database (first time only)
python init_db.py

# Start the development server
python app.py

# For production deployment
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## API Documentation

The API provides endpoints for:
- `/api/compliance_data` - Get current compliance status
- `/api/trigger_scan` - Trigger a compliance scan
- `/api/reports` - Access compliance reports

## Integration with Other Agents

The UI Agent integrates with:
- Data Discovery Agent - For triggering scans and displaying results
- Monitoring Agent - For displaying real-time monitoring data
- Reporting Agent - For generating and displaying reports
- Remediation Agent - For triggering and tracking remediation actions
