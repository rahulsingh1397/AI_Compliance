import os
# Set the FLASK_APP environment variable
os.environ['FLASK_APP'] = 'AIComplianceMonitoring.agents.monitoring.api'
# Import the app object
from AIComplianceMonitoring.agents.monitoring.api import app

if __name__ == '__main__':
    # Run the app on port 5001
    app.run(host='0.0.0.0', port=5001, debug=True)
