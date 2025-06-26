from flask import Flask, jsonify, request
import logging
import os
from .alert_module import AlertModule

# Step 3: Re-integrate the Real AlertModule
# This version connects to the actual alert generation and database logic.

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

log.info("Attempting to load api.py with real AlertModule...")

# Default configuration for the monitoring agent API
# In a real application, this would come from a more robust config system
config = {
    'database_path': os.path.join(os.path.dirname(__file__), '..', '..', '..', 'instance', 'alerts.db'),
    'deduplication_window_seconds': 3600
}

app = Flask(__name__)

try:
    log.info("Initializing AlertModule...")
    alert_module = AlertModule(config)
    log.info("AlertModule initialized successfully.")
except Exception as e:
    log.error("Failed to initialize AlertModule!", exc_info=True)
    # If the module fails, we create a placeholder to avoid crashing the server
    alert_module = None

@app.route('/health')
def health_check():
    """Health check endpoint to confirm the service is running."""
    return jsonify({'status': 'ok', 'message': 'Monitoring Agent API is running.'})

@app.route('/stats')
def get_stats():
    """Endpoint to retrieve alert statistics."""
    if not alert_module:
        return jsonify({"error": "AlertModule not initialized"}), 500
    stats = alert_module.get_stats()
    return jsonify(stats)

@app.route('/alerts')
def get_alerts():
    """Endpoint to retrieve alerts."""
    if not alert_module:
        return jsonify({"error": "AlertModule not initialized"}), 500
    
    priority = request.args.get('priority')
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)

    alerts = alert_module.get_alerts(priority=priority, limit=limit, offset=offset)
    return jsonify(alerts)

log.info("Flask app with real AlertModule created.")
