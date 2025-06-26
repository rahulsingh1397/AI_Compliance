import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Minimal Runner
# This script will run our barebones api.py and log any errors.

try:
    logging.info("Attempting to import 'app' from minimal api.py")
    from AIComplianceMonitoring.agents.monitoring.api import app
    logging.info("'app' imported successfully.")

    if __name__ == '__main__':
        logging.info("Starting minimal Flask server on port 5001...")
        app.run(host='0.0.0.0', port=5001, debug=True)

except ImportError as e:
    logging.error(f"Failed to import 'app' from api.py. This is often a path or dependency issue. Error: {e}", exc_info=True)
except Exception as e:
    logging.error(f"An unexpected error occurred while trying to start the server: {e}", exc_info=True)

