import time
import logging
from datetime import datetime, timezone
from .config import Config
from .jira import JiraIntegration
from .qradar import QRadarIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IntegrationAgent:
    """Orchestrates the integration of compliance alerts with external systems."""
    def __init__(self, config_path='../../integration_config.yaml'):
        self.config = Config(config_path)
        self.running = False
        self.ticketing_integration = None
        self.siem_integration = None
        self.last_run_timestamp = None
        self._initialize_integrations()

    def _initialize_integrations(self):
        """Initializes integration clients based on the configuration."""
        # Ticketing Integration (Jira)
        ticketing_config = self.config.ticketing_config
        if ticketing_config.get('provider') == 'jira':
            self.ticketing_integration = JiraIntegration(ticketing_config.get('jira', {}))
            logging.info("Jira ticketing integration loaded.")
        
        # SIEM Integration (QRadar)
        siem_config = self.config.get('siem', {})
        if siem_config.get('provider') == 'qradar':
            self.siem_integration = QRadarIntegration(siem_config.get('qradar', {}))
            logging.info("QRadar SIEM integration loaded.")

    def run(self):
        """Starts the agent's main loop."""
        logging.info("Integration Agent started.")
        self.running = True
        # For the first run, fetch alerts from the last 24 hours as a baseline
        self.last_run_timestamp = datetime.now(timezone.utc) - timedelta(days=1)
        while self.running:
            self.process_alerts()
            time.sleep(self.config.polling_interval)

    def stop(self):
        """Stops the agent's main loop."""
        logging.info("Integration Agent stopping.")
        self.running = False

    def process_alerts(self):
        """Fetches new alerts and triggers integrations."""
        current_run_timestamp = datetime.now(timezone.utc)
        logging.info(f"Checking for new alerts since {self.last_run_timestamp.isoformat()}")
        
        alerts = self.get_new_alerts()
        
        if not alerts:
            logging.info("No new alerts found.")
        else:
            for alert in alerts:
                logging.info(f"Processing alert: {alert.get('title')}")
                ticket_url = None
                if self.ticketing_integration:
                    ticket_url = self.ticketing_integration.create_ticket(alert)
                
                if ticket_url:
                    logging.info(f"Ticket created successfully: {ticket_url}")
                else:
                    logging.warning(f"Failed to create ticket for alert: {alert.get('title')}")
        
        # Update the timestamp for the next run
        self.last_run_timestamp = current_run_timestamp
        logging.info(f"Processing complete. Next check will be for alerts after {self.last_run_timestamp.isoformat()}")

    def get_new_alerts(self):
        """Fetches new alerts from the configured SIEM integration."""
        if not self.siem_integration:
            logging.warning("No SIEM integration configured. Cannot fetch alerts.")
            return []
        
        return self.siem_integration.get_new_alerts(self.last_run_timestamp)

def run_agent_once(config_file='../../integration_config.yaml'):
    """Initializes and runs the agent for a single cycle for testing."""
    logging.info("--- Running Integration Agent for a single cycle ---")
    try:
        agent = IntegrationAgent(config_path=config_file)
        # Set a fixed historical timestamp for a consistent test run
        agent.last_run_timestamp = datetime.now(timezone.utc).replace(year=2024)
        agent.process_alerts()
    except FileNotFoundError as e:
        logging.error(f"Configuration file error: {e}. Make sure '{config_file}' is correctly located.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the agent run: {e}", exc_info=True)
    logging.info("--- Single cycle run complete ---")


if __name__ == '__main__':
    # To run this test, ensure you are in the project's root directory
    # and execute as a module: python -m AIComplianceMonitoring.integrations.agent
    run_agent_once()
