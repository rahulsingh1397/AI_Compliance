import logging
import requests
from datetime import datetime
from .base import SIEMIntegration

# Configure logging
log = logging.getLogger(__name__)

class QRadarIntegration(SIEMIntegration):
    """Handles fetching offenses from IBM QRadar."""

    def __init__(self, qradar_config):
        """Initializes the QRadar client."""
        self.server_url = qradar_config.get('server')
        self.api_token = qradar_config.get('api_token')
        self.api_version = qradar_config.get('api_version', '12.0')
        self.offense_filter = qradar_config.get('offense_filter', 'status="OPEN"')
        
        if not all([self.server_url, self.api_token]):
            raise ValueError("QRadar configuration is missing required fields (server, api_token).")
            
        self.api_endpoint = f"{self.server_url.rstrip('/')}/api"
        self.headers = {
            'Accept': 'application/json',
            'Version': self.api_version,
            'SEC': self.api_token,
        }
        # In a production environment, certificate verification should be handled properly.
        # For development, we disable warnings about self-signed certificates.
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
        log.info("QRadar integration initialized successfully.")

    def get_new_alerts(self, last_run_timestamp=None):
        """
        Fetches new offenses from QRadar since the last run.

        Args:
            last_run_timestamp (datetime, optional): The timestamp of the last successful run.

        Returns:
            list: A list of standardized alert dictionaries.
        """
        offenses_url = f"{self.api_endpoint}/siem/offenses"
        
        # Build the filter query
        filters = [self.offense_filter]
        if last_run_timestamp:
            # QRadar API expects time in milliseconds since epoch
            qradar_timestamp = int(last_run_timestamp.timestamp() * 1000)
            filters.append(f"start_time > {qradar_timestamp}")
        
        params = {'filter': ' and '.join(filter for filter in filters if filter)}

        try:
            log.info(f"Fetching QRadar offenses with filter: {params['filter']}")
            # Disabling SSL verification for development purposes. Should be enabled in production.
            response = requests.get(offenses_url, headers=self.headers, params=params, verify=False)
            response.raise_for_status()
            
            qradar_offenses = response.json()
            log.info(f"Found {len(qradar_offenses)} new offenses in QRadar.")
            
            return [self._transform_offense_to_alert(offense) for offense in qradar_offenses]

        except requests.exceptions.RequestException as e:
            log.error(f"Failed to fetch offenses from QRadar: {e}")
            return []
        except ValueError: # Handles JSON decoding errors
            log.error(f"Failed to parse QRadar response. Response text: {response.text}")
            return []

    def _transform_offense_to_alert(self, offense):
        """Transforms a QRadar offense into a standardized alert dictionary."""
        return {
            'id': offense.get('id'),
            'title': offense.get('description', 'Untitled QRadar Offense').strip(),
            'description': f"QRadar Offense ID: {offense.get('id')}\n"
                           f"Status: {offense.get('status')}\n"
                           f"Severity: {offense.get('severity')}\n"
                           f"Source IPs: {offense.get('source_address_ids', [])}\n"
                           f"Destination IPs: {offense.get('local_destination_address_ids', [])}\n"
                           f"Categories: {offense.get('categories', [])}",
            'severity': self._map_severity(offense.get('severity')),
            'source': 'QRadar',
            'raw_offense': offense
        }

    def _map_severity(self, qradar_severity):
        """Maps QRadar severity (0-10) to a standard string."""
        if qradar_severity is None:
            return 'Unknown'
        if qradar_severity >= 9:
            return 'Critical'
        if qradar_severity >= 7:
            return 'High'
        if qradar_severity >= 4:
            return 'Medium'
        return 'Low'
