import logging
from jira import JIRA, JIRAError
from .base import TicketingIntegration

# Configure logging
log = logging.getLogger(__name__)

class JiraIntegration(TicketingIntegration):
    """Handles ticket creation in Jira."""

    def __init__(self, jira_config):
        """Initializes the Jira client."""
        try:
            self.project_key = jira_config.get('project_key')
            self.issue_type = jira_config.get('issue_type', 'Task')
            
            if not all([self.project_key, jira_config.get('server'), jira_config.get('user'), jira_config.get('api_token')]):
                raise ValueError("Jira configuration is missing required fields (server, user, api_token, project_key).")

            self.client = JIRA(
                server=jira_config['server'],
                basic_auth=(jira_config['user'], jira_config['api_token'])
            )
            log.info("Jira integration initialized successfully.")
        except (JIRAError, ValueError) as e:
            log.error(f"Failed to initialize Jira client: {e}")
            self.client = None

    def create_ticket(self, alert_data):
        """
        Creates a new ticket in Jira based on the alert data.

        Args:
            alert_data (dict): A dictionary containing alert details.

        Returns:
            str: The URL of the created ticket, or None if creation failed.
        """
        if not self.client:
            log.error("Cannot create Jira ticket, client not initialized.")
            return None

        summary = f"Compliance Alert: {alert_data.get('title', 'Untitled Alert')}"
        description = alert_data.get('description', 'No description provided.')
        
        issue_fields = {
            'project': {'key': self.project_key},
            'summary': summary,
            'description': description,
            'issuetype': {'name': self.issue_type},
        }

        try:
            new_issue = self.client.create_issue(fields=issue_fields)
            ticket_url = new_issue.permalink()
            log.info(f"Successfully created Jira ticket: {new_issue.key} at {ticket_url}")
            return ticket_url
        except JIRAError as e:
            log.error(f"Failed to create Jira ticket: {e.status_code} - {e.text}")
            return None
