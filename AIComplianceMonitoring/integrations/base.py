from abc import ABC, abstractmethod

class TicketingIntegration(ABC):
    """Abstract base class for ticketing system integrations."""
    @abstractmethod
    def create_ticket(self, alert_data):
        pass

class NotificationIntegration(ABC):
    """Abstract base class for notification system integrations."""
    @abstractmethod
    def send_notification(self, message, ticket_url):
        pass

class SIEMIntegration(ABC):
    """Abstract base class for SIEM integrations."""
    @abstractmethod
    def get_new_alerts(self, last_run_timestamp):
        """Fetches new alerts from the SIEM since the last run."""
        pass
