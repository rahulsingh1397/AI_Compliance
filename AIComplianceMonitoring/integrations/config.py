import yaml
import os

class Config:
    """Loads configuration from a YAML file."""
    def __init__(self, config_path='integration_config.yaml'):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, key, default=None):
        """Retrieves a configuration value."""
        return self.config.get(key, default)

    @property
    def polling_interval(self):
        return self.get('polling_interval', 60)

    @property
    def ticketing_config(self):
        return self.get('ticketing', {})

    @property
    def notification_config(self):
        return self.get('notification', {})
