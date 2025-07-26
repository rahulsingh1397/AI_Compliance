# Integrations

This directory contains integration modules for connecting the AI Compliance Monitoring system with external services, compliance data sources, and enterprise systems.

## Key Components

### 1. Base Integration (base.py)
- Abstract base class for all integrations
- Common interface and methods
- Configuration handling

### 2. Integration Agent (agent.py)
- Manages multiple integrations
- Coordinates data flow between integrations
- Handles authentication and connection management

### 3. Configuration (config.py)
- Integration configuration management
- Environment-specific settings
- Default configuration values

## Available Integrations

### Compliance Screening List Service (csl_service.py)
- Integration with compliance screening lists
- Supports checking against sanctions and restricted party lists
- Used by the monitoring agent's compliance checker

### JIRA Integration (jira.py)
- Integration with Atlassian JIRA
- Creates tickets for compliance issues
- Tracks remediation activities

### QRadar Integration (qradar.py)
- Integration with IBM QRadar SIEM
- Sends compliance alerts to security monitoring
- Retrieves security events for compliance analysis

### Slack Integration (slack.py)
- Notifications via Slack channels
- Alert delivery to team members
- Interactive compliance workflows

## Usage

Integrations are typically used by agent modules rather than directly:

```python
from AIComplianceMonitoring.integrations.csl_service import CslService

# Initialize the integration
csl_service = CslService(csl_url="https://example.com/csl.json", cache_ttl_seconds=3600)

# Use the integration
is_restricted = csl_service.search_name("Example Entity")
```

## Adding New Integrations

To add a new integration:

1. Create a new Python file in this directory
2. Extend the `BaseIntegration` class from `base.py`
3. Implement the required methods
4. Register the integration with the IntegrationAgent if needed
