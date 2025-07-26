# Monitoring Agent

The Monitoring Agent provides real-time monitoring of AI systems, log ingestion, anomaly detection, and compliance checking capabilities. It's designed to detect potential compliance violations and security issues in AI operations.

## Key Components

### Core Components
- **MonitoringAgent (agent.py)**: Main agent implementation that coordinates monitoring activities
- **API (api.py)**: REST API for interacting with the monitoring agent
- **Run Script (run.py)**: Entry point for running the monitoring agent

### Log Processing
- **LogIngestionModule (log_ingestion.py)**: Handles log collection from various sources with parallel processing
- **ComplianceChecker (compliance_checker.py)**: Checks logs against compliance rules and standards

### Anomaly Detection
- **AnomalyDetection (anomaly_detection.py)**: Core anomaly detection using machine learning
- **TorchAutoencoder (torch_autoencoder.py)**: PyTorch-based autoencoder for deep learning anomaly detection
- **HybridAnomalyDetector (hybrid_anomaly_detector.py)**: Combines multiple detection methods
- **HybridMonitor (hybrid_monitor.py)**: Integrates rule-based and ML-based monitoring

### Reinforcement Learning Feedback Loop
- **FeedbackLoop (feedback_loop.py)**: Coordinates the feedback and learning loop
- **RLFeedbackManager (rl_feedback_manager.py)**: Manages reinforcement learning for model improvement
- **ValidationAgent (validation_agent.py)**: Validates detected anomalies
- **HumanInteractionAgent (human_interaction_agent.py)**: Manages human review of anomalies
- **FeedbackIntegration (feedback_integration.py)**: Integrates feedback into model improvement

### Alert Management
- **AlertModule (alert_module.py)**: Generates and manages compliance alerts

## Persistence and State Management
- **DB Directory**: Database files and scripts
- **Models Directory**: Saved ML models for anomaly detection
- **MonitoringState Directory**: Persistent state information

## Integration Points

The Monitoring Agent integrates with:
- **Data Discovery Agent**: Uses discovered data to inform monitoring rules
- **Reporting Agent**: Provides monitoring data for compliance reports
- **UI Agent**: Sends alerts and monitoring status to the dashboard
- **External Systems**: Integrates with SIEM tools like QRadar

## Usage

The agent can be run in several modes:
```bash
# Run the full monitoring agent
python run.py

# Populate test data
python populate_db.py

# Run specific components (example)
python -c "from AIComplianceMonitoring.agents.monitoring.log_ingestion import LogIngestionModule; LogIngestionModule().run()"
```

## Advanced Features

- **Parallel Log Processing**: Efficiently processes logs from multiple sources
- **Hybrid Detection**: Combines rule-based and ML approaches
- **Reinforcement Learning**: Improves over time with feedback
- **Human-in-the-loop**: Incorporates human judgment for complex cases
- **Anomaly Severity Scoring**: Prioritizes alerts based on severity
