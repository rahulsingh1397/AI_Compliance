# Tests Directory

This directory contains all test files for the AI Compliance Monitoring project.

## Test Categories

### Anomaly Detection Tests
- `test_anomaly_detection.py` - Core anomaly detection functionality tests
- `test_anomaly_detection_unit.py` - Unit tests for individual anomaly detection components
- `test_anomaly_detection_with_feedback.py` - Integration tests for anomaly detection with RL feedback loop

### Agent Tests
- `test_alert_module.py` - Tests for the alert generation module
- `test_compliance_checker.py` - Tests for the compliance checking module
- `test_data_discovery.py` - Tests for the data discovery agent
- `test_remediation.py` - Tests for the remediation agent
- `test_original_agent.py` - Legacy tests for the original agent implementation

### Integration Tests
- `test_csl_integration.py` - Tests for CSL integration
- `test_hybrid_monitoring.py` - Tests for hybrid monitoring architecture
- `test_deps_fixed.py` - Tests for dependency fixes

### Utility Tests
- `test_import.py` - Simple import tests to verify module structure
- `test_spacy.py` - Tests for spaCy NLP integration
- `test_spacy_file.py` - File-based tests for spaCy

## Test Data

The `test_data/` directory contains fixtures and sample data used in tests.

## Test Results

The `test_results/` directory contains output from test runs, including metrics, visualizations, and logs.

## Running Tests

Run all tests from the project root:
```
python -m unittest discover tests
```

Run a specific test:
```
python -m unittest tests/test_anomaly_detection.py
```
