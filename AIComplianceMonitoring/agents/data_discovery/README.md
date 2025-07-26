# Data Discovery Agent

The Data Discovery Agent is responsible for scanning systems and identifying sensitive data that may be subject to compliance requirements. It uses machine learning and natural language processing to classify data and determine its sensitivity level.

## Key Components

### 1. DataDiscoveryAgent (agent.py)
- Main agent implementation
- Coordinates scanning operations
- Manages discovery workflows
- Integrates with other agents

### 2. MLClassifier (ml_classifier.py)
- Machine learning-based sensitive data classification
- Predicts data sensitivity levels
- Identifies PII, PHI, and other regulated data types
- Supports incremental learning from feedback

### 3. NLPModel (nlp_model.py)
- Natural language processing for text-based data
- Named entity recognition for identifying sensitive information
- Context-aware classification of text data
- Specialized processing for unstructured content

### 4. MetadataHandler (metadata_handler.py)
- Extraction and management of file and data metadata
- Classification based on metadata signals
- Provenance tracking of data assets
- Historical metadata analysis

### 5. Scripts
- Utility scripts for running scans and related operations
- Command-line interfaces for data discovery functionality
- Batch processing tools

## Old Version
The `Old_version` directory contains previous implementations that have been superseded by the current code. This directory is maintained for reference purposes only.

## Integration Points

The Data Discovery Agent integrates with:
- Privacy-Preserving Agent - To protect sensitive data once discovered
- Remediation Agent - To address compliance issues with discovered data
- Reporting Agent - To include discovery findings in compliance reports
- UI Agent - To present discovery results in the web interface

## Usage

The Data Discovery Agent can be triggered:
- Manually through the web interface
- Programmatically via the API
- On a scheduled basis using the reporting agent's scheduler
- Ad-hoc using the command-line scripts in the `scripts` directory
