# Privacy-Preserving Agent Examples

This directory contains example scripts demonstrating the functionality of the Privacy-Preserving Agent components.

## Available Examples

### demo.py

Comprehensive demonstration script showing how to use:
- Zero-Knowledge Machine Learning (ZKML)
- Federated Learning
- Secure Audit Log
- Data Protection components

The script creates a synthetic dataset and demonstrates each component in a full privacy-preserving workflow.

### federated_example.py

Detailed example of Federated Learning implementation with:
- Fully Homomorphic Encryption (FHE) using TenSEAL
- Differential Privacy
- Client-side model training and encryption
- Secure aggregation of encrypted model updates

## Usage

Run the examples from the project root with:

```
# Run the main demo
python -m AIComplianceMonitoring.agents.privacy_preserving.examples.demo

# Run the federated learning example
python -m AIComplianceMonitoring.agents.privacy_preserving.examples.federated_example
```
