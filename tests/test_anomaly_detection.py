"""
Test script for the anomaly detection module with autoencoder.

This script creates sample log data and tests both the Isolation Forest
and Autoencoder models to ensure they're detecting anomalies as expected.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to allow imports from AIComplianceMonitoring
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the anomaly detection module
from AIComplianceMonitoring.agents.monitoring.anomaly_detection import AnomalyDetectionModule

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_normal_logs(n_samples=500):
    """Generate synthetic normal log data"""
    np.random.seed(42)  # For reproducibility
    
    # Create a dataframe with common log fields
    logs = pd.DataFrame({
        'timestamp': [datetime.now() - timedelta(hours=i) for i in range(n_samples)],
        'user_id': np.random.choice(['user1', 'user2', 'user3', 'user4', 'user5'], n_samples),
        'resource_id': np.random.choice(['res1', 'res2', 'res3', 'res4'], n_samples),
        'action': np.random.choice(['read', 'write', 'delete', 'update'], n_samples),
        'status_code': np.random.choice([200, 201, 204, 400, 403, 404, 500], 
                                       p=[0.7, 0.1, 0.1, 0.03, 0.03, 0.02, 0.02], 
                                       size=n_samples),
        'response_time': np.random.normal(100, 20, n_samples),  # Normal distribution around 100ms
        'data_size': np.random.lognormal(3, 1, n_samples),  # Log-normal distribution for data size
        'ip_address': np.random.choice(['192.168.1.1', '10.0.0.1', '172.16.0.1'], n_samples),
        'is_logged_in': np.random.choice([True, False], p=[0.9, 0.1], size=n_samples),
        'session_duration': np.random.gamma(5, 100, n_samples),  # Session durations
        'attempts': np.random.poisson(1, n_samples)  # Number of attempts follows Poisson
    })
    
    return logs

def generate_anomalous_logs(n_samples=50):
    """Generate synthetic anomalous log data"""
    np.random.seed(43)  # Different seed than normal logs
    
    # Create anomalous logs with unusual patterns
    anomalous_logs = pd.DataFrame({
        'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(n_samples)],
        'user_id': np.random.choice(['user99', 'admin', 'system', 'unknown'] + 
                                   ['user'+str(i) for i in range(6, 10)], n_samples),
        'resource_id': np.random.choice(['sensitive_data', 'admin_panel', 'security_config', 'logs'] +
                                      ['res'+str(i) for i in range(5, 8)], n_samples),
        'action': np.random.choice(['delete', 'configure', 'export', 'escalate', 'read', 'write'], n_samples),
        'status_code': np.random.choice([200, 201, 403, 500, 503], 
                                       p=[0.4, 0.1, 0.2, 0.2, 0.1], 
                                       size=n_samples),  # More errors
        'response_time': np.concatenate([
            np.random.normal(500, 100, n_samples//2),  # Much slower responses
            np.random.normal(10, 5, n_samples//2)      # Suspiciously fast responses
        ]),
        'data_size': np.concatenate([
            np.random.lognormal(8, 1, n_samples//2),   # Very large data transfers
            np.random.lognormal(1, 0.5, n_samples//2)  # Very small data transfers
        ]),
        'ip_address': np.random.choice(['8.8.8.8', '1.1.1.1', '0.0.0.0', '255.255.255.255'], n_samples),
        'is_logged_in': np.random.choice([True, False], size=n_samples),
        'session_duration': np.concatenate([
            np.random.gamma(1, 10, n_samples//2),       # Very short sessions 
            np.random.gamma(20, 500, n_samples//2)      # Very long sessions
        ]),
        'attempts': np.concatenate([
            np.random.poisson(10, n_samples//2),        # Many login attempts
            np.ones(n_samples//2)                       # Just one attempt (could be successful breach)
        ])
    })
    
    return anomalous_logs

def run_tests():
    """Run the anomaly detection tests"""
    # Create a temporary directory for models
    model_dir = Path("./test_models")
    model_dir.mkdir(exist_ok=True)
    
    # Initialize anomaly detection module
    config = {
        "model_directory": "./test_models"
    }
    detector = AnomalyDetectionModule(config)
    
    # Generate synthetic log data
    logger.info("Generating synthetic logs...")
    normal_logs = generate_normal_logs(500)
    anomalous_logs = generate_anomalous_logs(50)
    
    # Split normal logs for training and testing
    train_logs = normal_logs.sample(frac=0.7, random_state=42)
    test_normal_logs = normal_logs.drop(train_logs.index)
    
    # Train the models
    logger.info("Training models...")
    detector.update_model(train_logs)
    
    # Test both models on a mixed dataset
    test_logs = pd.concat([test_normal_logs, anomalous_logs])
    
    # Detect anomalies
    logger.info("Testing anomaly detection...")
    result_df, stats = detector.detect_anomalies(test_logs)
    
    # Evaluate results
    true_anomalies = len(anomalous_logs)
    detected_anomalies = stats["anomalies"]
    precision = sum(result_df.loc[anomalous_logs.index, 'is_anomaly']) / detected_anomalies if detected_anomalies > 0 else 0
    recall = sum(result_df.loc[anomalous_logs.index, 'is_anomaly']) / true_anomalies if true_anomalies > 0 else 0
    
    logger.info(f"Anomaly Stats: {stats}")
    logger.info(f"Precision: {precision:.2f}")
    logger.info(f"Recall: {recall:.2f}")
    logger.info(f"Found {detected_anomalies} anomalies, {true_anomalies} were planted")
    
    if "anomaly_types" in stats:
        for model_name, count in stats["anomaly_types"].items():
            logger.info(f"Model {model_name} detected {count} anomalies")
    
    # Clean up test models
    logger.info("Tests completed, cleaning up...")
    for file in model_dir.glob("*"):
        try:
            file.unlink()
        except Exception as e:
            logger.warning(f"Could not delete {file}: {e}")
    model_dir.rmdir()
    
    return {
        "precision": precision,
        "recall": recall,
        "detected": detected_anomalies,
        "actual": true_anomalies
    }

if __name__ == "__main__":
    results = run_tests()
    summary = (
        "Test Results Summary:\n"
        "=====================\n"
        f"Precision: {results['precision']:.2f}\n"
        f"Recall: {results['recall']:.2f}\n"
        f"Detected anomalies: {results['detected']}\n"
        f"Actual anomalies: {results['actual']}"
    )
    print(f"\n{summary}")
    output_path = os.path.join('tests', 'test_results', 'anomaly_detection_results.txt')
    with open(output_path, "w") as f:
        f.write(summary)
