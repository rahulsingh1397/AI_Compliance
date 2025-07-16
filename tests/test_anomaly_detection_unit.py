import unittest
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest

# Temporarily add the project root to the Python path to resolve imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AIComplianceMonitoring.agents.monitoring.anomaly_detection import AnomalyDetectionModule, Autoencoder

class TestConfig:
    """Configuration for the unit test."""
    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'test_results', 'anomaly_detection_models')
    N_SAMPLES = 1000
    N_FEATURES = 10
    ANOMALY_FRACTION = 0.05
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TestAnomalyDetectionModule(unittest.TestCase):
    """Unit tests for the AnomalyDetectionModule."""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment, creating a directory for models."""
        os.makedirs(TestConfig.MODEL_DIR, exist_ok=True)
        cls.config = {'model_directory': TestConfig.MODEL_DIR, 'device': TestConfig.DEVICE}
        cls.module = AnomalyDetectionModule(cls.config)

    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment by removing the model directory."""
        if os.path.exists(TestConfig.MODEL_DIR):
            shutil.rmtree(TestConfig.MODEL_DIR)

    def _generate_synthetic_data(self):
        """Generates synthetic data with a mix of normal and anomalous samples."""
        rng = np.random.RandomState(42)
        
        # Generate normal data
        n_normal = int(TestConfig.N_SAMPLES * (1 - TestConfig.ANOMALY_FRACTION))
        normal_data = rng.randn(n_normal, TestConfig.N_FEATURES)
        
        # Generate anomalous data
        n_anomalies = TestConfig.N_SAMPLES - n_normal
        anomalous_data = rng.uniform(low=-4, high=4, size=(n_anomalies, TestConfig.N_FEATURES))
        
        # Combine and shuffle
        data = np.vstack([normal_data, anomalous_data])
        labels = np.array([1] * n_normal + [-1] * n_anomalies) # 1 for normal, -1 for anomaly
        
        permutation = rng.permutation(len(data))
        return data[permutation], labels[permutation]

    def test_01_initialization(self):
        """Test if the module and its models initialize correctly."""
        self.assertIsNone(self.module.isolation_forest)
        self.assertIsNone(self.module.autoencoder)
        self.module.initialize_models(n_features=TestConfig.N_FEATURES)
        self.assertIsInstance(self.module.isolation_forest, IsolationForest)
        self.assertIsInstance(self.module.autoencoder, Autoencoder)
        self.assertIn(str(TestConfig.DEVICE), str(self.module.autoencoder.device)) # Check device placement

    def test_02_training(self):
        """Test the training process for both models."""
        data, _ = self._generate_synthetic_data()
        self.module.initialize_models(n_features=data.shape[1])
        
        # Ensure training runs without errors
        try:
            self.module.train_models(data)
        except Exception as e:
            self.fail(f"train_models() raised an exception unexpectedly: {e}")

    def test_03_model_persistence(self):
        """Test saving and loading models."""
        data, _ = self._generate_synthetic_data()
        self.module.initialize_models(n_features=data.shape[1])
        self.module.train_models(data)

        # Save models
        self.module._save_model(self.module.isolation_forest, 'isolation_forest.joblib')
        self.module._save_model(self.module.autoencoder, 'autoencoder.pth')

        # Check if files exist
        self.assertTrue(os.path.exists(os.path.join(TestConfig.MODEL_DIR, 'isolation_forest.joblib')))
        self.assertTrue(os.path.exists(os.path.join(TestConfig.MODEL_DIR, 'autoencoder.pth')))

        # Load models into a new instance
        new_module = AnomalyDetectionModule(self.config)
        new_module.initialize_models(n_features=data.shape[1])
        new_module._load_models()

        self.assertIsNotNone(new_module.isolation_forest)
        self.assertIsNotNone(new_module.autoencoder)

    def test_04_anomaly_detection(self):
        """Test the anomaly detection process and its output."""
        data, labels = self._generate_synthetic_data()
        self.module.initialize_models(n_features=data.shape[1])
        self.module.train_models(data)

        results = self.module.detect_anomalies(data)

        self.assertIn('indices', results)
        self.assertIn('scores', results)
        self.assertIn('is_anomaly', results)
        self.assertIn('metadata', results)
        self.assertEqual(len(results['is_anomaly']), len(data))

        # Check if at least some anomalies are detected
        detected_anomalies = np.sum(results['is_anomaly'])
        self.assertGreater(detected_anomalies, 0, "Should detect at least one anomaly")

        # A simple check on accuracy: more than 50% of true anomalies should be flagged
        true_anomalies = np.where(labels == -1)[0]
        detected_as_anomalies = results['indices'][results['is_anomaly']]
        correctly_detected = np.intersect1d(true_anomalies, detected_as_anomalies)
        
        self.assertGreater(len(correctly_detected) / len(true_anomalies), 0.5, "Detection accuracy should be reasonable")

if __name__ == '__main__':
    unittest.main(verbosity=2)
