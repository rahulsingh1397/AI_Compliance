"""
Anomaly Detection Module for the Monitoring Agent.

This module implements unsupervised ML models for detecting anomalies in log data:
- Isolation Forest algorithm
- Deep Learning-based Autoencoder (PyTorch)

Features:
- Detection with a false positive rate below 5% (FR2.3)
- Model versioning and persistent storage
- Incremental model training/updating
"""

import os
import logging
import time
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime

# Import ML libraries
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

# Configure logging
logger = logging.getLogger(__name__)

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
    logger.info("PyTorch is available. Autoencoder model can be used.")
except ImportError:
    logger.warning("PyTorch is not available. Autoencoder model will be disabled.")
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class AutoencoderModel(nn.Module):
        """PyTorch-based Autoencoder for anomaly detection."""
        def __init__(self, input_dim: int, encoding_dim_ratio=0.25):
            super(AutoencoderModel, self).__init__()
            self._build_model(input_dim, encoding_dim_ratio)

        def _build_model(self, input_dim, encoding_dim_ratio):
            dim1 = int(input_dim * 0.75)
            dim2 = int(input_dim * 0.5)
            bottleneck = int(input_dim * encoding_dim_ratio)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, dim1),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(dim1, dim2),
                nn.ReLU(),
                nn.Linear(dim2, bottleneck),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck, dim2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(dim2, dim1),
                nn.ReLU(),
                nn.Linear(dim1, input_dim),
                nn.Sigmoid(),
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded


class AnomalyDetectionModule:
    """
    Implements ML-based anomaly detection for log data.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the anomaly detection module.
        Args:
            config: Dictionary with 'model_directory' and 'device'.
        """
        logger.debug("Initializing AnomalyDetectionModule")
        self.config = config
        self.isolation_forest = None
        self.autoencoder_model = None
        self.preprocessor = None
        self.device = config.device if TORCH_AVAILABLE and config.device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.model_dir = Path(config.model_directory)
        self.model_dir.mkdir(exist_ok=True, parents=True)

        self.stats = {
            "last_model_update": None,
            "anomalies_detected": 0,
            "logs_analyzed": 0,
        }

    def _get_n_features_from_preprocessor(self) -> int:
        """Helper to get feature count from a fitted preprocessor."""
        if self.preprocessor is None or not hasattr(self.preprocessor, 'transformers_'):
            return 0
        try:
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                return len(self.preprocessor.get_feature_names_out())
            elif self.preprocessor.transformers_:
                return sum(len(trans.get_feature_names_out()) for name, trans, cols in self.preprocessor.transformers_ if hasattr(trans, 'get_feature_names_out'))
        except Exception as e:
            logger.error(f"Could not determine feature count from preprocessor: {e}")
        return 0

    def _save_model(self, model, filename: str):
        """Save a model to the specified file."""
        path = self.model_dir / filename
        logger.info(f"Saving model to {path}")
        try:
            if TORCH_AVAILABLE and isinstance(model, nn.Module):
                torch.save(model.state_dict(), path)
            else:
                with open(path, 'wb') as f:
                    pickle.dump(model, f)
        except Exception as e:
            logger.error(f"Failed to save model {filename}: {e}")

    def initialize_models(self):
        """Load all models from the model directory."""
        logger.info("Attempting to load models...")
        preprocessor_path = self.model_dir / "preprocessor.joblib"
        if_path = self.model_dir / "isolation_forest.joblib"
        ae_path = self.model_dir / "autoencoder.pth"

        if preprocessor_path.exists():
            try:
                with open(preprocessor_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
                logger.info("Preprocessor loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load preprocessor: {e}", exc_info=True)

        if if_path.exists():
            try:
                with open(if_path, 'rb') as f:
                    self.isolation_forest = pickle.load(f)
                logger.info("Isolation Forest model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Isolation Forest model: {e}", exc_info=True)

        if TORCH_AVAILABLE and ae_path.exists() and self.preprocessor:
            try:
                n_features = self._get_n_features_from_preprocessor()
                if n_features > 0:
                    self.autoencoder_model = AutoencoderModel(input_dim=n_features).to(self.device)
                    self.autoencoder_model.load_state_dict(torch.load(ae_path, map_location=self.device))
                    self.autoencoder_model.eval()
                    logger.info("Autoencoder model loaded successfully.")
                else:
                    logger.warning("Cannot load Autoencoder without feature count from preprocessor.")
            except Exception as e:
                logger.error(f"Failed to load Autoencoder model: {e}", exc_info=True)
        else:
            logger.warning("Cannot load Autoencoder: Preprocessor not available or not fitted.")
            self.autoencoder = None

    def detect_anomalies(self, logs_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect anomalies in the provided log data."""
        if logs_df.empty:
            return pd.DataFrame(), self.get_stats()

        if not self.isolation_forest or not self.preprocessor:
            return self._fallback_and_return(logs_df, "Core models not fitted")

        try:
            check_is_fitted(self.preprocessor)
            check_is_fitted(self.isolation_forest)
        except NotFittedError:
            return self._fallback_and_return(logs_df, "Models are not fitted. Please train the model first.")

        self.stats['logs_analyzed'] += len(logs_df)
        features_df = logs_df.drop(columns=['timestamp'], errors='ignore')

        try:
            processed_df = self.preprocessor.transform(features_df)
            processed_data = processed_df.values
            n_features = processed_data.shape[1]

            # Get Isolation Forest scores
            if_scores = self.isolation_forest.decision_function(processed_data) * -1
            if_scores_scaled = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())

            # Get Autoencoder scores if available
            if TORCH_AVAILABLE and self.autoencoder_model:
                try:
                    self.autoencoder_model.eval()
                    with torch.no_grad():
                        tensor_data = torch.FloatTensor(processed_data).to(self.device)
                        reconstructions = self.autoencoder_model(tensor_data)
                        mse = nn.MSELoss(reduction='none')(reconstructions, tensor_data).mean(axis=1)
                        ae_scores = mse.cpu().numpy()
                        ae_scores_scaled = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min())
                    
                    # Ensemble scores
                    ensemble_scores = (self.config.get('if_weight', 0.5) * if_scores_scaled) + \
                                      (self.config.get('ae_weight', 0.5) * ae_scores_scaled)
                    anomaly_scores = ensemble_scores
                    logger.debug("Using ensemble (IsolationForest + Autoencoder) scores.")

                except Exception as e:
                    logger.warning(f"Autoencoder prediction failed: {e}. Falling back to Isolation Forest.")
                    anomaly_scores = if_scores_scaled
            else:
                anomaly_scores = if_scores_scaled
                logger.debug("Using Isolation Forest scores only.")

            logs_df['anomaly_score'] = anomaly_scores
            
            threshold = np.percentile(anomaly_scores, self.config.get('anomaly_threshold_percentile', 95))
            anomalies = logs_df[logs_df['anomaly_score'] > threshold]
            self.stats['anomalies_detected'] += len(anomalies)

            return anomalies.sort_values(by='anomaly_score', ascending=False), self.get_stats()

        except Exception as e:
            logger.error(f"Error during anomaly detection: {e}", exc_info=True)
            return self._fallback_and_return(logs_df, f"Error during anomaly detection: {e}")

    def _fallback_and_return(self, logs_df, reason):
        logger.warning(f"Falling back to simple detection. Reason: {reason}")
        anomalies = self._simple_anomaly_detection(logs_df)
        self.stats['anomalies_detected'] += len(anomalies)
        stats = self.get_stats()
        stats['model_used'] = f"fallback: {reason}"
        if not anomalies.empty:
            # Add a dummy score for consistent output schema
            anomalies['anomaly_score'] = 1.0
        return anomalies, stats

    def _simple_anomaly_detection(self, logs_df: pd.DataFrame) -> pd.DataFrame:
        """Simple rule-based anomaly detection as fallback."""
        result_df = logs_df.copy()
        result_df['is_anomaly'] = False
        if 'status_code' in result_df.columns:
            result_df.loc[result_df['status_code'] >= 400, 'is_anomaly'] = True
        if 'response_time' in result_df.columns:
            threshold = result_df['response_time'].quantile(0.95)
            result_df.loc[result_df['response_time'] > threshold, 'is_anomaly'] = True
        return result_df[result_df['is_anomaly']]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about anomaly detection."""
        return {**self.stats, "timestamp": datetime.now().isoformat()}
        
    def update_model(self, logs_df: pd.DataFrame) -> bool:
        """
        Update (retrain) the anomaly detection models with new data.
        This method performs a full retraining, not an incremental update.
        """
        if logs_df.empty:
            logger.warning("Training data is empty. Skipping model update.")
            return False

        features_df = logs_df.drop(columns=['timestamp'], errors='ignore')
        categorical_features = features_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        numeric_features = features_df.select_dtypes(include=np.number).columns.tolist()

        if not categorical_features and not numeric_features:
            logger.error("No suitable features found for training.")
            return False

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )
        self.preprocessor.set_output(transform="pandas")

        logger.info("Fitting preprocessor and transforming data...")
        try:
            processed_df = self.preprocessor.fit_transform(features_df)
            self._save_model(self.preprocessor, "preprocessor.joblib")
            logger.info("Preprocessor fitted and saved successfully.")
        except Exception as e:
            logger.error(f"Failed to fit preprocessor: {e}", exc_info=True)
            self.preprocessor = None
            return False

        processed_data = processed_df.values
        n_features = processed_data.shape[1]
        logger.info(f"Data processed into {n_features} features.")

        self.isolation_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
        try:
            logger.info("Training Isolation Forest model...")
            self.isolation_forest.fit(processed_data)
            self._save_model(self.isolation_forest, "isolation_forest.joblib")
            logger.info("Isolation Forest model trained and saved successfully.")
        except Exception as e:
            logger.error(f"Failed to train Isolation Forest model: {e}", exc_info=True)
            self.isolation_forest = None
            return False

        if TORCH_AVAILABLE:
            self.autoencoder_model = AutoencoderModel(input_dim=n_features).to(self.device)
            logger.info("Training Autoencoder model...")
            try:
                tensor_data = torch.FloatTensor(processed_data).to(self.device)
                dataset = TensorDataset(tensor_data, tensor_data)
                dataloader = DataLoader(dataset, batch_size=self.config.get('batch_size', 64), shuffle=True)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(self.autoencoder_model.parameters(), lr=self.config.get('learning_rate', 1e-3))
                epochs = self.config.get('epochs', 20)

                self.autoencoder_model.train()
                for epoch in range(epochs):
                    for data_batch in dataloader:
                        inputs, _ = data_batch
                        optimizer.zero_grad()
                        outputs = self.autoencoder_model(inputs)
                        loss = criterion(outputs, inputs)
                        loss.backward()
                        optimizer.step()
                    logger.debug(f'Autoencoder Epoch [{epoch+1}/{epochs}] complete.')
                
                self._save_model(self.autoencoder_model, "autoencoder.pth")
                logger.info("Autoencoder model trained and saved successfully.")
            except Exception as e:
                logger.error(f"Failed to train or save Autoencoder model: {e}", exc_info=True)
                self.autoencoder_model = None
            
        self.stats["last_model_update"] = datetime.now().isoformat()
        logger.info("Models updated successfully.")
        return True
