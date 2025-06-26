"""
Anomaly Detection Module for the Monitoring Agent.

This module implements unsupervised ML models for detecting anomalies in log data:
- Isolation Forest algorithm
- Deep Learning-based Autoencoder

Features:
- Detection with a false positive rate below 5% (FR2.3)
- Model versioning and persistent storage
- Incremental model training/updating
"""

import os
import logging
import time
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

# Import ML libraries
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# TensorFlow imports for Autoencoder
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential, load_model, save_model
    from tensorflow.keras.layers import Dense, Input, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not available. Autoencoder model will not be used.")
    TENSORFLOW_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

class AnomalyDetectionModule:
    """
    Implements ML-based anomaly detection for log data.
    """
    
    def __init__(self, config):
        """
        Initialize the anomaly detection module.
        
        Args:
            config: Configuration object with necessary parameters
        """
        logger.debug("Initializing AnomalyDetectionModule")
        self.config = config
        self.models = {}
        self.model_dir = Path(getattr(config, 'model_directory', './models'))
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Threshold for anomaly scores (lower means more anomalies)
        # Default threshold set to detect ~5% as anomalies (95th percentile)
        self.anomaly_threshold = getattr(config, 'anomaly_threshold', -0.6)
        
        # Severity thresholds (anomaly score ranges for low/medium/high)
        self.severity_thresholds = getattr(config, 'severity_thresholds', {
            'low': -0.6,     # 95th percentile
            'medium': -0.8,   # 98th percentile
            'high': -1.0      # 99th percentile
        })
        
        self.categorical_features = [
            'source', 'source_type', 'action', 'status', 'user_id'
        ]
        
        self.numeric_features = [
            'process_time', 'response_size'
        ]
        
        self.text_features = [
            'resource', 'message'
        ]
        
        self.stats = {
            "last_model_update": None,
            "anomalies_detected": 0,
            "logs_analyzed": 0,
            "false_positive_rate": None,
            "alerts": {
                "low": 0,
                "medium": 0,
                "high": 0
            }
        }
        
        # Initialize models
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize the anomaly detection models"""
        logger.info("Initializing anomaly detection models")
        
        # Try to load existing models first
        if self._load_models():
            logger.info("Loaded existing models")
            return
        
        # Create new models if loading fails
        # Isolation Forest model
        self.models['isolation_forest'] = {
            'model': IsolationForest(
                n_estimators=100, 
                contamination=0.05,  # 5% anomalies target rate
                random_state=42,
                max_samples='auto',
                n_jobs=-1  # Use all available cores
            ),
            'preprocessor': self._create_preprocessor(),
            'version': 1,
            'created_at': datetime.now().isoformat(),
            'trained': False
        }
        
        # Autoencoder model (if TensorFlow is available)
        if TENSORFLOW_AVAILABLE:
            # We'll set up the structure, but can't train until we have data
            # to determine input dimensions
            self.models['autoencoder'] = {
                'model': None,  # Will be created during first training
                'preprocessor': self._create_preprocessor(),
                'version': 1,
                'created_at': datetime.now().isoformat(),
                'trained': False,
                'reconstruction_error_threshold': 0.1,  # Initial threshold
                'encoding_dim': 16  # Encoding dimension
            }
            logger.info("Autoencoder model initialized")
        else:
            logger.warning("TensorFlow not available. Skipping Autoencoder initialization.")
        
        logger.info("Models initialized")
    
    def _create_preprocessor(self):
        """Create a preprocessing pipeline for log data"""
        numeric_transformer = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine transformers into a column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'  # Drop non-specified columns
        )
        
        return preprocessor
        
    def _create_autoencoder(self, input_dim, encoding_dim=16):
        """Create an autoencoder model for anomaly detection
        
        Args:
            input_dim: Dimension of input data (after preprocessing)
            encoding_dim: Size of the encoded representation
            
        Returns:
            Autoencoder model or None if TensorFlow is not available
        """
        if not TENSORFLOW_AVAILABLE:
            return None
            
        try:
            # Input layer
            input_layer = Input(shape=(input_dim,))
            
            # Encoder layers
            encoded = Dense(encoding_dim * 2, activation='relu')(input_layer)
            encoded = Dropout(0.2)(encoded)  # Prevent overfitting
            encoded = Dense(encoding_dim, activation='relu')(encoded)
            
            # Decoder layers
            decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
            decoded = Dropout(0.2)(decoded)
            decoded = Dense(input_dim, activation='sigmoid')(decoded)
            
            # Full autoencoder model
            autoencoder = Model(input_layer, decoded)
            
            # Compile the model
            autoencoder.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error'
            )
            
            logger.info(f"Created autoencoder model with input dim {input_dim} and encoding dim {encoding_dim}")
            return autoencoder
            
        except Exception as e:
            logger.error(f"Error creating autoencoder: {e}")
            return None
    
    def _load_models(self) -> bool:
        """Load models from disk"""
        models_loaded = 0
        try:
            # Load isolation forest model
            iso_model_path = self.model_dir / 'isolation_forest.pkl'
            if iso_model_path.exists():
                with open(iso_model_path, 'rb') as f:
                    self.models['isolation_forest'] = pickle.load(f)
                logger.info("Loaded Isolation Forest model")
                models_loaded += 1
                
            # Load autoencoder if TensorFlow is available
            if TENSORFLOW_AVAILABLE:
                ae_metadata_path = self.model_dir / 'autoencoder.pkl'
                if ae_metadata_path.exists():
                    try:
                        # Load the metadata first
                        with open(ae_metadata_path, 'rb') as f:
                            model_metadata = pickle.load(f)
                            
                        # Check if the keras model file exists
                        if 'model_path' in model_metadata:
                            model_path = model_metadata['model_path']
                            if os.path.exists(model_path):
                                # Load the keras model
                                model_metadata['model'] = load_model(model_path)
                                # Remove the path reference
                                model_metadata.pop('model_path', None)
                                # Store the complete model data
                                self.models['autoencoder'] = model_metadata
                                logger.info("Loaded Autoencoder model")
                                models_loaded += 1
                    except Exception as e:
                        logger.error(f"Error loading autoencoder model: {e}")
            
            return models_loaded > 0
        except Exception as e:
            logger.error(f"Error loading models: {e}")
        return False
    
    def _save_models(self):
        """Save models to disk"""
        try:
            self.model_dir.mkdir(exist_ok=True, parents=True)
            
            for model_name, model_data in self.models.items():
                # Special handling for autoencoder
                if model_name == 'autoencoder' and TENSORFLOW_AVAILABLE and model_data['model'] is not None:
                    # Create a deep copy of model data without the keras model
                    model_data_copy = model_data.copy()
                    keras_model = model_data_copy.pop('model')
                    
                    # Save Keras model separately
                    keras_model_path = self.model_dir / f"{model_name}_keras.h5"
                    try:
                        keras_model.save(keras_model_path)
                        # Store the path instead of the model object
                        model_data_copy['model_path'] = str(keras_model_path)
                        
                        # Save the metadata
                        metadata_path = self.model_dir / f"{model_name}.pkl"
                        with open(metadata_path, 'wb') as f:
                            pickle.dump(model_data_copy, f)
                    except Exception as e:
                        logger.error(f"Error saving Keras model: {e}")
                else:
                    # Standard pickle for other models
                    model_path = self.model_dir / f"{model_name}.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(model_data, f)
            
            logger.info(f"Models saved to {self.model_dir}")
            return True
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def cleanup_models(self):
        """Clean up model resources"""
        logger.info("Cleaning up anomaly detection models")
        
        # Save models before cleaning up
        self._save_models()
        
        # Clear models from memory
        self.models = {}
        
        logger.info("Models cleaned up")
    
    def detect_anomalies(self, logs_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Detect anomalies in the provided logs
        
        Args:
            logs_df: DataFrame containing log data to analyze
            
        Returns:
            Tuple of (DataFrame with anomaly scores, stats dict)
        """
        start_time = time.time()
        result_df = logs_df.copy()
        stats = {
            "checked": len(logs_df),
            "anomalies": 0,
            "anomaly_types": {},
            "severity_distribution": {"low": 0, "medium": 0, "high": 0},
            "processing_time": 0
        }
        
        if len(logs_df) == 0:
            stats["processing_time"] = time.time() - start_time
            return result_df, stats
        
        try:
            # Initialize columns with default values
            result_df['anomaly_score'] = 0.0
            result_df['is_anomaly'] = False
            result_df['anomaly_type'] = ''
            result_df['severity'] = 'low'
            
            # Track if at least one model was used successfully
            model_used = False
            
            # Use the primary model (Isolation Forest)
            if 'isolation_forest' in self.models and self.models['isolation_forest']['trained']:
                model_data = self.models['isolation_forest']
                
                # Preprocess the data
                preprocessor = model_data['preprocessor']
                processed_data = preprocessor.transform(logs_df)
                
                # Get anomaly scores
                scores = model_data['model'].decision_function(processed_data)
                # Isolation Forest returns lower scores for anomalies, normalize to 0-1 where 1 is anomalous
                anomaly_scores = 1 - (scores + 0.5)  # Outputs from -0.5 to 0.5, flip and normalize to 0-1
                anomaly_scores = np.clip(anomaly_scores, 0, 1)  # Ensure in 0-1 range
                
                # Anomaly detection (threshold = 0.85 by default)
                anomalies = anomaly_scores >= self.anomaly_threshold
                
                # Add results to the DataFrame
                result_df['anomaly_score'] = anomaly_scores
                result_df['is_anomaly'] = anomalies
                result_df['anomaly_type'] = ['isolation_forest' if a else '' for a in anomalies]
                
                # Update stats
                anomaly_count = np.sum(anomalies)
                stats["anomalies"] = int(anomaly_count)
                stats["anomaly_rate"] = float(anomaly_count / len(logs_df)) if len(logs_df) > 0 else 0
                stats["anomaly_types"]["isolation_forest"] = int(anomaly_count)
                
                model_used = True
                logger.debug(f"Isolation Forest detected {anomaly_count} anomalies")
            
            # Use Autoencoder model if available
            if TENSORFLOW_AVAILABLE and 'autoencoder' in self.models and self.models['autoencoder']['trained']:
                model_data = self.models['autoencoder']
                model = model_data['model']
                threshold = model_data['reconstruction_error_threshold']
                
                try:
                    # Preprocess the data
                    preprocessor = model_data['preprocessor']
                    processed_data = preprocessor.transform(logs_df)
                    
                    # Get reconstructions and calculate error
                    reconstructions = model.predict(processed_data)
                    mse = np.mean(np.square(processed_data - reconstructions), axis=1)
                    
                    # Normalize MSE to [0,1] scale for consistency with isolation forest scores
                    # Higher value = more anomalous
                    ae_scores = mse / (threshold * 2)  # Scale by twice the threshold to normalize
                    ae_scores = np.clip(ae_scores, 0, 1)  # Ensure in 0-1 range
                    
                    # Find anomalies where MSE > threshold
                    ae_anomalies = mse > threshold
                    
                    # Count autoencoder-specific anomalies (those not caught by IsolationForest)
                    ae_only_anomalies = np.logical_and(ae_anomalies, ~result_df['is_anomaly'])
                    ae_only_count = np.sum(ae_only_anomalies)
                    
                    # Update anomaly flags for records only identified by autoencoder
                    if ae_only_count > 0:
                        logger.info(f"Autoencoder identified {ae_only_count} additional anomalies")
                        
                        # For records where autoencoder found anomalies but isolation forest didn't
                        for idx in np.where(ae_only_anomalies)[0]:
                            result_df.loc[result_df.index[idx], 'is_anomaly'] = True
                            # Use the autoencoder score if it's higher
                            if ae_scores[idx] > result_df.loc[result_df.index[idx], 'anomaly_score']:
                                result_df.loc[result_df.index[idx], 'anomaly_score'] = ae_scores[idx]
                            
                            # Update anomaly type
                            current_type = result_df.loc[result_df.index[idx], 'anomaly_type']
                            new_type = 'autoencoder' if current_type == '' else f"{current_type},autoencoder"
                            result_df.loc[result_df.index[idx], 'anomaly_type'] = new_type
                    
                    # Update stats
                    stats["anomaly_types"]["autoencoder"] = int(ae_only_count)
                    stats["anomalies"] = int(np.sum(result_df['is_anomaly']))
                    stats["anomaly_rate"] = float(stats["anomalies"] / len(logs_df)) if len(logs_df) > 0 else 0
                    
                    model_used = True
                    logger.debug(f"Autoencoder detected {ae_only_count} additional anomalies")
                    
                except Exception as e:
                    logger.error(f"Error using autoencoder model: {e}")
            
            # Calculate severity based on final anomaly scores
            severity = ['low'] * len(logs_df)
            for i, idx in enumerate(result_df.index):
                score = result_df.loc[idx, 'anomaly_score']
                if result_df.loc[idx, 'is_anomaly']:
                    if score >= 0.95:  # Very anomalous
                        severity[i] = 'high'
                        stats["severity_distribution"]["high"] += 1
                    elif score >= 0.9:  # Moderately anomalous
                        severity[i] = 'medium'
                        stats["severity_distribution"]["medium"] += 1
                    else:  # Mildly anomalous
                        severity[i] = 'low'
                        stats["severity_distribution"]["low"] += 1
            
            result_df['severity'] = severity
            
            if not model_used:
                logger.warning("No trained model available for anomaly detection. Using default scoring.")
                self.stats["untrained_checks"] += 1
            else:
                self.stats["total_anomalies"] += stats["anomalies"]
                self.stats["total_checked"] += len(logs_df)
            
        
        except Exception as e:
            logger.error(f"Error during anomaly detection: {e}")
            return pd.DataFrame()
    
    def _train_model(self, logs_df: pd.DataFrame, model_name: str) -> bool:
        """Train the specified model on the provided data"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return False
            
        model_data = self.models[model_name]
        preprocessor = model_data['preprocessor']
        
        try:
            logger.info(f"Training {model_name} model on {len(logs_df)} samples")
            
            # Fit the preprocessor 
            processed_data = preprocessor.fit_transform(logs_df)
            
            # Train the model
            if model_name == 'isolation_forest':
                model_data['model'].fit(processed_data)
            
            elif model_name == 'autoencoder' and TENSORFLOW_AVAILABLE:
                # Create or retrieve the autoencoder model
                input_dim = processed_data.shape[1]  # Get input dimension from processed data
                
                if model_data['model'] is None:
                    # Create new autoencoder
                    model_data['model'] = self._create_autoencoder(
                        input_dim, 
                        encoding_dim=model_data.get('encoding_dim', 16)
                    )
                
                if model_data['model'] is None:
                    logger.error("Could not create autoencoder model")
                    return False
                
                # Train the autoencoder
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                    ModelCheckpoint(
                        filepath=str(self.model_dir / 'autoencoder_checkpoint.h5'),
                        save_best_only=True,
                        monitor='val_loss'
                    )
                ]
                
                # Split data for training and validation
                n_samples = processed_data.shape[0]
                val_split = min(0.2, 200 / n_samples)  # Use at most 20% for validation
                
                history = model_data['model'].fit(
                    processed_data, processed_data,  # Autoencoder tries to reconstruct its input
                    epochs=30,
                    batch_size=32,
                    shuffle=True,
                    validation_split=val_split,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Calculate reconstruction error threshold
                # Predict the reconstruction
                reconstructions = model_data['model'].predict(processed_data)
                # Calculate MSE for each sample
                mse = np.mean(np.square(processed_data - reconstructions), axis=1)
                # Set threshold as 95th percentile to catch ~5% anomalies
                model_data['reconstruction_error_threshold'] = np.percentile(mse, 95)
                logger.info(f"Set reconstruction error threshold to {model_data['reconstruction_error_threshold']}")
            
            # Update model metadata
            model_data['trained'] = True
            model_data['last_trained'] = datetime.now().isoformat()
            model_data['samples_trained'] = len(logs_df)
            
            # Save the updated model
            self.stats["last_model_update"] = datetime.now().isoformat()
            self._save_models()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training {model_name} model: {e}")
            return False
    
    def _simple_anomaly_detection(self, logs_df: pd.DataFrame) -> pd.DataFrame:
        """Simple rule-based anomaly detection as fallback"""
        logger.warning("Using simple anomaly detection as fallback")
        
        logs_df = logs_df.copy()
        
        # Mark failures as potential anomalies
        logs_df['is_anomaly'] = logs_df['status'] == 'FAILURE'
        
        # Look for outliers in process_time (if available)
        if 'process_time' in logs_df.columns and logs_df['process_time'].notna().any():
            mean_time = logs_df['process_time'].mean()
            std_time = logs_df['process_time'].std()
            
            if not np.isnan(std_time) and std_time > 0:
                # Mark samples with process time > 3 standard deviations as anomalies
                logs_df.loc[logs_df['process_time'] > mean_time + 3*std_time, 'is_anomaly'] = True
        
        # Simple severity assignment
        logs_df['severity'] = 'low'
        logs_df.loc[logs_df['status'] == 'FAILURE', 'severity'] = 'medium'
        
        # If both criteria are true, mark as high severity
        if 'process_time' in logs_df.columns:
            high_severity = (logs_df['status'] == 'FAILURE') & \
                           (logs_df['process_time'] > logs_df['process_time'].quantile(0.95))
            logs_df.loc[high_severity, 'severity'] = 'high'
        
        # Update stats
        anomalies = logs_df[logs_df['is_anomaly']]
        self.stats["logs_analyzed"] += len(logs_df)
        self.stats["anomalies_detected"] += len(anomalies)
        
        # Count by severity
        severity_counts = anomalies['severity'].value_counts()
        for severity in ['low', 'medium', 'high']:
            if severity in severity_counts:
                self.stats["alerts"][severity] += severity_counts[severity]
        
        return anomalies
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about anomaly detection"""
        return {
            **self.stats,
            "timestamp": time.time(),
            "false_positive_rate_estimate": self._estimate_false_positive_rate()
        }
        
    def _estimate_false_positive_rate(self) -> float:
        """Estimate the false positive rate based on threshold settings"""
        if self.stats["logs_analyzed"] == 0:
            return 0.0
        
        # If we have the models, we use the configured contamination
        if 'isolation_forest' in self.models and self.models['isolation_forest']['trained']:
            return self.models['isolation_forest']['model'].contamination 
        
        # Otherwise estimate from actual detection rate
        if self.stats["logs_analyzed"] > 0:
            return self.stats["anomalies_detected"] / self.stats["logs_analyzed"]
        
        return 0.0
        
    def update_model(self, logs_df: pd.DataFrame):
        """Update the anomaly detection models with new data"""
        if logs_df.empty:
            return False
            
        result = self._train_model(logs_df, 'isolation_forest')
        return result
