"""
Anomaly Validation Agent

This module implements a validation agent that reviews detected anomalies
and classifies them based on patterns and historical feedback.
"""

import logging
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from .feedback_loop import FeedbackLoopBase

# Configure logging
logger = logging.getLogger(__name__)

class AnomalyValidationAgent(FeedbackLoopBase):
    """Agent that validates anomalies using ML models trained on historical feedback"""
    
    def __init__(self, config):
        """Initialize the validation agent
        
        Args:
            config: Configuration object with required settings
        """
        super().__init__(config)
        
        # Agent-specific attributes
        self.model_dir = Path(getattr(config, 'validation_model_directory', './validation_models'))
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Features used by the validation model
        self.numeric_features = [
            'anomaly_score', 'data_size', 'response_time', 'session_duration', 'attempts'
        ]
        self.categorical_features = [
            'anomaly_type', 'user_id', 'resource_id', 'action', 'ip_address'
        ]
        
        # Initialize models dictionary
        self.models = {}
        self.validation_threshold = 0.7  # Confidence threshold for validation
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize validation models"""
        # Try to load existing models
        if self._load_models():
            logger.info("Loaded existing validation models")
            return
            
        # Create new model if loading fails
        self.models['validation_classifier'] = {
            'model': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42
            ),
            'preprocessor': self._create_preprocessor(),
            'version': 1,
            'created_at': datetime.now().isoformat(),
            'trained': False,
            'training_samples': 0,
            'accuracy': 0.0
        }
        logger.info("Initialized new validation model")
        
    def _create_preprocessor(self):
        """Create a preprocessing pipeline for anomaly features"""
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
        
    def _load_models(self) -> bool:
        """Load models from disk"""
        model_path = self.model_dir / 'validation_classifier.pkl'
        try:
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models['validation_classifier'] = pickle.load(f)
                logger.info(f"Loaded validation model from {model_path}")
                return True
        except Exception as e:
            logger.error(f"Error loading validation model: {e}")
        return False
        
    def _save_models(self):
        """Save models to disk"""
        try:
            self.model_dir.mkdir(exist_ok=True, parents=True)
            
            for model_name, model_data in self.models.items():
                model_path = self.model_dir / f"{model_name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                    
            logger.info(f"Models saved to {self.model_dir}")
            return True
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def validate_anomaly(self, anomaly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an anomaly using the trained model
        
        Args:
            anomaly_data: Dictionary containing anomaly features
            
        Returns:
            Dictionary with validation results
        """
        # Default result with low confidence
        result = {
            'is_valid_anomaly': True,  # Default to true for safety
            'confidence': 0.5,
            'requires_human_review': True,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Check if we have a trained model
        if not self.models.get('validation_classifier', {}).get('trained', False):
            logger.warning("Validation model not trained, returning default validation")
            result['reason'] = "Model not trained"
            return result
            
        try:
            # Convert anomaly data to dataframe
            anomaly_df = pd.DataFrame([anomaly_data])
            
            # Ensure required features exist
            for feature in self.numeric_features:
                if feature not in anomaly_df:
                    anomaly_df[feature] = 0.0
                    
            for feature in self.categorical_features:
                if feature not in anomaly_df:
                    anomaly_df[feature] = 'unknown'
            
            # Get the model and preprocessor
            model_data = self.models['validation_classifier']
            model = model_data['model']
            preprocessor = model_data['preprocessor']
            
            # Preprocess the data
            processed_data = preprocessor.transform(anomaly_df)
            
            # Get prediction probability
            probs = model.predict_proba(processed_data)[0]
            
            # Find probability of true positive (class 1)
            if len(probs) >= 2:  # Binary classification [False, True]
                confidence = probs[1]  # Probability of class 1 (True)
            else:
                confidence = probs[0]  # If only one class, use that prob
            
            # Determine if it requires human review
            requires_review = confidence < self.validation_threshold
            
            # Update result
            result['is_valid_anomaly'] = confidence > 0.5  # More likely true than false
            result['confidence'] = float(confidence)
            result['requires_human_review'] = requires_review
            
            logger.info(f"Validated anomaly with confidence {confidence:.2f}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error validating anomaly: {e}")
            result['error'] = str(e)
            return result
    
    def train_from_feedback(self, max_samples: int = 10000) -> bool:
        """Train the validation model using collected feedback
        
        Args:
            max_samples: Maximum number of samples to use for training
            
        Returns:
            bool: Success of training
        """
        # Collect feedback data
        feedback_data = self._collect_feedback_data(max_samples)
        
        if len(feedback_data) < 10:  # Need at least 10 samples to train
            logger.warning(f"Not enough feedback data to train (found {len(feedback_data)})")
            return False
            
        # Convert to dataframe
        feedback_df = pd.DataFrame(feedback_data)
        
        # Prepare features and target
        X = feedback_df.drop(columns=['is_true_positive', 'anomaly_id', 'feedback_source',
                                     'timestamp', 'context'])
        y = feedback_df['is_true_positive']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        try:
            # Get model data
            model_data = self.models['validation_classifier']
            model = model_data['model']
            preprocessor = model_data['preprocessor']
            
            # Fit preprocessor on training data
            preprocessor.fit(X_train)
            
            # Transform data
            X_train_processed = preprocessor.transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            # Train the model
            model.fit(X_train_processed, y_train)
            
            # Evaluate
            accuracy = model.score(X_test_processed, y_test)
            
            # Update model metadata
            model_data['trained'] = True
            model_data['last_trained'] = datetime.now().isoformat()
            model_data['training_samples'] = len(X_train)
            model_data['accuracy'] = accuracy
            
            # Save the model
            self._save_models()
            
            # Update stats
            self.stats["model_updates"] += 1
            self.stats["last_update"] = datetime.now().isoformat()
            self.stats["accuracy"] = accuracy
            self._save_stats()
            
            logger.info(f"Trained validation model on {len(X_train)} samples with accuracy {accuracy:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training validation model: {e}")
            return False
    
    def _collect_feedback_data(self, max_samples: int = 10000) -> List[Dict[str, Any]]:
        """Collect feedback data from stored feedback files
        
        Args:
            max_samples: Maximum number of samples to collect
            
        Returns:
            List of feedback data dictionaries
        """
        feedback_data = []
        
        # Get all feedback files
        feedback_files = list(self.feedback_dir.glob("feedback_*.json"))
        
        # Limit the number of files to process
        if max_samples and len(feedback_files) > max_samples:
            feedback_files = feedback_files[:max_samples]
            
        # Process each file
        for file_path in feedback_files:
            try:
                with open(file_path, 'r') as f:
                    feedback = json.load(f)
                    
                # Combine feedback with its context
                feedback_entry = {**feedback.get('context', {})}
                feedback_entry['is_true_positive'] = feedback.get('is_true_positive', False)
                feedback_entry['anomaly_id'] = feedback.get('anomaly_id', '')
                feedback_entry['anomaly_type'] = feedback.get('anomaly_type', '')
                feedback_entry['feedback_source'] = feedback.get('feedback_source', '')
                feedback_entry['timestamp'] = feedback.get('timestamp', '')
                
                # Add to dataset
                feedback_data.append(feedback_entry)
                
            except Exception as e:
                logger.error(f"Error processing feedback file {file_path}: {e}")
                
        logger.info(f"Collected {len(feedback_data)} feedback entries for training")
        return feedback_data
