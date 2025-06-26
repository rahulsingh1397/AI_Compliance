"""
Feedback Integration Component

This module integrates feedback from human and automated validation
to retrain and improve the anomaly detection models.
"""

import logging
import json
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import tensorflow as tf

from .feedback_loop import FeedbackLoopBase
from .anomaly_detection import AnomalyDetectionModule as AnomalyDetector

# Configure logging
logger = logging.getLogger(__name__)

class FeedbackIntegration(FeedbackLoopBase):
    """Component that integrates feedback to improve anomaly detection models"""
    
    def __init__(self, config, anomaly_detector=None):
        """Initialize the feedback integration component
        
        Args:
            config: Configuration object with required settings
            anomaly_detector: Optional instance of AnomalyDetector
        """
        super().__init__(config)
        
        # Store reference to anomaly detector
        self.anomaly_detector = anomaly_detector
        
        # Integration settings
        self.retraining_interval_hours = getattr(config, 'retraining_interval_hours', 24)
        self.min_feedback_for_retraining = getattr(config, 'min_feedback_for_retraining', 50)
        self.last_retrain_time = None
        
        # Load last retrain time if available
        self._load_retrain_state()
        
    def _load_retrain_state(self):
        """Load retraining state from disk"""
        state_path = self.feedback_dir / 'retrain_state.json'
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                    self.last_retrain_time = state.get('last_retrain_time')
                logger.info(f"Loaded retrain state, last retrain: {self.last_retrain_time}")
            except Exception as e:
                logger.error(f"Error loading retrain state: {e}")
                
    def _save_retrain_state(self):
        """Save retraining state to disk"""
        state_path = self.feedback_dir / 'retrain_state.json'
        try:
            with open(state_path, 'w') as f:
                json.dump({
                    'last_retrain_time': self.last_retrain_time
                }, f)
        except Exception as e:
            logger.error(f"Error saving retrain state: {e}")
            
    def should_retrain(self) -> bool:
        """Check if models should be retrained based on time and feedback volume
        
        Returns:
            bool: True if retraining is recommended
        """
        # Check if we have enough feedback
        if self.stats.get('total_reviewed', 0) < self.min_feedback_for_retraining:
            logger.info(f"Not enough feedback for retraining ({self.stats.get('total_reviewed', 0)}/{self.min_feedback_for_retraining})")
            return False
            
        # Check if enough time has passed since last retraining
        if self.last_retrain_time:
            last_time = datetime.fromisoformat(self.last_retrain_time)
            hours_since_last = (datetime.now() - last_time).total_seconds() / 3600
            
            if hours_since_last < self.retraining_interval_hours:
                logger.info(f"Not enough time since last retraining ({hours_since_last:.1f}/{self.retraining_interval_hours} hours)")
                return False
                
        # If we reach here, we should retrain
        logger.info("Retraining recommended based on feedback volume and time interval")
        return True
        
    def collect_feedback_dataset(self) -> pd.DataFrame:
        """Collect feedback data and prepare a dataset for retraining
        
        Returns:
            DataFrame containing feedback data
        """
        feedback_data = []
        
        # Get all feedback files
        feedback_files = list(self.feedback_dir.glob("feedback_*.json"))
        
        # Process each file
        for file_path in feedback_files:
            try:
                with open(file_path, 'r') as f:
                    feedback = json.load(f)
                    
                # Add to dataset
                feedback_data.append(feedback)
                
            except Exception as e:
                logger.error(f"Error processing feedback file {file_path}: {e}")
                
        # Convert to dataframe
        if feedback_data:
            df = pd.DataFrame(feedback_data)
            logger.info(f"Collected {len(df)} feedback entries for retraining")
            return df
        else:
            logger.warning("No feedback data collected")
            return pd.DataFrame()
            
    def retrain_models(self) -> Dict[str, Any]:
        """Retrain anomaly detection models using feedback data
        
        Returns:
            Dictionary with retraining results
        """
        if self.anomaly_detector is None:
            logger.error("Cannot retrain models: no anomaly detector provided")
            return {'status': 'error', 'message': 'No anomaly detector provided'}
            
        results = {
            'status': 'started',
            'timestamp': datetime.now().isoformat(),
            'models_retrained': []
        }
        
        try:
            # Collect feedback dataset
            feedback_df = self.collect_feedback_dataset()
            
            if len(feedback_df) < self.min_feedback_for_retraining:
                logger.warning(f"Not enough feedback for retraining: {len(feedback_df)} < {self.min_feedback_for_retraining}")
                results['status'] = 'skipped'
                results['message'] = f"Insufficient feedback data: {len(feedback_df)} < {self.min_feedback_for_retraining}"
                return results
                
            # Get true positive and false positive examples
            true_positives = feedback_df[feedback_df['is_true_positive'] == True]
            false_positives = feedback_df[feedback_df['is_true_positive'] == False]
            
            logger.info(f"Retraining with {len(true_positives)} true positives and {len(false_positives)} false positives")
            
            # Adjust Isolation Forest model
            if 'iforest' in self.anomaly_detector.models:
                # Update contamination parameter based on feedback
                total_samples = len(true_positives) + len(false_positives)
                if total_samples > 0:
                    estimated_contamination = len(true_positives) / total_samples
                    # Bound contamination between reasonable values
                    new_contamination = min(0.5, max(0.01, estimated_contamination))
                    
                    # Get model data
                    model_data = self.anomaly_detector.models['iforest']
                    model = model_data['model']
                    
                    # Adjust contamination
                    if hasattr(model, 'set_params'):
                        model.set_params(contamination=new_contamination)
                        results['models_retrained'].append({
                            'model': 'iforest',
                            'adjustment': 'contamination',
                            'new_value': new_contamination
                        })
                        logger.info(f"Adjusted Isolation Forest contamination to {new_contamination}")
            
            # Adjust Autoencoder thresholds based on feedback
            if 'autoencoder' in self.anomaly_detector.models:
                model_data = self.anomaly_detector.models['autoencoder']
                
                # Extract context from feedback that contains reconstruction errors
                try:
                    # Get reconstruction errors from context
                    tp_errors = [f['context'].get('reconstruction_error', 0) 
                               for f in true_positives.to_dict('records')
                               if 'context' in f and 'reconstruction_error' in f['context']]
                    
                    fp_errors = [f['context'].get('reconstruction_error', 0) 
                               for f in false_positives.to_dict('records')
                               if 'context' in f and 'reconstruction_error' in f['context']]
                    
                    if tp_errors and fp_errors:
                        # Find optimal threshold that separates true/false positives
                        all_errors = np.array(tp_errors + fp_errors)
                        all_labels = np.array([1] * len(tp_errors) + [0] * len(fp_errors))
                        
                        # Try different thresholds and find best F1 score
                        best_f1 = 0
                        best_threshold = model_data['reconstruction_error_threshold']
                        
                        # Sort errors for efficient threshold testing
                        sorted_indices = np.argsort(all_errors)
                        sorted_errors = all_errors[sorted_indices]
                        sorted_labels = all_labels[sorted_indices]
                        
                        # Test each unique error value as a threshold
                        for i in range(len(sorted_errors)):
                            threshold = sorted_errors[i]
                            predictions = (all_errors >= threshold).astype(int)
                            
                            # Calculate precision and recall
                            tp = np.sum((predictions == 1) & (all_labels == 1))
                            fp = np.sum((predictions == 1) & (all_labels == 0))
                            fn = np.sum((predictions == 0) & (all_labels == 1))
                            
                            precision = tp / (tp + fp) if tp + fp > 0 else 0
                            recall = tp / (tp + fn) if tp + fn > 0 else 0
                            
                            # Calculate F1 score
                            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                            
                            if f1 > best_f1:
                                best_f1 = f1
                                best_threshold = threshold
                        
                        # Update threshold
                        if best_threshold != model_data['reconstruction_error_threshold']:
                            model_data['reconstruction_error_threshold'] = best_threshold
                            results['models_retrained'].append({
                                'model': 'autoencoder',
                                'adjustment': 'threshold',
                                'new_value': float(best_threshold),
                                'f1_score': float(best_f1)
                            })
                            logger.info(f"Adjusted Autoencoder threshold to {best_threshold} (F1={best_f1:.2f})")
                except Exception as e:
                    logger.error(f"Error adjusting Autoencoder threshold: {e}")
            
            # Save updated models
            self.anomaly_detector.save_models()
            
            # Update retrain state
            self.last_retrain_time = datetime.now().isoformat()
            self._save_retrain_state()
            
            # Update stats
            self.stats['model_updates'] += 1
            self.stats['last_update'] = self.last_retrain_time
            self._save_stats()
            
            results['status'] = 'completed'
            logger.info(f"Retraining completed: {len(results['models_retrained'])} models updated")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
            results['status'] = 'error'
            results['message'] = str(e)
            return results
            
    def update_models_from_feedback(self) -> Dict[str, Any]:
        """Check if retraining is needed and update models if necessary
        
        Returns:
            Dictionary with update status
        """
        if self.should_retrain():
            logger.info("Initiating model retraining based on feedback")
            result = self.retrain_models()
            return result
        else:
            logger.info("Model retraining not needed at this time")
            return {
                'status': 'skipped',
                'message': 'Retraining criteria not met',
                'last_retrain_time': self.last_retrain_time
            }
