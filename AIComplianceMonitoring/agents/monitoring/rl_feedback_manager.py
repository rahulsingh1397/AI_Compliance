"""
Reinforcement Learning Feedback Manager

This module provides the main interface for the multi-agent reinforcement learning
feedback system, coordinating between the validation agent, human interaction agent,
and feedback integration components.
"""

import logging
import json
import os
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

from .validation_agent import AnomalyValidationAgent
from .human_interaction_agent import HumanInteractionAgent
from .feedback_integration import FeedbackIntegration
from .anomaly_detection import AnomalyDetectionModule

# Configure logging
logger = logging.getLogger(__name__)

class RLFeedbackManager:
    """Manager class for the reinforcement learning feedback system"""
    
    def __init__(self, config, anomaly_detector=None):
        """Initialize the feedback manager
        
        Args:
            config: Configuration object with required settings
            anomaly_detector: Optional instance of AnomalyDetectionModule
        """
        self.config = config
        self.anomaly_detector = anomaly_detector
        
        # Initialize components
        self.validation_agent = AnomalyValidationAgent(config)
        self.human_interaction = HumanInteractionAgent(config)
        self.feedback_integration = FeedbackIntegration(config, anomaly_detector)
        
        # Auto-update thread
        self.auto_update_interval = getattr(config, 'auto_update_interval_hours', 24)
        self.auto_update_thread = None
        self.stop_auto_update = threading.Event()
        
        # Stats
        self.stats = {
            "anomalies_processed": 0,
            "anomalies_validated": 0,
            "anomalies_queued_for_review": 0,
            "feedback_collected": 0,
            "model_updates": 0,
            "last_activity": datetime.now().isoformat()
        }
    
    def process_anomaly(self, anomaly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a detected anomaly through the feedback loop
        
        Args:
            anomaly_data: Dictionary containing anomaly data
            
        Returns:
            Dictionary with processing results
        """
        result = {
            'anomaly_id': anomaly_data.get('id', str(time.time())),
            'status': 'processed',
            'requires_review': False,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Update stats
            self.stats["anomalies_processed"] += 1
            self.stats["last_activity"] = datetime.now().isoformat()
            
            # Step 1: Validate the anomaly
            validation_result = self.validation_agent.validate_anomaly(anomaly_data)
            self.stats["anomalies_validated"] += 1
            
            # Add validation results to the response
            result['validation'] = {
                'is_valid_anomaly': validation_result.get('is_valid_anomaly', True),
                'confidence': validation_result.get('confidence', 0.5)
            }
            
            # Step 2: Determine if human review is needed
            requires_review = validation_result.get('requires_human_review', True)
            result['requires_review'] = requires_review
            
            # Step 3: If review is required, queue it
            if requires_review:
                queue_result = self.human_interaction.queue_anomaly_for_review(
                    anomaly_data, validation_result
                )
                result['review_queue'] = queue_result
                self.stats["anomalies_queued_for_review"] += 1
                
            logger.info(f"Processed anomaly {result['anomaly_id']} - Requires review: {requires_review}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing anomaly: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
            return result
    
    def get_next_review_item(self, max_priority: str = 'high') -> Optional[Dict[str, Any]]:
        """Get the next item for human review
        
        Args:
            max_priority: Maximum priority to consider ('high', 'medium', 'low')
            
        Returns:
            Review item or None if queue is empty
        """
        return self.human_interaction.get_next_review_item(max_priority)
    
    def submit_feedback(self, review_id: str, is_true_positive: bool, 
                       notes: str = None) -> Dict[str, Any]:
        """Submit human feedback for a reviewed anomaly
        
        Args:
            review_id: ID of the review item
            is_true_positive: Whether the anomaly is a true positive
            notes: Optional notes from the reviewer
            
        Returns:
            Status dictionary
        """
        result = self.human_interaction.submit_feedback(review_id, is_true_positive, notes)
        
        if result.get('status') == 'success':
            self.stats["feedback_collected"] += 1
            
            # Check if we should update models
            if self.feedback_integration.should_retrain():
                threading.Thread(
                    target=self.update_models,
                    daemon=True
                ).start()
                
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all components
        
        Returns:
            Combined statistics dictionary
        """
        stats = self.stats.copy()
        
        # Add validation stats
        validation_stats = self.validation_agent.get_feedback_stats()
        stats['validation'] = validation_stats
        
        # Add review queue stats
        queue_stats = self.human_interaction.get_review_queue_stats()
        stats['review_queue'] = queue_stats
        
        # Add model update info
        if hasattr(self.feedback_integration, 'last_retrain_time'):
            stats['last_model_update'] = self.feedback_integration.last_retrain_time
            
        return stats
    
    def update_models(self) -> Dict[str, Any]:
        """Trigger model update from feedback
        
        Returns:
            Update status dictionary
        """
        result = self.feedback_integration.update_models_from_feedback()
        
        if result.get('status') == 'completed':
            self.stats["model_updates"] += 1
            
        return result
    
    def start_auto_update(self):
        """Start background thread for automatic model updates"""
        if self.auto_update_thread is not None and self.auto_update_thread.is_alive():
            logger.warning("Auto-update thread is already running")
            return False
            
        self.stop_auto_update.clear()
        self.auto_update_thread = threading.Thread(
            target=self._auto_update_worker,
            daemon=True
        )
        self.auto_update_thread.start()
        logger.info(f"Started auto-update thread (interval: {self.auto_update_interval} hours)")
        return True
    
    def stop_auto_update_thread(self):
        """Stop the auto-update thread"""
        if self.auto_update_thread is not None and self.auto_update_thread.is_alive():
            self.stop_auto_update.set()
            self.auto_update_thread.join(timeout=5)
            logger.info("Stopped auto-update thread")
            return True
        return False
    
    def _auto_update_worker(self):
        """Background worker that periodically updates models"""
        logger.info("Auto-update worker started")
        
        while not self.stop_auto_update.is_set():
            try:
                # Check if update is needed
                if self.feedback_integration.should_retrain():
                    logger.info("Auto-update worker triggering model update")
                    self.update_models()
                
                # Sleep for the configured interval
                for _ in range(int(self.auto_update_interval * 3600)):
                    if self.stop_auto_update.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in auto-update worker: {e}")
                # Sleep for 1 hour before retrying after error
                for _ in range(3600):
                    if self.stop_auto_update.is_set():
                        break
                    time.sleep(1)
                
        logger.info("Auto-update worker stopped")
        
    def cleanup(self):
        """Perform cleanup operations"""
        # Stop auto-update thread
        self.stop_auto_update_thread()
        
        # Clean old items from review queue
        self.human_interaction.clean_review_queue(max_age_days=30)
