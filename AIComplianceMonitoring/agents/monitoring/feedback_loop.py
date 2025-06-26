"""
Reinforcement Learning Feedback Loop for Anomaly Detection

This module implements a multi-agent feedback system that:
1. Reviews detected anomalies through a validation agent
2. Interacts with humans to confirm true/false positives
3. Incorporates feedback to improve anomaly detection over time

The feedback loop aims to continuously improve the detection accuracy
while maintaining a false positive rate below 5% (FR2.3).
"""

import logging
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

class FeedbackLoopBase:
    """Base class for the reinforcement learning feedback loop system"""
    
    def __init__(self, config):
        """Initialize the feedback loop
        
        Args:
            config: Configuration object with required settings
        """
        self.config = config
        self.feedback_dir = Path(getattr(config, 'feedback_directory', './feedback_data'))
        self.feedback_dir.mkdir(exist_ok=True, parents=True)
        
        # Statistics tracking
        self.stats = {
            "total_reviewed": 0,
            "true_positives": 0,
            "false_positives": 0,
            "feedback_incorporated": 0,
            "model_updates": 0,
            "last_update": None,
            "accuracy": 0.0
        }
        
        # Load previous stats if they exist
        self._load_stats()
        
    def _load_stats(self):
        """Load statistics from disk if available"""
        stats_file = self.feedback_dir / 'feedback_stats.json'
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    self.stats = json.load(f)
                logger.info(f"Loaded feedback statistics: {self.stats['total_reviewed']} reviews")
            except Exception as e:
                logger.error(f"Error loading feedback statistics: {e}")
    
    def _save_stats(self):
        """Save current statistics to disk"""
        stats_file = self.feedback_dir / 'feedback_stats.json'
        try:
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f)
        except Exception as e:
            logger.error(f"Error saving feedback statistics: {e}")
    
    def record_feedback(self, anomaly_id: str, is_true_positive: bool, 
                       anomaly_type: str, feedback_source: str, 
                       context: Dict[str, Any] = None) -> bool:
        """Record feedback for an anomaly
        
        Args:
            anomaly_id: Unique identifier for the anomaly
            is_true_positive: Whether this was a true positive (True) or false positive (False)
            anomaly_type: Type of anomaly (e.g., 'unauthorized_access', 'unusual_transfer')
            feedback_source: Source of feedback (e.g., 'human_expert', 'automatic')
            context: Additional contextual information about the feedback
            
        Returns:
            bool: Success of the operation
        """
        if context is None:
            context = {}
            
        feedback = {
            "anomaly_id": anomaly_id,
            "is_true_positive": is_true_positive,
            "anomaly_type": anomaly_type, 
            "feedback_source": feedback_source,
            "timestamp": datetime.now().isoformat(),
            "context": context
        }
        
        # Save feedback to disk
        try:
            feedback_file = self.feedback_dir / f"feedback_{anomaly_id}.json"
            with open(feedback_file, 'w') as f:
                json.dump(feedback, f)
                
            # Update statistics
            self.stats["total_reviewed"] += 1
            if is_true_positive:
                self.stats["true_positives"] += 1
            else:
                self.stats["false_positives"] += 1
                
            # Calculate accuracy
            if self.stats["total_reviewed"] > 0:
                self.stats["accuracy"] = self.stats["true_positives"] / self.stats["total_reviewed"]
                
            self._save_stats()
            logger.info(f"Recorded feedback for anomaly {anomaly_id}: {'true positive' if is_true_positive else 'false positive'}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording feedback for anomaly {anomaly_id}: {e}")
            return False
            
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get current feedback statistics
        
        Returns:
            Dict with feedback statistics
        """
        # Update with the current timestamp
        stats = self.stats.copy()
        stats["current_time"] = datetime.now().isoformat()
        stats["false_positive_rate"] = (stats["false_positives"] / stats["total_reviewed"] 
                                      if stats["total_reviewed"] > 0 else 0)
        return stats
