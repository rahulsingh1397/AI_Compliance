"""
Human Interaction Agent

This module implements an agent that prioritizes anomalies for human review
and collects feedback to improve the anomaly detection system over time.
"""

import logging
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

from .feedback_loop import FeedbackLoopBase

# Configure logging
logger = logging.getLogger(__name__)

class HumanInteractionAgent(FeedbackLoopBase):
    """Agent that interacts with human experts to collect feedback on anomalies"""
    
    def __init__(self, config):
        """Initialize the human interaction agent
        
        Args:
            config: Configuration object with required settings
        """
        super().__init__(config)
        
        # Agent-specific attributes
        self.review_queue_path = self.feedback_dir / 'review_queue.json'
        self.priority_thresholds = {
            'high': getattr(config, 'high_priority_threshold', 0.8),
            'medium': getattr(config, 'medium_priority_threshold', 0.5),
            'low': getattr(config, 'low_priority_threshold', 0.3)
        }
        
        self.review_queue = {
            'high': [],
            'medium': [],
            'low': []
        }
        
        # Load existing review queue if available
        self._load_review_queue()
        
    def _load_review_queue(self):
        """Load the review queue from disk"""
        if self.review_queue_path.exists():
            try:
                with open(self.review_queue_path, 'r') as f:
                    self.review_queue = json.load(f)
                logger.info(f"Loaded review queue: {sum(len(q) for q in self.review_queue.values())} items")
            except Exception as e:
                logger.error(f"Error loading review queue: {e}")
                
    def _save_review_queue(self):
        """Save the review queue to disk"""
        try:
            with open(self.review_queue_path, 'w') as f:
                json.dump(self.review_queue, f)
            logger.info(f"Saved review queue: {sum(len(q) for q in self.review_queue.values())} items")
        except Exception as e:
            logger.error(f"Error saving review queue: {e}")
    
    def queue_anomaly_for_review(self, anomaly_data: Dict[str, Any], 
                                validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Queue an anomaly for human review
        
        Args:
            anomaly_data: Dictionary containing anomaly data
            validation_result: Result from the validation agent
            
        Returns:
            Dictionary with queue information
        """
        # Add timestamp and generate review ID
        if 'timestamp' not in anomaly_data:
            anomaly_data['timestamp'] = datetime.now().isoformat()
        
        review_id = f"{anomaly_data.get('id', 'unknown')}_{int(time.time())}"
        anomaly_data['review_id'] = review_id
        
        # Determine priority based on confidence and severity
        confidence = validation_result.get('confidence', 0.5)
        severity = anomaly_data.get('severity', 'medium')
        
        # Invert confidence - lower confidence means higher review priority
        review_urgency = 1.0 - confidence
        
        # Adjust by severity
        if severity == 'high':
            priority = 'high'
        elif severity == 'medium':
            # High review urgency can promote to high priority
            priority = 'high' if review_urgency > self.priority_thresholds['high'] else 'medium'
        else:  # low severity
            if review_urgency > self.priority_thresholds['high']:
                priority = 'medium'
            else:
                priority = 'low'
        
        # Create review item
        review_item = {
            'review_id': review_id,
            'anomaly_data': anomaly_data,
            'validation_result': validation_result,
            'queued_at': datetime.now().isoformat(),
            'priority': priority,
            'review_urgency': float(review_urgency),
            'status': 'pending'
        }
        
        # Add to appropriate queue
        self.review_queue[priority].append(review_item)
        
        # Save updated queue
        self._save_review_queue()
        
        logger.info(f"Added anomaly {review_id} to {priority} priority review queue")
        
        return {
            'review_id': review_id,
            'priority': priority,
            'position': len(self.review_queue[priority]),
            'status': 'queued'
        }
    
    def get_next_review_item(self, max_priority: str = 'high') -> Optional[Dict[str, Any]]:
        """Get the next item for review
        
        Args:
            max_priority: Maximum priority to consider ('high', 'medium', 'low')
            
        Returns:
            Review item or None if queue is empty
        """
        # Priority order
        priorities = ['high', 'medium', 'low']
        
        # Only consider priorities up to max_priority
        max_idx = priorities.index(max_priority)
        considered_priorities = priorities[:max_idx+1]
        
        # Check each priority queue in order
        for priority in considered_priorities:
            if self.review_queue[priority]:
                # Get the oldest item in this priority queue
                item = self.review_queue[priority][0]
                
                # Update status
                item['status'] = 'in_review'
                
                # Save queue
                self._save_review_queue()
                
                logger.info(f"Retrieved {priority} priority review item {item['review_id']}")
                return item
                
        logger.info(f"No review items found with priority <= {max_priority}")
        return None
    
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
        # Search for the review item
        for priority, queue in self.review_queue.items():
            for i, item in enumerate(queue):
                if item.get('review_id') == review_id:
                    # Found the item
                    anomaly_data = item.get('anomaly_data', {})
                    
                    # Record feedback
                    context = {
                        'anomaly_score': anomaly_data.get('anomaly_score', 0),
                        'severity': anomaly_data.get('severity', 'medium'),
                        'data_size': anomaly_data.get('data_size', 0),
                        'anomaly_type': anomaly_data.get('type', 'unknown'),
                        'user_id': anomaly_data.get('user_id', 'unknown'),
                        'resource_id': anomaly_data.get('resource_id', 'unknown'),
                        'action': anomaly_data.get('action', 'unknown'),
                        'ip_address': anomaly_data.get('ip_address', 'unknown'),
                        'notes': notes,
                        'validation_confidence': item.get('validation_result', {}).get('confidence', 0)
                    }
                    
                    # Record the feedback
                    feedback_success = self.record_feedback(
                        anomaly_id=anomaly_data.get('id', review_id),
                        is_true_positive=is_true_positive,
                        anomaly_type=anomaly_data.get('type', 'unknown'),
                        feedback_source='human_expert',
                        context=context
                    )
                    
                    # Remove from queue
                    self.review_queue[priority].pop(i)
                    self._save_review_queue()
                    
                    logger.info(f"Recorded human feedback for anomaly {review_id}: {'true positive' if is_true_positive else 'false positive'}")
                    
                    return {
                        'status': 'success',
                        'review_id': review_id,
                        'feedback_recorded': feedback_success
                    }
        
        logger.warning(f"Review item {review_id} not found in queue")
        return {
            'status': 'error',
            'message': f"Review item {review_id} not found"
        }
    
    def get_review_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the review queue
        
        Returns:
            Dictionary with queue statistics
        """
        stats = {
            'high_priority_count': len(self.review_queue['high']),
            'medium_priority_count': len(self.review_queue['medium']),
            'low_priority_count': len(self.review_queue['low']),
            'total_pending': sum(len(q) for q in self.review_queue.values()),
            'oldest_high_priority': None,
            'oldest_medium_priority': None,
            'oldest_low_priority': None,
        }
        
        # Get timestamp of oldest item in each queue
        for priority in ['high', 'medium', 'low']:
            if self.review_queue[priority]:
                oldest = min(self.review_queue[priority], 
                           key=lambda x: x.get('queued_at', datetime.now().isoformat()))
                stats[f'oldest_{priority}_priority'] = oldest.get('queued_at')
        
        # Add feedback stats
        stats.update(self.get_feedback_stats())
        
        return stats
        
    def clean_review_queue(self, max_age_days: int = 30) -> int:
        """Clean old items from the review queue
        
        Args:
            max_age_days: Maximum age of items to keep (in days)
            
        Returns:
            Number of items removed
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        cutoff_str = cutoff_date.isoformat()
        
        removed_count = 0
        
        for priority in self.review_queue:
            original_len = len(self.review_queue[priority])
            self.review_queue[priority] = [
                item for item in self.review_queue[priority]
                if item.get('queued_at', datetime.now().isoformat()) > cutoff_str
            ]
            removed_count += original_len - len(self.review_queue[priority])
        
        
        if removed_count > 0:
            self._save_review_queue()
            logger.info(f"Removed {removed_count} items older than {max_age_days} days from review queue")
            
        return removed_count
