"""
Federated Learning Manager

This module implements a federated learning system that enables collaborative model
training across multiple parties while keeping data decentralized.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import hashlib

@dataclass
class ModelUpdate:
    """Container for model updates in federated learning."""
    weights: Dict[str, np.ndarray]
    samples_count: int
    client_id: str
    metadata: Dict[str, Any] = None
    signature: Optional[bytes] = None

class FederatedLearningManager:
    """
    Manages federated learning operations including model aggregation and update distribution.
    """
    
    def __init__(self, model: Any, num_clients: int = 5):
        """
        Initialize the Federated Learning manager.
        
        Args:
            model: The base model architecture to use
            num_clients: Expected number of clients in the federation
        """
        self.global_model = model
        self.num_clients = num_clients
        self.clients: Dict[str, Dict] = {}
        self.round = 0
        
    def add_client(self, client_id: str, metadata: Optional[Dict] = None) -> None:
        """
        Register a new client in the federated learning system.
        
        Args:
            client_id: Unique identifier for the client
            metadata: Additional client metadata
        """
        self.clients[client_id] = {
            'metadata': metadata or {},
            'last_update': None,
            'participation_count': 0
        }
    
    def get_global_model_weights(self) -> Dict[str, np.ndarray]:
        """
        Get the current global model weights.
        Handles both Keras-style models (get_weights) and scikit-learn models.

        Returns:
            Dictionary containing model weights
        """
        if hasattr(self.global_model, 'get_weights'):
            # Handle Keras-style models
            return self.global_model.get_weights()
        elif hasattr(self.global_model, 'coef_') and hasattr(self.global_model, 'intercept_'):
            # Handle scikit-learn linear models
            return {
                'coef_': self.global_model.coef_,
                'intercept_': self.global_model.intercept_
            }
        else:
            raise ValueError("Unsupported model type. Model must have get_weights() or coef_/intercept_ attributes.")
    
    def aggregate_updates(
        self, 
        updates: List[ModelUpdate],
        aggregation_method: str = 'fedavg'
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate model updates from multiple clients.
        
        Args:
            updates: List of model updates from clients
            aggregation_method: Method to use for aggregation ('fedavg', 'fedprox', etc.)
            
        Returns:
            Aggregated model weights
        """
        if not updates:
            return self.get_global_model_weights()
            
        # Verify all updates before aggregation
        verified_updates = []
        total_samples = 0
        
        for update in updates:
            if self._verify_update(update):
                verified_updates.append(update)
                total_samples += update.samples_count
        
        if not verified_updates:
            raise ValueError("No valid updates to aggregate")
            
        # Perform weighted averaging of updates
        avg_weights = {}
        for key in verified_updates[0].weights.keys():
            weighted_sum = np.zeros_like(verified_updates[0].weights[key])
            for update in verified_updates:
                weight = update.samples_count / total_samples
                weighted_sum += update.weights[key] * weight
            avg_weights[key] = weighted_sum
            
        return avg_weights
    
    def update_global_model(self, aggregated_weights: Dict[str, np.ndarray]) -> None:
        """
        Update the global model with aggregated weights.
        
        Args:
            aggregated_weights: Aggregated model weights
        """
        self.global_model.set_weights(aggregated_weights)
        self.round += 1
    
    def _verify_update(self, update: ModelUpdate) -> bool:
        """
        Verify the integrity and authenticity of a model update.
        
        Args:
            update: The model update to verify
            
        Returns:
            bool: True if the update is valid, False otherwise
        """
        # Verify client is registered
        if update.client_id not in self.clients:
            return False
            
        # Verify update signature if present
        if update.signature:
            # In a real implementation, verify the signature using the client's public key
            pass
            
        # Verify weights structure matches expected
        current_weights = self.get_global_model_weights()
        if update.weights.keys() != current_weights.keys():
            return False
            
        # Verify sample count is reasonable
        if update.samples_count <= 0 or update.samples_count > 1_000_000:  # Arbitrary upper limit
            return False
            
        return True
    
    def get_model_hash(self) -> str:
        """
        Calculate a hash of the current model weights.
        
        Returns:
            str: Hex digest of the model hash
        """
        weights = self.get_global_model_weights()
        hash_obj = hashlib.sha256()
        
        # Sort keys for consistent hashing
        for key in sorted(weights.keys()):
            hash_obj.update(weights[key].tobytes())
            
        return hash_obj.hexdigest()
