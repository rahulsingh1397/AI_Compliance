"""
Federated Learning Manager

This module implements a federated learning system that enables collaborative model
training across multiple parties while keeping data decentralized.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import hashlib
import tenseal as ts

@dataclass
class ModelUpdate:
    """Container for model updates in federated learning."""
    weights: Dict[str, Any]  # Can be np.ndarray or ts.CKKSVector
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

        # Initialize FHE context (CKKS scheme for floating-point operations)
        self.fhe_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.fhe_context.global_scale = 2**40
        self.fhe_context.generate_galois_keys()
        
    def add_client(self, client_id: str, metadata: Optional[Dict] = None) -> None:
        """
        Registers a new client in the federated learning system.

        This method adds a client to the internal registry, allowing it to participate
        in future federated learning rounds. Each client is tracked with its
        metadata and participation history.

        Args:
            client_id (str): A unique identifier for the client.
            metadata (Optional[Dict]): A dictionary for storing arbitrary client
                metadata, such as hardware specifications or location.
        """
        self.clients[client_id] = {
            'metadata': metadata or {},
            'last_update': None,
            'participation_count': 0
        }
    
    def get_global_model_weights(self) -> Dict[str, np.ndarray]:
        """
        Retrieves the weights of the current global model.

        This function is designed to be model-agnostic, supporting popular
        frameworks by detecting the appropriate method for weight extraction.
        Currently, it handles Keras models via `get_weights()` and scikit-learn
        linear models by accessing `coef_` and `intercept_`.

        Returns:
            Dict[str, np.ndarray]: A dictionary where keys are layer/weight names
            and values are the corresponding NumPy arrays of the weights.

        Raises:
            ValueError: If the model is of an unsupported type and its weights
                cannot be extracted.
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
    ) -> Dict[str, ts.CKKSVector]:
        """
        Aggregates model updates from multiple clients using FHE to form a new global model.

        This function first verifies the integrity of each client update.
        It then combines the verified updates (which are encrypted TenSEAL vectors)
        using the specified aggregation strategy.

        The default strategy is Federated Averaging ('fedavg'), where encrypted client
        updates are multiplied by their plaintext weights (based on sample size)
        and then summed together homomorphically.

        Args:
            updates (List[ModelUpdate]): A list of model updates from clients,
                where weights are encrypted TenSEAL vectors.
            aggregation_method (str): The algorithm to use for aggregation.
                Currently supports 'fedavg'. Defaults to 'fedavg'.

        Returns:
            Dict[str, ts.CKKSVector]: A dictionary containing the new, aggregated
            model weights as encrypted TenSEAL vectors.

        Raises:
            ValueError: If no valid updates are provided after verification.
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

        # Perform weighted averaging of updates using FHE
        avg_weights = {}
        # Assume all updates have the same structure, take the first one for keys
        first_update_weights = verified_updates[0].weights
        for key in first_update_weights.keys():
            # Initialize an encrypted zero vector for the sum.
            # We must decrypt one vector to know the shape, which is a limitation.
            # A better approach would be to know the model shape beforehand.
            decrypted_shape_ref = first_update_weights[key].decrypt()
            weighted_sum = ts.ckks_vector(self.fhe_context, np.zeros_like(decrypted_shape_ref))

            for update in verified_updates:
                weight = update.samples_count / total_samples  # Plaintext scalar
                # Homomorphically multiply encrypted weights by plaintext weight and add to sum
                weighted_term = update.weights[key] * weight
                weighted_sum += weighted_term
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
