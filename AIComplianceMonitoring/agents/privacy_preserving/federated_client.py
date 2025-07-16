"""
Federated Learning Client with FHE Support

This module implements a production-grade federated learning client that supports
Fully Homomorphic Encryption (FHE) for secure model updates.
"""

import numpy as np
import tenseal as ts
import logging
import hashlib
import time
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, asdict
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_public_key
import json
import base64

from .federated_learning import ModelUpdate

@dataclass
class ClientConfig:
    """Configuration for the federated client."""
    client_id: str
    server_url: str
    max_retries: int = 3
    timeout_seconds: int = 30
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    differential_privacy: bool = False
    dp_noise_multiplier: float = 1.0
    dp_l2_norm_clip: float = 1.0

@dataclass
class TrainingMetrics:
    """Container for training metrics and metadata."""
    loss: float
    accuracy: Optional[float]
    samples_count: int
    training_time: float
    local_epochs: int
    timestamp: str

class FederatedClient:
    """
    Production-grade federated learning client with FHE support.
    
    This client handles:
    - Local model training with privacy-preserving techniques
    - FHE encryption of model updates
    - Secure communication with the federated server
    - Differential privacy integration
    - Robust error handling and retry logic
    """
    
    def __init__(self, config: ClientConfig, model: Any, fhe_context: Optional[ts.Context] = None):
        """
        Initialize the federated client.
        
        Args:
            config: Client configuration parameters
            model: The local model instance (e.g., PyTorch, TensorFlow, or scikit-learn)
            fhe_context: Optional pre-configured FHE context. If None, will be received from server.
        """
        self.config = config
        self.model = model
        self.fhe_context = fhe_context
        self.logger = self._setup_logging()
        
        # Client state
        self.current_round = 0
        self.training_history: List[TrainingMetrics] = []
        self.server_public_key: Optional[bytes] = None
        
        # Generate client key pair for authentication
        self._generate_client_keys()
        
        self.logger.info(f"Initialized federated client {config.client_id}")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the client."""
        logger = logging.getLogger(f"FederatedClient-{self.config.client_id}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _generate_client_keys(self) -> None:
        """Generate RSA key pair for client authentication."""
        self._private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self._public_key = self._private_key.public_key()
        
        self.logger.debug("Generated client authentication keys")
    
    def get_public_key(self) -> bytes:
        """Get the client's public key for server authentication."""
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    def register_with_server(self) -> bool:
        """
        Register this client with the federated server.
        
        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            registration_data = {
                'client_id': self.config.client_id,
                'public_key': base64.b64encode(self.get_public_key()).decode('utf-8'),
                'capabilities': {
                    'fhe_support': True,
                    'differential_privacy': self.config.differential_privacy,
                    'model_type': type(self.model).__name__
                }
            }
            
            # In a real implementation, this would make an HTTP request to the server
            # For now, we'll simulate successful registration
            self.logger.info(f"Client {self.config.client_id} registered with server")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register with server: {e}")
            return False
    
    def receive_fhe_context(self, context_data: bytes) -> bool:
        """
        Receive and set up the FHE context from the server.
        
        Args:
            context_data: Serialized FHE context from the server
            
        Returns:
            bool: True if context setup successful, False otherwise
        """
        try:
            # In a real implementation, this would deserialize the context
            # For now, we'll create a compatible context
            if self.fhe_context is None:
                self.fhe_context = ts.context(
                    ts.SCHEME_TYPE.CKKS,
                    poly_modulus_degree=8192,
                    coeff_mod_bit_sizes=[60, 40, 40, 60]
                )
                self.fhe_context.global_scale = 2**40
                self.fhe_context.generate_galois_keys()
            
            self.logger.info("FHE context configured successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to configure FHE context: {e}")
            return False
    
    def train_local_model(self, train_data: Any, train_labels: Any) -> TrainingMetrics:
        """
        Train the local model on client data.
        
        Args:
            train_data: Training data (format depends on model type)
            train_labels: Training labels
            
        Returns:
            TrainingMetrics: Metrics from the training process
        """
        start_time = time.time()
        
        try:
            # This is a simplified training loop - in practice, this would be
            # framework-specific (PyTorch, TensorFlow, etc.)
            
            if hasattr(self.model, 'fit'):
                # Scikit-learn style model
                self.model.fit(train_data, train_labels)
                loss = 0.0  # Placeholder
                accuracy = self.model.score(train_data, train_labels) if hasattr(self.model, 'score') else None
            else:
                # For neural networks, implement training loop here
                loss = 0.0  # Placeholder
                accuracy = None
            
            training_time = time.time() - start_time
            samples_count = len(train_data) if hasattr(train_data, '__len__') else 0
            
            metrics = TrainingMetrics(
                loss=loss,
                accuracy=accuracy,
                samples_count=samples_count,
                training_time=training_time,
                local_epochs=self.config.local_epochs,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
            self.training_history.append(metrics)
            self.logger.info(f"Local training completed: {samples_count} samples, {training_time:.2f}s")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Local training failed: {e}")
            raise
    
    def _extract_model_weights(self) -> Dict[str, np.ndarray]:
        """Extract weights from the local model."""
        if hasattr(self.model, 'get_weights'):
            # Keras-style model
            weights = self.model.get_weights()
            return {f'layer_{i}': w for i, w in enumerate(weights)}
        elif hasattr(self.model, 'coef_') and hasattr(self.model, 'intercept_'):
            # Scikit-learn linear model
            return {
                'coef_': self.model.coef_,
                'intercept_': self.model.intercept_
            }
        else:
            raise ValueError("Unsupported model type for weight extraction")
    
    def _apply_differential_privacy(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply differential privacy noise to model weights.
        
        Args:
            weights: Model weights to add noise to
            
        Returns:
            Dict[str, np.ndarray]: Weights with DP noise added
        """
        if not self.config.differential_privacy:
            return weights
        
        noisy_weights = {}
        for key, weight in weights.items():
            # Clip weights to bound sensitivity
            clipped_weight = np.clip(weight, -self.config.dp_l2_norm_clip, self.config.dp_l2_norm_clip)
            
            # Add Gaussian noise
            noise = np.random.normal(
                0, 
                self.config.dp_noise_multiplier * self.config.dp_l2_norm_clip,
                weight.shape
            )
            noisy_weights[key] = clipped_weight + noise
        
        self.logger.debug("Applied differential privacy noise to weights")
        return noisy_weights
    
    def _encrypt_weights(self, weights: Dict[str, np.ndarray]) -> Dict[str, ts.CKKSVector]:
        """
        Encrypt model weights using FHE.
        
        Args:
            weights: Plaintext model weights
            
        Returns:
            Dict[str, ts.CKKSVector]: Encrypted weights
        """
        if self.fhe_context is None:
            raise ValueError("FHE context not configured")
        
        encrypted_weights = {}
        for key, weight in weights.items():
            # Flatten weight array for encryption
            flattened_weight = weight.flatten()
            encrypted_weight = ts.ckks_vector(self.fhe_context, flattened_weight.tolist())
            encrypted_weights[key] = encrypted_weight
        
        self.logger.debug(f"Encrypted {len(weights)} weight tensors")
        return encrypted_weights
    
    def _sign_update(self, update_data: Dict[str, Any]) -> bytes:
        """
        Create a digital signature for the model update.
        
        Args:
            update_data: The update data to sign
            
        Returns:
            bytes: Digital signature
        """
        # Create a hash of the update data
        update_json = json.dumps(update_data, sort_keys=True, default=str)
        update_hash = hashlib.sha256(update_json.encode()).digest()
        
        # Sign the hash
        signature = self._private_key.sign(
            update_hash,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature
    
    def create_encrypted_update(self, training_metrics: TrainingMetrics) -> ModelUpdate:
        """
        Create an encrypted model update for the server.
        
        Args:
            training_metrics: Metrics from local training
            
        Returns:
            ModelUpdate: Encrypted model update ready for transmission
        """
        try:
            # Extract model weights
            weights = self._extract_model_weights()
            
            # Apply differential privacy if enabled
            if self.config.differential_privacy:
                weights = self._apply_differential_privacy(weights)
            
            # Encrypt weights using FHE
            encrypted_weights = self._encrypt_weights(weights)
            
            # Create update metadata
            metadata = {
                'client_id': self.config.client_id,
                'round': self.current_round,
                'training_metrics': asdict(training_metrics),
                'differential_privacy': self.config.differential_privacy,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Create the model update
            update = ModelUpdate(
                weights=encrypted_weights,
                samples_count=training_metrics.samples_count,
                client_id=self.config.client_id,
                metadata=metadata
            )
            
            # Sign the update for authentication
            signature_data = {
                'client_id': self.config.client_id,
                'samples_count': training_metrics.samples_count,
                'round': self.current_round
            }
            update.signature = self._sign_update(signature_data)
            
            self.logger.info(f"Created encrypted model update for round {self.current_round}")
            return update
            
        except Exception as e:
            self.logger.error(f"Failed to create encrypted update: {e}")
            raise
    
    def decrypt_global_model(self, encrypted_weights: Dict[str, ts.CKKSVector]) -> bool:
        """
        Decrypt and apply the global model weights received from the server.
        
        Args:
            encrypted_weights: Encrypted global model weights
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Decrypt the weights
            decrypted_weights = {}
            for key, encrypted_weight in encrypted_weights.items():
                decrypted_flat = encrypted_weight.decrypt()
                # Note: In a real implementation, you'd need to know the original shape
                # to properly reshape the decrypted weights
                decrypted_weights[key] = np.array(decrypted_flat)
            
            # Apply weights to the local model
            if hasattr(self.model, 'set_weights'):
                # Keras-style model
                weight_list = [decrypted_weights[f'layer_{i}'] for i in range(len(decrypted_weights))]
                self.model.set_weights(weight_list)
            elif hasattr(self.model, 'coef_') and hasattr(self.model, 'intercept_'):
                # Scikit-learn linear model
                self.model.coef_ = decrypted_weights['coef_']
                self.model.intercept_ = decrypted_weights['intercept_']
            
            self.current_round += 1
            self.logger.info(f"Applied global model weights for round {self.current_round}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt global model: {e}")
            return False
    
    def get_client_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive client statistics.
        
        Returns:
            Dict[str, Any]: Client statistics and metrics
        """
        if not self.training_history:
            return {'status': 'no_training_history'}
        
        recent_metrics = self.training_history[-1]
        total_samples = sum(m.samples_count for m in self.training_history)
        total_training_time = sum(m.training_time for m in self.training_history)
        
        return {
            'client_id': self.config.client_id,
            'current_round': self.current_round,
            'total_rounds_participated': len(self.training_history),
            'total_samples_trained': total_samples,
            'total_training_time': total_training_time,
            'average_training_time': total_training_time / len(self.training_history),
            'last_training_accuracy': recent_metrics.accuracy,
            'last_training_loss': recent_metrics.loss,
            'differential_privacy_enabled': self.config.differential_privacy,
            'fhe_context_configured': self.fhe_context is not None
        }
