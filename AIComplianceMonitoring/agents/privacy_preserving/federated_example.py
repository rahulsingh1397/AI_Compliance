"""
Federated Learning with FHE - Complete Example

This module demonstrates how to use the FederatedClient and FederatedLearningManager
together for secure, privacy-preserving federated learning.
"""

import numpy as np
import tenseal as ts
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import logging

from .federated_client import FederatedClient, ClientConfig
from .federated_learning import FederatedLearningManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 1000, n_features: int = 20, n_clients: int = 3):
    """
    Create sample data distributed across multiple clients.
    
    Args:
        n_samples: Total number of samples
        n_features: Number of features
        n_clients: Number of clients to distribute data across
        
    Returns:
        List of (X, y) tuples for each client
    """
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features//2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split data among clients (non-IID distribution)
    client_data = []
    samples_per_client = n_samples // n_clients
    
    for i in range(n_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < n_clients - 1 else n_samples
        
        client_X = X[start_idx:end_idx]
        client_y = y[start_idx:end_idx]
        
        client_data.append((client_X, client_y))
    
    return client_data

def run_federated_learning_example():
    """
    Run a complete federated learning example with FHE encryption.
    """
    logger.info("Starting Federated Learning with FHE Example")
    
    # Configuration
    n_clients = 3
    n_rounds = 5
    
    # Create sample data
    client_datasets = create_sample_data(n_samples=1500, n_clients=n_clients)
    
    # Initialize the federated server
    base_model = LogisticRegression(random_state=42)
    server = FederatedLearningManager(model=base_model, num_clients=n_clients)
    
    # Initialize clients
    clients = []
    for i in range(n_clients):
        config = ClientConfig(
            client_id=f"client_{i}",
            server_url="http://localhost:8080",
            differential_privacy=True,
            dp_noise_multiplier=0.1,
            dp_l2_norm_clip=1.0
        )
        
        # Each client gets its own model instance
        client_model = LogisticRegression(random_state=42)
        client = FederatedClient(config=config, model=client_model, fhe_context=server.fhe_context)
        
        # Register client with server
        server.add_client(config.client_id, metadata={'dataset_size': len(client_datasets[i][0])})
        client.register_with_server()
        
        clients.append(client)
    
    logger.info(f"Initialized {n_clients} clients and server")
    
    # Federated learning rounds
    for round_num in range(n_rounds):
        logger.info(f"\n--- Round {round_num + 1}/{n_rounds} ---")
        
        # Collect updates from all clients
        client_updates = []
        
        for i, client in enumerate(clients):
            logger.info(f"Training client {client.config.client_id}")
            
            # Train local model
            X_train, y_train = client_datasets[i]
            training_metrics = client.train_local_model(X_train, y_train)
            
            # Create encrypted update
            encrypted_update = client.create_encrypted_update(training_metrics)
            client_updates.append(encrypted_update)
            
            logger.info(f"Client {client.config.client_id} - Samples: {training_metrics.samples_count}, "
                       f"Accuracy: {training_metrics.accuracy:.3f if training_metrics.accuracy else 'N/A'}")
        
        # Server aggregates encrypted updates
        logger.info("Server aggregating encrypted updates...")
        try:
            aggregated_weights = server.aggregate_updates(client_updates)
            logger.info("Aggregation successful")
            
            # In a real scenario, the server would send encrypted weights back to clients
            # For this example, we'll simulate the process
            for client in clients:
                success = client.decrypt_global_model(aggregated_weights)
                if success:
                    logger.info(f"Client {client.config.client_id} updated with global model")
                else:
                    logger.error(f"Client {client.config.client_id} failed to update")
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            break
    
    # Final evaluation
    logger.info("\n--- Final Evaluation ---")
    for i, client in enumerate(clients):
        stats = client.get_client_stats()
        logger.info(f"Client {stats['client_id']}:")
        logger.info(f"  - Rounds participated: {stats['total_rounds_participated']}")
        logger.info(f"  - Total samples: {stats['total_samples_trained']}")
        logger.info(f"  - Avg training time: {stats['average_training_time']:.2f}s")
        logger.info(f"  - DP enabled: {stats['differential_privacy_enabled']}")
        logger.info(f"  - FHE configured: {stats['fhe_context_configured']}")
    
    logger.info("Federated Learning with FHE Example completed successfully!")

def demonstrate_privacy_features():
    """
    Demonstrate the privacy-preserving features of the system.
    """
    logger.info("\n=== Privacy Features Demonstration ===")
    
    # Create a simple client for demonstration
    config = ClientConfig(
        client_id="demo_client",
        server_url="http://localhost:8080",
        differential_privacy=True,
        dp_noise_multiplier=1.0
    )
    
    model = LogisticRegression()
    client = FederatedClient(config=config, model=model)
    
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    # Train and create update
    metrics = client.train_local_model(X, y)
    encrypted_update = client.create_encrypted_update(metrics)
    
    logger.info("Privacy Features:")
    logger.info(f"1. Differential Privacy: {config.differential_privacy}")
    logger.info(f"2. FHE Encryption: {client.fhe_context is not None}")
    logger.info(f"3. Digital Signatures: {encrypted_update.signature is not None}")
    logger.info(f"4. Encrypted Weights: {type(list(encrypted_update.weights.values())[0])}")
    
    # Show that weights are encrypted (cannot be directly accessed)
    try:
        first_weight_key = list(encrypted_update.weights.keys())[0]
        encrypted_weight = encrypted_update.weights[first_weight_key]
        logger.info(f"5. Weight Protection: Encrypted weight type = {type(encrypted_weight)}")
        logger.info("   Raw encrypted data cannot be directly accessed without decryption")
    except Exception as e:
        logger.info(f"5. Weight Protection: {e}")

if __name__ == "__main__":
    # Run the complete example
    run_federated_learning_example()
    
    # Demonstrate privacy features
    demonstrate_privacy_features()
