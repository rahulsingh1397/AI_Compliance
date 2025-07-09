"""
Demonstration of the Privacy-Preserving Agent components.

This script shows how to use the ZKML, Federated Learning, Audit Log,
and Data Protection components together in a privacy-preserving AI system.
"""

import os
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import our privacy-preserving components
from .. import ZKMLManager, FederatedLearningManager, SecureAuditLog, DataProtectionManager
from ..audit_log import AuditLogEntryType

def setup_demo_environment():
    """Set up the demo environment with sample data and models."""
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        random_state=42
    )
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train a simple model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def demo_zkml():
    """Demonstrate Zero-Knowledge Machine Learning functionality."""
    print("\n=== ZKML Demo ===")
    
    # Set up environment
    model, X_test, _ = setup_demo_environment()
    
    # Initialize ZKML Manager
    zkml = ZKMLManager(model)
    
    # Generate a proof for a prediction
    sample_input = X_test[0:1]  # Single sample
    proof = zkml.generate_proof(sample_input)
    
    print(f"Generated proof for prediction on sample: {sample_input[0][:5]}...")
    print(f"Proof metadata: {proof.metadata}")
    
    # Verify the proof
    is_valid, details = zkml.verify_proof(proof)
    print(f"Proof verification: {'SUCCESS' if is_valid else 'FAILED'}")
    print(f"Verification details: {details}")
    
    return zkml, proof

def demo_federated_learning():
    """Demonstrate Federated Learning functionality."""
    print("\n=== Federated Learning Demo ===")
    
    # Set up environment
    model, X_test, y_test = setup_demo_environment()
    
    # Initialize Federated Learning Manager
    fl_manager = FederatedLearningManager(model, num_clients=3)
    
    # Add some clients
    for i in range(3):
        fl_manager.add_client(f"client_{i+1}")
    
    # Simulate client updates
    updates = []
    for i in range(3):
        # In a real scenario, these would come from different clients
        client_model = LogisticRegression(max_iter=1000)
        client_model.coef_ = model.coef_ + np.random.normal(0, 0.1, size=model.coef_.shape)
        client_model.intercept_ = model.intercept_
        
        update = ModelUpdate(
            weights={
                'coef_': client_model.coef_,
                'intercept_': client_model.intercept_
            },
            samples_count=100,  # Simulated sample count
            client_id=f"client_{i+1}",
            metadata={"round": 1, "client_version": "1.0"}
        )
        updates.append(update)
    
    # Aggregate updates
    aggregated_weights = fl_manager.aggregate_updates(updates)
    print(f"Aggregated weights for {len(updates)} client updates")
    print(f"Updated model hash: {fl_manager.get_model_hash()}")
    
    return fl_manager, updates

def demo_audit_log():
    """Demonstrate Secure Audit Log functionality."""
    print("\n=== Secure Audit Log Demo ===")
    
    # Initialize audit log
    audit_log = SecureAuditLog()
    
    # Add some entries
    audit_log.add_entry(
        entry_type=AuditLogEntryType.MODEL_TRAIN,
        user_id="user1",
        operation="train_model",
        details={"model_type": "LogisticRegression", "dataset": "synthetic"}
    )
    
    audit_log.add_entry(
        entry_type=AuditLogEntryType.DATA_ACCESS,
        user_id="user2",
        operation="read_sensitive_data",
        details={"dataset": "medical_records", "rows_accessed": 1000}
    )
    
    # Verify log integrity
    is_valid, issues = audit_log.verify_log_integrity()
    print(f"Audit log integrity check: {'VALID' if is_valid else 'INVALID'}")
    if issues:
        print(f"Issues found: {len(issues)}")
        for issue in issues[:3]:  # Show first 3 issues if any
            print(f"- {issue['issue']} (severity: {issue['severity']})")
    
    # Export and import
    log_data = audit_log.export_log()
    imported_log = SecureAuditLog.import_log(log_data)
    print(f"Successfully exported and re-imported audit log with {len(imported_log.entries)} entries")
    
    return audit_log

def demo_data_protection():
    """Demonstrate Data Protection functionality."""
    print("\n=== Data Protection Demo ===")
    
    # Initialize data protection
    dp = DataProtectionManager()
    
    # Encrypt and decrypt
    secret_message = "This is a sensitive message"
    protected = dp.encrypt(secret_message)
    decrypted = dp.decrypt(protected)
    
    print(f"Original: {secret_message}")
    print(f"Encrypted: {protected.data[:30]}...")
    print(f"Decrypted: {decrypted.decode('utf-8')}")
    
    # Pseudonymization
    sensitive_data = "user@example.com"
    pseudonym = dp.pseudonymize(sensitive_data)
    is_match = dp.verify_pseudonym(sensitive_data, pseudonym)
    
    print(f"\nOriginal: {sensitive_data}")
    print(f"Pseudonym: {pseudonym.data.hex()[:16]}...")
    print(f"Verification: {'MATCH' if is_match else 'NO MATCH'}")
    
    return dp, protected, pseudonym

def main():
    """Run all demos."""
    print("=== Privacy-Preserving Agent Demo ===\n")
    
    # Run demos
    zkml, zk_proof = demo_zkml()
    fl_manager, fl_updates = demo_federated_learning()
    audit_log = demo_audit_log()
    dp, protected_data, pseudonym = demo_data_protection()
    
    print("\n=== Demo Complete ===")
    print("Successfully demonstrated all privacy-preserving components!")

if __name__ == "__main__":
    main()
