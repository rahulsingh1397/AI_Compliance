"""
Unit tests for the privacy-preserving agent components.
"""

import os
import sys
import json
import pytest
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import components to test
from privacy_preserving import ZKMLManager, FederatedLearningManager, SecureAuditLog, DataProtectionManager
from privacy_preserving.audit_log import AuditLogEntry, AuditLogEntryType
from privacy_preserving.federated_learning import ModelUpdate

# Test data
@pytest.fixture
def sample_model():
    """Create a simple trained model for testing."""
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

@pytest.fixture
def sample_data():
    """Create sample input data for testing."""
    X, y = make_classification(n_samples=10, n_features=10, random_state=42)
    return X[0:1]  # Return first sample

class TestZKMLManager:
    """Test cases for ZKMLManager."""
    
    def test_initialization(self, sample_model):
        """Test ZKMLManager initialization."""
        zkml = ZKMLManager(sample_model)
        assert zkml.model == sample_model
        assert not zkml._setup_complete
        
    def test_setup(self, sample_model):
        """Test ZK circuit setup."""
        zkml = ZKMLManager(sample_model)
        zkml.setup()
        assert zkml._setup_complete
        
    def test_generate_proof(self, sample_model, sample_data):
        """Test proof generation."""
        zkml = ZKMLManager(sample_model)
        proof = zkml.generate_proof(sample_data)
        
        assert isinstance(proof.proof_data, dict)
        assert 'proof' in proof.proof_data
        assert 'output' in proof.public_inputs
        
    def test_verify_proof(self, sample_model, sample_data):
        """Test proof verification."""
        zkml = ZKMLManager(sample_model)
        proof = zkml.generate_proof(sample_data)
        is_valid, _ = zkml.verify_proof(proof)
        assert is_valid

class TestFederatedLearningManager:
    """Test cases for FederatedLearningManager."""
    
    @pytest.fixture
    def fl_manager(self, sample_model):
        """Create a FederatedLearningManager instance for testing."""
        return FederatedLearningManager(sample_model, num_clients=3)
    
    @pytest.fixture
    def model_update(self, sample_model):
        """Create a sample model update."""
        return ModelUpdate(
            weights={
                'coef_': sample_model.coef_.copy(),
                'intercept_': sample_model.intercept_.copy()
            },
            samples_count=100,
            client_id="test_client",
            metadata={"round": 1}
        )
    
    def test_add_client(self, fl_manager):
        """Test adding clients to the federated learning system."""
        fl_manager.add_client("client1")
        assert "client1" in fl_manager.clients
        assert fl_manager.clients["client1"]["participation_count"] == 0
    
    def test_aggregate_updates(self, fl_manager, sample_model, model_update):
        """Test aggregating model updates."""
        # Add clients first
        fl_manager.add_client("test_client")
        fl_manager.add_client("test_client2")
        
        # Create multiple updates
        updates = [model_update]
        
        # Create a slightly different update
        update2 = ModelUpdate(
            weights={
                'coef_': sample_model.coef_.copy() + 0.1,
                'intercept_': sample_model.intercept_.copy()
            },
            samples_count=150,
            client_id="test_client2",
            metadata={"round": 1}
        )
        updates.append(update2)
        
        # Aggregate updates
        aggregated = fl_manager.aggregate_updates(updates)
        
        # Check that we got weights back
        assert 'coef_' in aggregated
        assert 'intercept_' in aggregated
        
        # Check shape of aggregated weights
        assert aggregated['coef_'].shape == sample_model.coef_.shape
        
        # Verify the aggregation worked (weighted average)
        expected_coef = (
            model_update.weights['coef_'] * 100 + 
            update2.weights['coef_'] * 150
        ) / 250
        
        assert np.allclose(aggregated['coef_'], expected_coef, rtol=1e-5)
        
    def test_verify_update(self, fl_manager, model_update):
        """Test update verification."""
        fl_manager.add_client("test_client")
        assert fl_manager._verify_update(model_update)
        
        # Test with invalid client
        invalid_update = ModelUpdate(
            weights=model_update.weights,
            samples_count=100,
            client_id="nonexistent_client",
            metadata={"round": 1}
        )
        assert not fl_manager._verify_update(invalid_update)

class TestSecureAuditLog:
    """Test cases for SecureAuditLog."""
    
    @pytest.fixture
    def audit_log(self):
        """Create a SecureAuditLog instance for testing."""
        return SecureAuditLog()
    
    def test_add_entry(self, audit_log):
        """Test adding entries to the audit log."""
        entry = audit_log.add_entry(
            entry_type=AuditLogEntryType.MODEL_TRAIN,
            user_id="test_user",
            operation="train_model",
            details={"model": "test_model"}
        )
        
        assert len(audit_log.entries) == 1
        assert entry.entry_id is not None
        assert entry.previous_hash is None  # First entry
        
    def test_verify_log_integrity(self, audit_log):
        """Test log integrity verification."""
        # Add some entries
        audit_log.add_entry(
            entry_type=AuditLogEntryType.MODEL_TRAIN,
            user_id="user1",
            operation="train",
            details={"model": "test"}
        )
        
        audit_log.add_entry(
            entry_type=AuditLogEntryType.MODEL_PREDICT,
            user_id="user1",
            operation="predict",
            details={"samples": 10}
        )
        
        # Verify integrity
        is_valid, issues = audit_log.verify_log_integrity()
        assert is_valid
        assert not issues
        
    def test_export_import(self, audit_log):
        """Test exporting and importing the audit log."""
        # Add an entry
        audit_log.add_entry(
            entry_type=AuditLogEntryType.DATA_ACCESS,
            user_id="user1",
            operation="access",
            details={"resource": "data.csv"}
        )
        
        # Export and import
        log_data = audit_log.export_log()
        new_log = SecureAuditLog.import_log(log_data)
        
        # Verify
        assert len(new_log.entries) == 1
        assert new_log.entries[0].operation == "access"
        
        # Verify integrity of imported log
        is_valid, _ = new_log.verify_log_integrity()
        assert is_valid

class TestDataProtectionManager:
    """Test cases for DataProtectionManager."""
    
    @pytest.fixture
    def dp_manager(self):
        """Create a DataProtectionManager instance for testing."""
        return DataProtectionManager()
    
    def test_encrypt_decrypt(self, dp_manager):
        """Test encryption and decryption."""
        original = "This is a test message"
        protected = dp_manager.encrypt(original)
        decrypted = dp_manager.decrypt(protected)
        
        assert decrypted.decode('utf-8') == original
        assert protected.protection_level.name == "ENCRYPTED"
        
    def test_pseudonymization(self, dp_manager):
        """Test pseudonymization and verification."""
        email = "test@example.com"
        pseudonym = dp_manager.pseudonymize(email)
        
        # Should verify correctly with original data
        assert dp_manager.verify_pseudonym(email, pseudonym)
        
        # Should not verify with different data
        assert not dp_manager.verify_pseudonym("wrong@example.com", pseudonym)
    
    def test_asymmetric_encryption(self, dp_manager):
        """Test asymmetric encryption and decryption."""
        message = "Secret message for asymmetric encryption"
        
        # Get public key
        public_key = dp_manager.get_public_key()
        
        # Encrypt with public key
        encrypted = dp_manager.encrypt_asymmetric(message, public_key)
        
        # Decrypt with private key (handled internally)
        decrypted = dp_manager.decrypt_asymmetric(encrypted)
        
        assert decrypted.decode('utf-8') == message

# Run tests
if __name__ == "__main__":
    pytest.main(["-v", "tests/test_privacy_preserving.py"])
