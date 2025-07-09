"""
Data Protection Manager

This module provides data protection utilities including encryption, decryption,
and privacy-preserving data transformations for the privacy-preserving agent.
"""

import os
import json
import base64
import hashlib
import hmac
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum, auto
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.hmac import HMAC as CryptographyHMAC
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives.serialization import (
    load_pem_public_key, 
    load_pem_private_key,
    Encoding,
    PrivateFormat,
    PublicFormat,
    NoEncryption
)

class ProtectionLevel(Enum):
    """Levels of data protection."""
    UNPROTECTED = auto()
    ENCRYPTED = auto()
    PSEUDONYMIZED = auto()
    ANONYMIZED = auto()
    AGGREGATED = auto()

@dataclass
class ProtectedData:
    """Container for protected data with metadata."""
    data: bytes
    protection_level: ProtectionLevel
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary."""
        return {
            'data': base64.b64encode(self.data).decode('utf-8'),
            'protection_level': self.protection_level.name,
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProtectedData':
        """Create from a dictionary."""
        return cls(
            data=base64.b64decode(data['data']),
            protection_level=ProtectionLevel[data['protection_level']],
            metadata=data.get('metadata', {})
        )

class DataProtectionManager:
    """
    Manages data protection operations including encryption, decryption,
    pseudonymization, and anonymization.
    """
    
    def __init__(self, secret_key: Optional[bytes] = None):
        """
        Initialize the data protection manager.
        
        Args:
            secret_key: Optional secret key for encryption. If not provided,
                      a new key will be generated (not recommended for production).
        """
        self.secret_key = secret_key or Fernet.generate_key()
        self.fernet = Fernet(self.secret_key)
        
        # Initialize with default RSA key pair
        self._private_key = None
        self._public_key = None
        self._generate_rsa_keys()
    
    def _generate_rsa_keys(self) -> None:
        """Generate RSA key pair for asymmetric encryption."""
        self._private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self._public_key = self._private_key.public_key()
    
    def get_public_key(self) -> bytes:
        """Get the public key for encryption."""
        return self._public_key.public_bytes(
            encoding=Encoding.PEM,
            format=PublicFormat.SubjectPublicKeyInfo
        )
    
    def encrypt(self, data: Union[str, bytes], protection_level: ProtectionLevel = ProtectionLevel.ENCRYPTED) -> ProtectedData:
        """
        Encrypt data using the appropriate encryption method.
        
        Args:
            data: The data to encrypt (string or bytes)
            protection_level: The desired protection level
            
        Returns:
            ProtectedData containing the encrypted data
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        if protection_level == ProtectionLevel.UNPROTECTED:
            return ProtectedData(data=data, protection_level=protection_level)
            
        if protection_level == ProtectionLevel.ENCRYPTED:
            encrypted = self.fernet.encrypt(data)
            return ProtectedData(data=encrypted, protection_level=protection_level)
            
        if protection_level in [ProtectionLevel.PSEUDONYMIZED, ProtectionLevel.ANONYMIZED]:
            # For demonstration - in a real implementation, you'd use proper pseudonymization
            # or anonymization techniques based on your requirements
            hashed = hashlib.sha256(data).digest()
            return ProtectedData(
                data=hashed,
                protection_level=protection_level,
                metadata={'original_length': len(data)}
            )
            
        raise ValueError(f"Unsupported protection level: {protection_level}")
    
    def decrypt(self, protected_data: ProtectedData) -> bytes:
        """
        Decrypt protected data.
        
        Args:
            protected_data: The protected data to decrypt
            
        Returns:
            The decrypted data as bytes
            
        Raises:
            ValueError: If the data cannot be decrypted or the protection level is not supported
        """
        if protected_data.protection_level == ProtectionLevel.UNPROTECTED:
            return protected_data.data
            
        if protected_data.protection_level == ProtectionLevel.ENCRYPTED:
            try:
                return self.fernet.decrypt(protected_data.data)
            except InvalidToken:
                raise ValueError("Invalid or corrupted data")
        
        if protected_data.protection_level in [ProtectionLevel.PSEUDONYMIZED, 
                                             ProtectionLevel.ANONYMIZED,
                                             ProtectionLevel.AGGREGATED]:
            raise ValueError(f"Cannot reverse {protected_data.protection_level.name} data")
            
        raise ValueError(f"Unsupported protection level: {protected_data.protection_level}")
    
    def pseudonymize(self, data: Union[str, bytes], salt: Optional[bytes] = None) -> ProtectedData:
        """
        Pseudonymize data using a one-way function with optional salt.
        
        Args:
            data: The data to pseudonymize
            salt: Optional salt for the pseudonymization
            
        Returns:
            ProtectedData containing the pseudonymized data
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        if salt is None:
            salt = os.urandom(16)
            
        # Use HMAC for keyed hashing
        h = CryptographyHMAC(
            key=salt,
            algorithm=hashes.SHA256(),
        )
        h.update(data)
        pseudonym = h.finalize()
        
        return ProtectedData(
            data=pseudonym,
            protection_level=ProtectionLevel.PSEUDONYMIZED,
            metadata={
                'salt': base64.b64encode(salt).decode('utf-8'),
                'original_length': len(data)
            }
        )
    
    def verify_pseudonym(self, data: Union[str, bytes], pseudonym: ProtectedData) -> bool:
        """
        Verify if data matches a pseudonym.
        
        Args:
            data: The original data
            pseudonym: The pseudonym to verify against
            
        Returns:
            bool: True if the data matches the pseudonym
        """
        if pseudonym.protection_level != ProtectionLevel.PSEUDONYMIZED:
            return False
            
        if 'salt' not in pseudonym.metadata:
            return False
            
        salt = base64.b64decode(pseudonym.metadata['salt'])
        new_pseudonym = self.pseudonymize(data, salt)
        
        # Use Python's built-in hmac.compare_digest for constant-time comparison
        return hmac.compare_digest(
            new_pseudonym.data,
            pseudonym.data
        )
    
    def encrypt_asymmetric(self, data: Union[str, bytes], public_key: Optional[bytes] = None) -> bytes:
        """
        Encrypt data using RSA public key encryption.
        
        Args:
            data: The data to encrypt
            public_key: Optional public key in PEM format. If not provided, uses the manager's public key.
            
        Returns:
            Encrypted data as bytes
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        key = self._public_key
        if public_key is not None:
            key = load_pem_public_key(public_key)
            
        # RSA encryption has a size limit, so we'll use a hybrid approach:
        # 1. Generate a random AES key
        # 2. Encrypt the data with AES
        # 3. Encrypt the AES key with RSA
        # 4. Return the encrypted AES key + encrypted data
        
        # Generate a random AES key and IV
        aes_key = os.urandom(32)  # 256-bit key
        iv = os.urandom(16)       # 128-bit IV
        
        # Encrypt the data with AES
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        padder = padding.PKCS7(128).padder()
        
        padded_data = padder.update(data) + padder.finalize()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Encrypt the AES key with RSA
        encrypted_key = key.encrypt(
            aes_key + iv,  # Combine key and IV for simplicity
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Return the encrypted key + IV + encrypted data
        return encrypted_key + encrypted_data
    
    def decrypt_asymmetric(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data using the manager's private key.
        
        Args:
            encrypted_data: The data to decrypt
            
        Returns:
            Decrypted data as bytes
            
        Raises:
            ValueError: If decryption fails
        """
        if self._private_key is None:
            raise ValueError("No private key available for decryption")
            
        try:
            # The first 256 bytes are the encrypted AES key + IV
            encrypted_key = encrypted_data[:256]
            ciphertext = encrypted_data[256:]
            
            # Decrypt the AES key and IV
            key_iv = self._private_key.decrypt(
                encrypted_key,
                asym_padding.OAEP(
                    mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Extract the AES key and IV
            aes_key = key_iv[:32]
            iv = key_iv[32:48]  # 16 bytes IV
            
            # Decrypt the data
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
            decryptor = cipher.decryptor()
            
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Unpad the data
            unpadder = padding.PKCS7(128).unpadder()
            return unpadder.update(padded_data) + unpadder.finalize()
            
        except Exception as e:
            raise ValueError(f"Decryption failed: {str(e)}")
