import json
import hashlib
import logging
from datetime import datetime, timezone

# Configure logging
log = logging.getLogger(__name__)

class SecureAuditLog:
    """Creates a tamper-proof audit trail using a blockchain-like structure."""

    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self._initialize_log()

    def _initialize_log(self):
        """Initializes the log file with a genesis block if it doesn't exist."""
        try:
            with open(self.log_file_path, 'r') as f:
                # If the file is not empty, assume it's initialized
                if f.read(1):
                    return
            
            # If the file is empty or doesn't exist, create the genesis block
            genesis_block = {
                'index': 0,
                'timestamp': self._get_utc_timestamp(),
                'event': 'Log Initialized - Genesis Block',
                'data': {},
                'previous_hash': '0' * 64,
            }
            genesis_block['hash'] = self._hash_block(genesis_block)
            
            with open(self.log_file_path, 'w') as f:
                f.write(json.dumps(genesis_block) + '\n')
            log.info(f"Audit log initialized with genesis block at {self.log_file_path}")

        except FileNotFoundError:
            # This handles the case where the file doesn't exist yet
            self._create_genesis_block()

    def _create_genesis_block(self):
        """Helper to create the very first block in the log chain."""
        genesis_block = {
            'index': 0,
            'timestamp': self._get_utc_timestamp(),
            'event': 'Log Initialized - Genesis Block',
            'data': {},
            'previous_hash': '0' * 64,
        }
        genesis_block['hash'] = self._hash_block(genesis_block)
        
        with open(self.log_file_path, 'w') as f:
            f.write(json.dumps(genesis_block) + '\n')
        log.info(f"Audit log created with genesis block at {self.log_file_path}")

    def _get_last_block(self):
        """Retrieves the most recent block from the log file."""
        last_line = None
        try:
            with open(self.log_file_path, 'r') as f:
                for line in f:
                    last_line = line
            if last_line:
                return json.loads(last_line)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            log.error(f"Could not read or parse last block from audit log: {e}")
            return None
        return None # Should not be reached if genesis block logic is sound

    def _hash_block(self, block):
        """Hashes a block using SHA-256, ensuring consistent ordering."""
        # Create a deep copy to avoid modifying the original block
        block_copy = block.copy()
        # Remove the 'hash' field itself before hashing
        if 'hash' in block_copy:
            del block_copy['hash']
            
        # Use sort_keys=True to ensure the hash is always consistent
        block_string = json.dumps(block_copy, sort_keys=True).encode('utf-8')
        return hashlib.sha256(block_string).hexdigest()

    def log_event(self, event_type, event_data):
        """Logs a new event to the secure audit trail."""
        last_block = self._get_last_block()
        if not last_block:
            log.error("Cannot log event: audit log is not initialized or is corrupted.")
            return None

        new_block = {
            'index': last_block['index'] + 1,
            'timestamp': self._get_utc_timestamp(),
            'event': event_type,
            'data': event_data,
            'previous_hash': last_block['hash'],
        }
        new_block['hash'] = self._hash_block(new_block)

        with open(self.log_file_path, 'a') as f:
            f.write(json.dumps(new_block) + '\n')
        
        log.info(f"Logged new event '{event_type}' with index {new_block['index']}.")
        return new_block['hash']

    def verify_log_integrity(self):
        """Verifies the integrity of the entire audit log chain."""
        try:
            with open(self.log_file_path, 'r') as f:
                lines = f.readlines()

            previous_block_hash = '0' * 64
            for i, line in enumerate(lines):
                block = json.loads(line)
                
                # Check genesis block separately
                if block['index'] == 0:
                    if block['previous_hash'] != previous_block_hash:
                        log.warning(f"Integrity check failed: Genesis block has incorrect previous_hash.")
                        return False
                # Check subsequent blocks
                elif block['previous_hash'] != previous_block_hash:
                    log.warning(f"Integrity check failed: Block {block['index']} has a mismatched hash. Expected {previous_block_hash}, got {block['previous_hash']}.")
                    return False

                # Verify the block's own hash
                if block['hash'] != self._hash_block(block):
                    log.warning(f"Integrity check failed: Block {block['index']} content has been tampered with.")
                    return False
                
                previous_block_hash = block['hash']
            
            log.info("Audit log integrity verified successfully.")
            return True

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            log.error(f"Failed to verify audit log due to an error: {e}")
            return False

    def _get_utc_timestamp(self):
        """Returns the current UTC time in ISO 8601 format."""
        return datetime.now(timezone.utc).isoformat()
