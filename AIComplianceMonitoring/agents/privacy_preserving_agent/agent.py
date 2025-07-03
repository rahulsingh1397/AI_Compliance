import logging
import os
import json
from .audit_log import SecureAuditLog
from .zkml_manager import ZKMLManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PrivacyPreservingAgent:
    """An agent that uses zkML for verifiable, private computation and logs operations."""

    def __init__(self, ezkl_path, audit_log_path='privacy_audit.log', zk_working_dir='zk_data'):
        self.audit_log = SecureAuditLog(audit_log_path)
        self.zkml_manager = ZKMLManager(ezkl_path=ezkl_path, working_dir=zk_working_dir)
        logging.info("Privacy-Preserving Agent initialized.")
        self.audit_log.verify_log_integrity()

    def perform_verifiable_inference(self, data_id):
        """Runs the full zkML workflow and logs the proof."""
        logging.info(f"--- Starting Verifiable Inference for data_id: {data_id} ---")
        
        # 1. Export Model and Input
        self.zkml_manager.export_model_and_input()
        
        # 2. Generate and Calibrate Circuit
        if not self.zkml_manager.generate_circuit_settings(): return False
        
        # 3. Compile Circuit
        if not self.zkml_manager.compile_circuit(): return False
        
        # 4. Setup Keys
        if not self.zkml_manager.setup_srs_and_keys(): return False
        
        # 5. Generate Proof
        if not self.zkml_manager.generate_proof(): return False
        
        # 6. Verify Proof
        if not self.zkml_manager.verify_proof():
            logging.error("Verification of the generated proof failed!")
            return False
        
        logging.info("Proof generated and verified successfully.")

        # 7. Log the successful verification event
        with open(self.zkml_manager.proof_path, 'r') as f:
            proof_data = json.load(f)
            
        event_data = {
            'data_id': data_id,
            'analysis_type': 'zkml_inference',
            'model_path': self.zkml_manager.model_path,
            'verification_key_path': self.zkml_manager.vk_path,
            'proof': proof_data
        }
        event_hash = self.audit_log.log_event('VerifiableInferenceComplete', event_data)
        
        if event_hash:
            logging.info(f"Successfully logged verifiable inference. Log hash: {event_hash[:10]}...")
        else:
            logging.error("Failed to log the verifiable inference event.")
        
        return True

    def run_demonstration(self):
        """Runs a simple demonstration of the agent's capabilities."""
        logging.info("--- Starting Privacy-Preserving Agent Demonstration ---")
        
        # Perform a verifiable inference
        success = self.perform_verifiable_inference(data_id='report_xyz_123')
        
        if success:
            print("\n[SUCCESS] The zkML workflow completed successfully.")
        else:
            print("\n[FAILURE] The zkML workflow encountered an error.")

        # Final integrity check of the audit log
        logging.info("--- Verifying final log integrity ---")
        is_valid = self.audit_log.verify_log_integrity()
        if is_valid:
            print("[SUCCESS] The audit log is valid and has not been tampered with.")
        else:
            print("[FAILURE] The audit log has been tampered with or is corrupted!")
            
        logging.info("--- Demonstration Complete ---")

if __name__ == '__main__':
    print("--- AGENT DEMONSTRATION SCRIPT STARTED ---")
    # To run, navigate to the project root and execute:
    # python -m AIComplianceMonitoring.agents.privacy_preserving_agent.agent
    
    # Define paths relative to this script's location
    base_dir = os.path.dirname(__file__)
    log_file = os.path.join(base_dir, 'privacy_audit.log')
    zk_dir = os.path.join(base_dir, 'zk_data')
    
    # Define the path to the ezkl executable provided by the user
    ezkl_executable_path = r"E:\Projects\Private\build-artifacts.ezkl-windows-msvc\ezkl"

    agent = PrivacyPreservingAgent(ezkl_path=ezkl_executable_path, audit_log_path=log_file, zk_working_dir=zk_dir)
    agent.run_demonstration()
