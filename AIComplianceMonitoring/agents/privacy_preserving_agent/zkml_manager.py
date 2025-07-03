import os
import json
import torch
import logging
import subprocess
from .simple_model import create_model_and_input

# Configure logging
log = logging.getLogger(__name__)

class ZKMLManager:
    """Manages the full zkML workflow using the ezkl library."""

    def __init__(self, ezkl_path, working_dir='zk_data'):
        if not os.path.exists(ezkl_path):
            raise FileNotFoundError(f"ezkl executable not found at: {ezkl_path}")
        self.ezkl_path = ezkl_path
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)
        self.model_path = os.path.join(self.working_dir, 'network.onnx')
        self.input_path = os.path.join(self.working_dir, 'input.json')
        self.settings_path = os.path.join(self.working_dir, 'settings.json')
        self.compiled_model_path = os.path.join(self.working_dir, 'network.compiled')
        self.pk_path = os.path.join(self.working_dir, 'key.pk')
        self.vk_path = os.path.join(self.working_dir, 'key.vk')
        self.proof_path = os.path.join(self.working_dir, 'proof.json')

    def _run_ezkl_command(self, command):
        """Helper function to run an ezkl CLI command and handle output."""
        log.info(f"Running command: {' '.join(command)}")
        try:
            process = subprocess.run(command, check=True, capture_output=True, text=True)
            log.info(f"ezkl stdout: {process.stdout}")
            if process.stderr:
                log.warning(f"ezkl stderr: {process.stderr}")
            return True
        except subprocess.CalledProcessError as e:
            log.error(f"ezkl command failed with exit code {e.returncode}")
            log.error(f"Stderr: {e.stderr}")
            log.error(f"Stdout: {e.stdout}")
            return False

    def export_model_and_input(self):
        """Exports the PyTorch model to ONNX and creates a sample input file."""
        model, dummy_input = create_model_and_input()
        torch.onnx.export(model, dummy_input, self.model_path, export_params=True, do_constant_folding=True)
        
        # Serialize the dummy input to a JSON file
        input_data = dummy_input.numpy().tolist()
        json_data = {'input_data': [input_data]}
        with open(self.input_path, 'w') as f:
            json.dump(json_data, f)
        log.info(f"Model exported to {self.model_path} and input to {self.input_path}")

    def generate_circuit_settings(self):
        """Generates and calibrates the circuit settings file."""
        command = [
            self.ezkl_path,
            'gen-settings',
            '-M', self.model_path,
            '--settings-path', self.settings_path,
            '--input-visibility', 'public'
        ]
        if not self._run_ezkl_command(command):
            return False
        
        # Calibrate the settings to optimize for smaller proofs
        command = [
            self.ezkl_path,
            'calibrate-settings',
            '-M', self.model_path,
            '-D', self.input_path,
            '--settings-path', self.settings_path,
            '--target', 'resources'
        ]
        return self._run_ezkl_command(command)

    def compile_circuit(self):
        """Compiles the model into a zk-SNARK circuit."""
        command = [
            self.ezkl_path,
            'compile-circuit',
            '-M', self.model_path,
            '--settings-path', self.settings_path,
            '--compiled-circuit', self.compiled_model_path
        ]
        return self._run_ezkl_command(command)

    def setup_srs_and_keys(self):
        """Performs the trusted setup (for demo) and generates proving/verifying keys."""
        # Generate the Structured Reference String (SRS) for the circuit
        command = [
            self.ezkl_path,
            'get-srs',
            '--settings-path', self.settings_path
        ]
        if not self._run_ezkl_command(command):
            return False

        # Setup the proving and verification keys
        command = [
            self.ezkl_path,
            'setup',
            '--compiled-circuit', self.compiled_model_path,
            '--vk-path', self.vk_path,
            '--pk-path', self.pk_path,
        ]
        return self._run_ezkl_command(command)

    def generate_proof(self):
        """Generates a proof for the given input."""
        command = [
            self.ezkl_path,
            'prove',
            '--compiled-circuit', self.compiled_model_path,
            '-D', self.input_path,
            '--pk-path', self.pk_path,
            '--proof-path', self.proof_path,
        ]
        return self._run_ezkl_command(command)

    def verify_proof(self):
        """Verifies the generated proof."""
        command = [
            self.ezkl_path,
            'verify',
            '--proof-path', self.proof_path,
            '--settings-path', self.settings_path,
            '--vk-path', self.vk_path,
        ]
        return self._run_ezkl_command(command)
