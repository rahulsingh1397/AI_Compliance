import sys
import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'tests', 'test_results', 'original_agent_test.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

@dataclass
class AgentConfig:
    """Simple configuration class for the agent"""
    db_uri: str = None
    bert_classifier_model: str = "bert-base-uncased"
    nlp_model: str = "en_core_web_sm"
    batch_size: int = 10
    db_pool_size: int = 5
    db_max_overflow: int = 10
    sample_size: int = 1000

try:
    # Import the agent module
    logger.info("Attempting to import DataDiscoveryAgent...")
    from AIComplianceMonitoring.agents.data_discovery.agent import DataDiscoveryAgent
    logger.info("DataDiscoveryAgent imported successfully.")

    # Create configuration
    logger.info("Creating agent configuration...")
    config = AgentConfig()
    
    # Initialize the agent with fixed dependencies
    logger.info("Initializing DataDiscoveryAgent...")
    agent = DataDiscoveryAgent(config=config)
    logger.info("DataDiscoveryAgent initialized successfully.")
    
    # Run a scan
    logger.info("Running file scan on data directory...")
    target_dir = os.path.join(project_root, "AIComplianceMonitoring", "data")
    
    logger.info(f"Target directory: {target_dir}")
    start_time = time.time()
    scan_results = agent.scan_directory(target_dir)
    end_time = time.time()
    
    # Output results
    logger.info(f"Scan completed in {end_time - start_time:.2f} seconds.")
    logger.info(f"Files scanned: {len(scan_results)}")
    
    # Save results to file
    results_path = os.path.join(project_root, 'tests', 'test_results', 'original_agent_results.json')
    with open(results_path, "w") as f:
        json.dump({
            "scan_time": end_time - start_time,
            "files_scanned": len(scan_results),
            "results": scan_results
        }, f, indent=2)
    
    logger.info(f"Scan results saved to '{results_path}'")
    
    print("\n--- Scan Summary ---")
    print(f"Total files scanned: {len(scan_results)}")
    sensitive_files = [f for f in scan_results if scan_results[f].get('is_sensitive', False)]
    print(f"Sensitive files found: {len(sensitive_files)}")
    print("--- End of Summary ---")
    
except Exception as e:
    logger.exception(f"Error during test: {e}")
    print(f"Error: {e}. See agent_test.log for details.")
