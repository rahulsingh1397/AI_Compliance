"""
Run a data discovery scan using the DataDiscoveryAgent.
This script demonstrates how to initialize and use the agent with the fixed dependencies.
"""

import os

# Disable tokenizer parallelism to prevent issues with multi-threading
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('results/discovery_scan.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    try:
        # Add the project root to the Python path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        logger.info("Initializing DataDiscoveryAgent...")
        
        # Import the agent
        from AIComplianceMonitoring.agents.data_discovery.agent import DataDiscoveryAgent
        from AIComplianceMonitoring.agents.data_discovery.agent import AgentConfig
        
        # Configure the agent
        config = AgentConfig(
            spacy_model_name="en_core_web_sm",
            max_workers=4  # Adjust based on your system capabilities
        )
        
        # Initialize the agent
        with DataDiscoveryAgent(config=config) as agent:
            logger.info("DataDiscoveryAgent initialized successfully")
            
            # Define the directory to scan
            data_dir = project_root / "AIComplianceMonitoring" / "data"
            logger.info(f"Starting scan of directory: {data_dir}")
            
            # Run the scan using agent.batch_scan_files
            start_time = datetime.now()
            # Include .xlsx in the file extensions to scan
            file_extensions = list(agent.config.default_file_extensions) + ['.xlsx']
            results = agent.batch_scan_files(
                directory_path=str(data_dir),
                file_extensions=file_extensions,
                max_workers=agent.config.max_workers
            )
            duration = datetime.now() - start_time
            
            # Save results
            results_file = project_root / "results" / "discovery_results.json"
            with open(results_file, 'w') as f:
                import json
                json.dump(results, f, indent=2)
            
            # Print summary
            logger.info("\n=== Scan Complete ===")
            logger.info(f"Scanned directory: {data_dir}")
            logger.info(f"Duration: {duration}")
            logger.info(f"Files scanned: {results.get('total_files_scanned', 0)}")
            logger.info(f"Sensitive files found: {results.get('sensitive_files_found', 0)}")
            logger.info(f"Results saved to: {results_file}")
            logger.info("===================")
            
    except Exception as e:
        logger.exception("Error during discovery scan")
        sys.exit(1)

if __name__ == "__main__":
    main()
