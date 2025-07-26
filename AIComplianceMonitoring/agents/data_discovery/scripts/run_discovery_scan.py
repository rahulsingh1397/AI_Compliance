"""
Run a data discovery scan using the DataDiscoveryAgent.

This script provides a command-line interface for running data discovery scans
to identify sensitive data across file systems.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Disable tokenizer parallelism to prevent threading issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

def setup_logging(log_level=logging.INFO, log_file=None):
    """Configure logging for the script."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=handlers
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run data discovery scan')
    parser.add_argument('--data-dir', type=str, 
                       help='Directory to scan (default: project data directory)')
    parser.add_argument('--output-file', type=str, 
                       help='Output file for results (default: results/discovery_results.json)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of worker threads (default: 4)')
    parser.add_argument('--log-file', type=str, 
                       help='Log file path (default: results/discovery_scan.log)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = args.log_file or 'results/discovery_scan.log'
    setup_logging(log_level, log_file)
    
    try:
        # Add project root to Python path
        project_root = Path(__file__).parent.parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        logger.info("Initializing DataDiscoveryAgent...")
        
        # Import agent components
        from AIComplianceMonitoring.agents.data_discovery.agent import DataDiscoveryAgent, AgentConfig
        
        # Configure the agent
        config = AgentConfig(
            spacy_model_name="en_core_web_sm",
            max_workers=args.max_workers
        )
        
        # Determine scan directory
        if args.data_dir:
            data_dir = Path(args.data_dir)
        else:
            data_dir = project_root / "AIComplianceMonitoring" / "data"
        
        if not data_dir.exists():
            logger.error(f"Scan directory does not exist: {data_dir}")
            sys.exit(1)
        
        # Initialize and run the agent
        with DataDiscoveryAgent(config=config) as agent:
            logger.info("DataDiscoveryAgent initialized successfully")
            logger.info(f"Starting scan of directory: {data_dir}")
            
            start_time = datetime.now()
            
            # Run the scan
            file_extensions = list(agent.config.default_file_extensions) + ['.xlsx']
            results = agent.batch_scan_files(
                directory_path=str(data_dir),
                file_extensions=file_extensions,
                max_workers=config.max_workers
            )
            
            duration = datetime.now() - start_time
            
            # Save results
            output_file = Path(args.output_file) if args.output_file else Path('results/discovery_results.json')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Print summary
            logger.info("\n=== Scan Complete ===")
            logger.info(f"Scanned directory: {data_dir}")
            logger.info(f"Duration: {duration}")
            logger.info(f"Files scanned: {results.get('total_files_scanned', 0)}")
            logger.info(f"Sensitive files found: {results.get('sensitive_files_found', 0)}")
            logger.info(f"Results saved to: {output_file}")
            logger.info("===================")
            
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Make sure the project is properly installed and dependencies are available")
        sys.exit(1)
    except Exception as e:
        logger.exception("Error during discovery scan")
        sys.exit(1)

if __name__ == "__main__":
    main()
