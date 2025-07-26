"""
Monitoring API Server Runner

This script starts the monitoring agent API server with proper error handling
and configuration options.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

def setup_logging(log_level=logging.INFO):
    """Configure logging for the script."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/monitoring_api.log')
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run monitoring API server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5001,
                       help='Port to bind to (default: 5001)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Ensure logs directory exists
        Path('logs').mkdir(exist_ok=True)
        
        # Add project root to Python path
        project_root = Path(__file__).parent.parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        logger.info("Attempting to import monitoring API app...")
        from AIComplianceMonitoring.agents.monitoring.api import app
        logger.info("Monitoring API app imported successfully")
        
        logger.info(f"Starting monitoring API server on {args.host}:{args.port}...")
        app.run(
            host=args.host, 
            port=args.port, 
            debug=args.debug,
            use_reloader=False  # Prevent issues with path modifications
        )
        
    except ImportError as e:
        logger.error(f"Failed to import monitoring API app: {e}")
        logger.error("Make sure all dependencies are installed and the project structure is correct")
        sys.exit(1)
    except OSError as e:
        logger.error(f"Failed to start server (port may be in use): {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()

