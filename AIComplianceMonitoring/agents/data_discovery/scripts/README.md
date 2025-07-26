# Data Discovery Scripts

This directory contains utility scripts for running data discovery scans and related operations.

## Available Scripts

### run_discovery_scan.py

This script allows you to run a data discovery scan from the command line with various configuration options.

Usage:
```
python run_discovery_scan.py [options]
```

Options:
- `--config`: Path to configuration file
- `--output`: Path to output directory
- `--scan-type`: Type of scan to run (quick, full, deep)
- `--verbose`: Enable verbose logging

Example:
```
python run_discovery_scan.py --config=scan_config.yaml --scan-type=full
```

## Integration with UI

These scripts are used by the UI agent to trigger scans from the web interface.
