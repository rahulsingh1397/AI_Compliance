# Reporting Scripts

This directory contains utility scripts for generating compliance reports and related operations.

## Available Scripts

### generate_report.py

This script allows you to generate compliance reports from the command line with various configuration options.

Usage:
```
python generate_report.py [options]
```

Options:
- `--config`: Path to configuration file
- `--output`: Path to output directory
- `--format`: Report format (pdf, html, json, csv)
- `--period`: Reporting period (daily, weekly, monthly, quarterly)
- `--verbose`: Enable verbose logging

Example:
```
python generate_report.py --config=report_config.yaml --format=pdf --period=monthly
```

## Integration with UI

These scripts are used by the UI agent and ReportingAgent to generate reports from the web interface and scheduled tasks.
