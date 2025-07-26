# Deprecated User Interface Implementation

This module contains an older implementation of the UI Agent that has been replaced by the more modular implementation in the `ui_agent` directory.

## Why Deprecated?

The implementation in this directory:
- Uses a monolithic approach rather than Flask blueprints
- Lacks the modular structure of the newer implementation
- Doesn't include the latest features (scan triggering, API integration, etc.)

## Migration

All functionality from this module has been migrated to the `ui_agent` directory which:
- Uses a modern blueprint-based architecture
- Has more complete templates and static assets
- Supports more advanced features like data discovery scan triggering

**Note:** This directory is kept for reference purposes only and will be removed in a future release.
