"""
Log Ingestion Module for the Monitoring Agent.

This module handles ingestion of logs from various sources:
- AWS S3 buckets
- Azure Blob storage
- On-premises log files

Features:
- Parallel log processing
- Incremental processing (avoid reprocessing the same logs)
- Format normalization for consistent downstream processing
- Reconnection and retry logic for unreliable sources
- State persistence for tracking progress across restarts
"""

import os
import logging
import time
import json
import hashlib
import threading
import queue
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

# Placeholder imports for cloud services
# In production, these would be actual imports:
# import boto3  # For AWS S3
# import azure.storage.blob  # For Azure Blob

# Configure logging
logger = logging.getLogger(__name__)

# Define standard log schema for normalization
STANDARD_LOG_SCHEMA = {
    "timestamp": "datetime64[ns]",
    "source": "str",
    "source_type": "str",
    "log_level": "str",
    "message": "str",
    "user": "str",
    "resource": "str",
    "action": "str",
    "status": "str",
    "metadata": "object"
}

class SourceConfig:
    """Configuration for a log source"""
    def __init__(self, 
                 source_type: str,
                 name: str,
                 credentials: Dict[str, Any],
                 path_pattern: str,
                 format: str,
                 max_age_days: int = 30,
                 **kwargs):
        self.source_type = source_type  # aws_s3, azure_blob, on_prem
        self.name = name  # Friendly name
        self.credentials = credentials  # Auth credentials
        self.path_pattern = path_pattern  # Path/prefix/pattern to locate logs
        self.format = format  # Log format (json, csv, syslog)
        self.max_age_days = max_age_days  # How far back to fetch logs
        self.extra_config = kwargs  # Additional source-specific config

class ProcessingState:
    """State for tracking log processing progress"""
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.last_processed_timestamp = None  # Last processed log timestamp
        self.last_processed_file = None  # Last processed file name
        self.last_processed_position = 0  # Position within last file
        self.last_run_time = None  # When we last processed this source
        self.processed_file_hashes = {}  # Map of filename -> hash for processed files

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for persistence"""
        return {
            "source_name": self.source_name,
            "last_processed_timestamp": self.last_processed_timestamp.isoformat() if self.last_processed_timestamp else None,
            "last_processed_file": self.last_processed_file,
            "last_processed_position": self.last_processed_position,
            "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None,
            "processed_file_hashes": self.processed_file_hashes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingState':
        """Create state from dictionary"""
        state = cls(data["source_name"])
        if data.get("last_processed_timestamp"):
            state.last_processed_timestamp = datetime.fromisoformat(data["last_processed_timestamp"])
        state.last_processed_file = data.get("last_processed_file")
        state.last_processed_position = data.get("last_processed_position", 0)
        if data.get("last_run_time"):
            state.last_run_time = datetime.fromisoformat(data["last_run_time"])
        state.processed_file_hashes = data.get("processed_file_hashes", {})
        return state

class LogIngestionModule:
    """
    Handles ingestion of logs from various sources.
    """
    
    def __init__(self, config):
        """
        Initialize the log ingestion module.
        
        Args:
            config: Configuration object with necessary parameters
        """
        logger.debug("Initializing LogIngestionModule")
        self.config = config
        self.state_dir = Path(config.state_directory)
        self.state_file = self.state_dir / "log_ingestion_state.json"
        self.executor = None  # ThreadPoolExecutor for parallel processing
        
        # Create state directory if it doesn't exist
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize connections and state maps
        self.connections = {}
        self.processing_states = {}
        
        # Initialize statistics
        self.stats = {
            "last_ingestion_time": None,
            "total_logs_processed": 0,
            "sources_connected": 0,
            "sources_failed": 0,
            "errors": [],
            "source_stats": {}
        }
        
        # Load previous state if available
        self._load_state()
        
        # Initialize thread pool
        self.max_workers = config.max_ingestion_threads or 10
    
    def initialize_connections(self):
        """Initialize connections to all configured data sources"""
        logger.info("Initializing connections to log sources")
        
        # Initialize thread pool
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Get source configurations
        sources = self.config.log_sources if hasattr(self.config, 'log_sources') else []
        
        for source_config in sources:
            try:
                # Initialize connection based on source type
                if source_config.source_type == 'aws_s3':
                    self.connections[source_config.name] = self._init_aws_s3_connection(source_config)
                elif source_config.source_type == 'azure_blob':
                    self.connections[source_config.name] = self._init_azure_blob_connection(source_config)
                elif source_config.source_type == 'on_prem':
                    self.connections[source_config.name] = self._init_on_prem_connection(source_config)
                else:
                    logger.warning(f"Unknown source type: {source_config.source_type}")
                    continue
                    
                # Initialize processing state for this source if not exists
                if source_config.name not in self.processing_states:
                    self.processing_states[source_config.name] = ProcessingState(source_config.name)
                    
                # Update stats
                self.stats["sources_connected"] += 1
                
                logger.info(f"Successfully connected to source: {source_config.name}")
                
            except Exception as e:
                logger.error(f"Failed to connect to source {source_config.name}: {str(e)}")
                self.stats["sources_failed"] += 1
                self.stats["errors"].append({
                    "timestamp": datetime.now(),
                    "source": source_config.name,
                    "error": str(e)
                })
                
        logger.info(f"Connections initialized. Connected: {self.stats['sources_connected']}, Failed: {self.stats['sources_failed']}")
        
    def close_connections(self):
        """Close all active connections and resources"""
        logger.info("Closing connections to log sources")
        
        # Save current state
        self._save_state()
        
        # Close thread pool
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        
        # Close all connections
        for name, connection in self.connections.items():
            try:
                # Different close method depending on source type
                if hasattr(connection, 'close'):
                    connection.close()
                    
                logger.debug(f"Closed connection to {name}")
            except Exception as e:
                logger.error(f"Error closing connection to {name}: {str(e)}")
                
        self.connections = {}
        logger.info("All connections closed")
    
    def _load_state(self):
        """Load processing state from disk"""
        try:
            if self.state_file.exists():
                logger.info(f"Loading state from {self.state_file}")
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                    
                # Process each source state
                for source_name, source_state in state_data.get('sources', {}).items():
                    self.processing_states[source_name] = ProcessingState.from_dict(source_state)
                    
                # Load stats
                if 'stats' in state_data:
                    # Only copy specific stats to avoid overwriting the structure
                    self.stats['total_logs_processed'] = state_data['stats'].get('total_logs_processed', 0)
                    self.stats['last_ingestion_time'] = state_data['stats'].get('last_ingestion_time')
                    
                logger.info(f"State loaded for {len(self.processing_states)} sources")
        except Exception as e:
            logger.warning(f"Failed to load state: {str(e)}")
            # Create an empty state file for future use
            self._save_state()
    
    def _save_state(self):
        """Save processing state to disk for persistence"""
        try:
            logger.info(f"Saving state to {self.state_file}")
            
            # Prepare state data
            state_data = {
                'last_updated': datetime.now().isoformat(),
                'sources': {},
                'stats': {
                    'total_logs_processed': self.stats['total_logs_processed'],
                    'last_ingestion_time': self.stats['last_ingestion_time']
                }
            }
            
            # Add each source's state
            for name, state in self.processing_states.items():
                state_data['sources'][name] = state.to_dict()
            
            # Write to file
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
            logger.info(f"State saved for {len(self.processing_states)} sources")
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")
    
    def get_available_sources(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all available log sources based on configuration.
        
        Returns:
            Dict with source types as keys and lists of source configs as values
        """
        sources = {
            "aws_s3": [],
            "azure_blob": [],
            "on_prem": []
        }
        
        # Process configured sources
        configured_sources = self.config.log_sources if hasattr(self.config, 'log_sources') else []
        for source in configured_sources:
            if source.source_type in sources:
                # Convert source config to dict for API response
                source_info = {
                    "name": source.name,
                    "path_pattern": source.path_pattern,
                    "format": source.format,
                    "max_age_days": source.max_age_days,
                    "connected": source.name in self.connections
                }
                
                # Add source state if available
                if source.name in self.processing_states:
                    state = self.processing_states[source.name]
                    source_info["last_processed"] = {
                        "timestamp": state.last_processed_timestamp,
                        "file": state.last_processed_file,
                        "run_time": state.last_run_time
                    }
                
                sources[source.source_type].append(source_info)
        
        return sources
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _init_aws_s3_connection(self, source_config):
        """Initialize AWS S3 connection with retry logic"""
        logger.info(f"Connecting to AWS S3 source: {source_config.name}")
        
        # In a real implementation, this would use boto3 to connect to S3
        # For now, using a placeholder implementation
        
        # Example boto3 usage (commented out):
        # import boto3
        # session = boto3.Session(
        #     aws_access_key_id=source_config.credentials.get('access_key'),
        #     aws_secret_access_key=source_config.credentials.get('secret_key'),
        #     region_name=source_config.credentials.get('region')
        # )
        # s3_client = session.client('s3')
        # return s3_client
        
        # Placeholder return
        return {
            "type": "aws_s3",
            "name": source_config.name,
            "bucket": source_config.extra_config.get('bucket'),
            "connected": True
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _init_azure_blob_connection(self, source_config):
        """Initialize Azure Blob connection with retry logic"""
        logger.info(f"Connecting to Azure Blob source: {source_config.name}")
        
        # In a real implementation, this would use Azure SDK to connect to Blob storage
        # For now, using a placeholder implementation
        
        # Example Azure Blob usage (commented out):
        # from azure.storage.blob import BlobServiceClient
        # connection_string = source_config.credentials.get('connection_string')
        # blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        # return blob_service_client
        
        # Placeholder return
        return {
            "type": "azure_blob",
            "name": source_config.name,
            "container": source_config.extra_config.get('container'),
            "connected": True
        }
    
    def _init_on_prem_connection(self, source_config):
        """Initialize on-premises filesystem connection"""
        logger.info(f"Setting up on-premises source: {source_config.name}")
        
        # For on-premises sources, we verify the path exists
        path = source_config.path_pattern
        if os.path.exists(path):
            logger.info(f"Successfully located on-prem path: {path}")
        else:
            logger.warning(f"On-prem path does not exist: {path}")
            
        # Placeholder return
        return {
            "type": "on_prem",
            "name": source_config.name,
            "path": path,
            "connected": os.path.exists(path)
        }
    
    def ingest_logs(self, source_type: str, source_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Ingest logs from a specific source.
        
        Args:
            source_type: Type of the source (aws_s3, azure_blob, on_prem)
            source_config: Configuration for the source
            
        Returns:
            DataFrame containing the ingested logs
        """
        logger.info(f"Ingesting logs from {source_type}: {source_config.get('name', 'unnamed')}")
        
        # Convert dict config to SourceConfig object if needed
        if isinstance(source_config, dict):
            # Create a SourceConfig from dictionary
            config_obj = SourceConfig(
                source_type=source_type,
                name=source_config.get('name', f"unnamed-{source_type}"),
                credentials=source_config.get('credentials', {}),
                path_pattern=source_config.get('path_pattern', ''),
                format=source_config.get('format', 'json'),
                max_age_days=source_config.get('max_age_days', 30)
            )
        else:
            config_obj = source_config
        
        # Get state for this source
        source_name = config_obj.name
        state = self.processing_states.get(source_name) or ProcessingState(source_name)
        
        # Choose the appropriate ingestion method based on source type
        if source_type == 'aws_s3':
            logs_df = self._ingest_from_aws_s3(config_obj, state)
        elif source_type == 'azure_blob':
            logs_df = self._ingest_from_azure_blob(config_obj, state)
        elif source_type == 'on_prem':
            logs_df = self._ingest_from_on_prem(config_obj, state)
        else:
            logger.error(f"Unknown source type: {source_type}")
            return pd.DataFrame()
        
        # Update state
        if not logs_df.empty:
            state.last_run_time = datetime.now()
            if 'timestamp' in logs_df.columns:
                state.last_processed_timestamp = logs_df['timestamp'].max()
            
            # Update stats
            self.stats['total_logs_processed'] += len(logs_df)
            self.stats['last_ingestion_time'] = datetime.now().isoformat()
            
            if source_name not in self.stats['source_stats']:
                self.stats['source_stats'][source_name] = {}
            
            self.stats['source_stats'][source_name]['logs_processed'] = \
                self.stats['source_stats'][source_name].get('logs_processed', 0) + len(logs_df)
            self.stats['source_stats'][source_name]['last_ingestion'] = datetime.now().isoformat()
        
        # Save processing state
        self.processing_states[source_name] = state
        self._save_state()
        
        return logs_df
    
    def _ingest_from_aws_s3(self, config: SourceConfig, state: ProcessingState) -> pd.DataFrame:
        """Ingest logs from AWS S3"""
        logger.info(f"Ingesting from AWS S3: {config.name}, Pattern: {config.path_pattern}")
        
        # Placeholder implementation
        # In a real implementation, we would:
        # 1. Use boto3 to list objects in the bucket matching the prefix/pattern
        # 2. Filter objects based on last modified time and state
        # 3. Download new objects and parse them based on the format
        # 4. Update state with information about processed files
        
        # Generate some mock log data
        return self._generate_mock_logs(config.name, 'aws_s3', 50)
    
    def _ingest_from_azure_blob(self, config: SourceConfig, state: ProcessingState) -> pd.DataFrame:
        """Ingest logs from Azure Blob Storage"""
        logger.info(f"Ingesting from Azure Blob: {config.name}, Pattern: {config.path_pattern}")
        
        # Placeholder implementation
        # In a real implementation, we would:
        # 1. Use Azure SDK to list blobs matching the prefix/pattern
        # 2. Filter blobs based on last modified time and state
        # 3. Download new blobs and parse them based on the format
        # 4. Update state with information about processed files
        
        # Generate some mock log data
        return self._generate_mock_logs(config.name, 'azure_blob', 30)
    
    def _ingest_from_on_prem(self, config: SourceConfig, state: ProcessingState) -> pd.DataFrame:
        """Ingest logs from on-premises file system"""
        logger.info(f"Ingesting from on-premises: {config.name}, Path: {config.path_pattern}")
        
        # Placeholder implementation
        # In a real implementation, we would:
        # 1. Use glob or similar to find files matching the pattern
        # 2. Filter files based on modification time and state
        # 3. Read and parse files based on the format
        # 4. Update state with information about processed files
        
        # Generate some mock log data
        return self._generate_mock_logs(config.name, 'on_prem', 20)
    
    def _normalize_logs(self, logs_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize logs to a standard format"""
        logger.debug(f"Normalizing {len(logs_df)} log entries")
        
        # Ensure all required columns are present
        for column, dtype in STANDARD_LOG_SCHEMA.items():
            if column not in logs_df.columns:
                # Add missing columns with default values based on dtype
                if dtype == 'datetime64[ns]':
                    logs_df[column] = pd.NaT
                elif dtype == 'object':
                    logs_df[column] = None
                elif dtype == 'str':
                    logs_df[column] = ''
                else:
                    logs_df[column] = None
            
            # Convert column to appropriate dtype if possible
            try:
                if dtype != 'object':
                    logs_df[column] = logs_df[column].astype(dtype)
            except Exception as e:
                logger.warning(f"Failed to convert column {column} to dtype {dtype}: {str(e)}")
        
        return logs_df
    
    def _generate_mock_logs(self, source_name: str, source_type: str, count: int) -> pd.DataFrame:
        """Generate mock log data for testing and development"""
        logger.debug(f"Generating {count} mock logs for {source_name}")
        
        log_levels = ['INFO', 'WARNING', 'ERROR', 'DEBUG', 'CRITICAL']
        actions = ['LOGIN', 'LOGOUT', 'CREATE', 'READ', 'UPDATE', 'DELETE', 'EXPORT']
        statuses = ['SUCCESS', 'FAILURE', 'PENDING', 'IN_PROGRESS']
        resources = ['USER_DATA', 'PAYMENT_INFO', 'MEDICAL_RECORD', 'PERSONAL_INFO', 'ANALYTICS']
        users = [f'user{i}' for i in range(1, 6)]
        
        now = datetime.now()
        
        # Generate log entries
        logs = []
        for i in range(count):
            timestamp = now - timedelta(minutes=i*5)  # Spread logs over time
            log_level = np.random.choice(log_levels, p=[0.6, 0.2, 0.1, 0.05, 0.05])
            action = np.random.choice(actions)
            status = np.random.choice(statuses, p=[0.7, 0.15, 0.1, 0.05])
            resource = np.random.choice(resources)
            user = np.random.choice(users)
            
            # Different message patterns based on log level
            if log_level == 'ERROR':
                message = f"Failed to {action} {resource}: Access denied"
            elif log_level == 'WARNING':
                message = f"Unusual {action} activity on {resource}"
            else:
                message = f"User {user} performed {action} on {resource}"
            
            # Build log entry
            log_entry = {
                'timestamp': timestamp,
                'source': source_name,
                'source_type': source_type,
                'log_level': log_level,
                'message': message,
                'user': user,
                'resource': resource,
                'action': action,
                'status': status,
                'metadata': {
                    'client_ip': f"192.168.1.{i % 255}",
                    'session_id': f"sess-{hashlib.md5(f'{user}-{i}'.encode()).hexdigest()[:8]}"
                }
            }
            
            logs.append(log_entry)
        
        # Convert to DataFrame and normalize
        logs_df = pd.DataFrame(logs)
        return self._normalize_logs(logs_df)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about log ingestion"""
        return {
            **self.stats,
            "timestamp": time.time()
        }
    
    def process_in_parallel(self, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process multiple log sources in parallel using thread pool.
        
        Args:
            sources: List of source configurations to process. If None, use all configured sources.
            
        Returns:
            Dictionary with processing results and statistics
        """
        logger.info("Starting parallel log processing")
        
        if not sources:
            # Get all configured sources
            all_sources = self.config.log_sources if hasattr(self.config, 'log_sources') else []
            sources = [{
                'source_type': source.source_type,
                'name': source.name,
                'path_pattern': source.path_pattern,
                'format': source.format,
                'max_age_days': source.max_age_days,
                'credentials': source.credentials
            } for source in all_sources]
        
        if not self.executor:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Submit tasks to thread pool
        future_to_source = {}
        for source in sources:
            source_type = source['source_type']
            source_name = source['name']
            
            logger.debug(f"Submitting task for source: {source_name}")
            future = self.executor.submit(self.ingest_logs, source_type, source)
            future_to_source[future] = source
        
        # Process results as they complete
        results = {
            'sources_processed': 0,
            'sources_failed': 0,
            'total_logs_processed': 0,
            'source_results': {}
        }
        
        for future in future_to_source:
            source = future_to_source[future]
            source_name = source['name']
            
            try:
                logs_df = future.result()
                log_count = len(logs_df)
                
                results['sources_processed'] += 1
                results['total_logs_processed'] += log_count
                results['source_results'][source_name] = {
                    'status': 'success',
                    'logs_processed': log_count
                }
                
                logger.info(f"Successfully processed {log_count} logs from {source_name}")
                
            except Exception as e:
                results['sources_failed'] += 1
                results['source_results'][source_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                
                logger.error(f"Error processing source {source_name}: {str(e)}")
        
        logger.info(f"Parallel processing complete. Processed {results['sources_processed']} sources with {results['total_logs_processed']} logs")
        return results
    
    def batch_process(self, batch_size: int = 100) -> pd.DataFrame:
        """
        Process logs in batches to handle high volumes efficiently.
        
        Args:
            batch_size: Number of logs to process in each batch
            
        Returns:
            DataFrame containing all processed logs
        """
        logger.info(f"Starting batch processing with batch size {batch_size}")
        
        # Get all configured sources
        all_sources = self.config.log_sources if hasattr(self.config, 'log_sources') else []
        all_logs = []
        
        for source in all_sources:
            try:
                source_type = source.source_type
                source_name = source.name
                
                logger.debug(f"Batch processing source: {source_name}")
                
                # Choose the appropriate ingestion method based on source type
                if source_type == 'aws_s3':
                    logs_df = self._batch_ingest_from_aws_s3(source, batch_size)
                elif source_type == 'azure_blob':
                    logs_df = self._batch_ingest_from_azure_blob(source, batch_size)
                elif source_type == 'on_prem':
                    logs_df = self._batch_ingest_from_on_prem(source, batch_size)
                else:
                    logger.error(f"Unknown source type: {source_type}")
                    continue
                
                # Append to all logs
                if not logs_df.empty:
                    all_logs.append(logs_df)
                    logger.info(f"Processed {len(logs_df)} logs from {source_name}")
                
            except Exception as e:
                logger.error(f"Error batch processing source {source.name}: {str(e)}")
        
        # Combine all logs
        if all_logs:
            combined_logs = pd.concat(all_logs, ignore_index=True)
            logger.info(f"Batch processing complete. Total logs: {len(combined_logs)}")
            return combined_logs
        else:
            logger.info("Batch processing complete. No logs found.")
            return pd.DataFrame()
    
    def _batch_ingest_from_aws_s3(self, config: SourceConfig, batch_size: int) -> pd.DataFrame:
        """
        Batch ingest logs from AWS S3 with incremental processing.
        
        Args:
            config: Source configuration
            batch_size: Number of logs to process in each batch
            
        Returns:
            DataFrame containing all processed logs
        """
        logger.info(f"Batch ingesting from AWS S3: {config.name}")
        
        # Get state for this source
        state = self.processing_states.get(config.name) or ProcessingState(config.name)
        
        # Set cutoff time for incremental processing
        if state.last_processed_timestamp:
            cutoff_time = state.last_processed_timestamp
        else:
            # Default to config.max_age_days ago if no previous state
            cutoff_time = datetime.now() - timedelta(days=config.max_age_days)
        
        logger.debug(f"Using cutoff time: {cutoff_time} for incremental processing")
        
        # Placeholder implementation - in a real implementation would:
        # 1. List S3 objects with pagination
        # 2. Filter by last modified date > cutoff_time
        # 3. For each matching object:
        #    a. Check if already processed based on state.processed_file_hashes
        #    b. Download and process in batches
        #    c. Update state with processed file info
        
        # Mock implementation for demonstration
        logs = []
        for i in range(3):  # Simulate processing 3 files
            file_name = f"logs-{i}.json"
            file_hash = f"hash-{file_name}"
            
            # Skip if we've already processed this file with the same hash
            if file_name in state.processed_file_hashes and state.processed_file_hashes[file_name] == file_hash:
                logger.debug(f"Skipping already processed file: {file_name}")
                continue
                
            # Generate mock logs for this file
            file_logs = self._generate_mock_logs(
                config.name, 'aws_s3', 
                batch_size//3  # Distribute batch size across files
            )
            
            # Track that we've processed this file
            state.processed_file_hashes[file_name] = file_hash
            logs.append(file_logs)
        
        # Update state
        if logs:
            combined_logs = pd.concat(logs, ignore_index=True)
            if not combined_logs.empty and 'timestamp' in combined_logs.columns:
                state.last_processed_timestamp = combined_logs['timestamp'].max()
            
            state.last_run_time = datetime.now()
            self.processing_states[config.name] = state
            self._save_state()
            
            return combined_logs
        else:
            return pd.DataFrame()
    
    def _batch_ingest_from_azure_blob(self, config: SourceConfig, batch_size: int) -> pd.DataFrame:
        """
        Batch ingest logs from Azure Blob with incremental processing.
        
        Args:
            config: Source configuration
            batch_size: Number of logs to process in each batch
            
        Returns:
            DataFrame containing all processed logs
        """
        # Implementation would be similar to AWS S3 but using Azure SDK
        # Using mock data for now
        return self._generate_mock_logs(config.name, 'azure_blob', batch_size)
    
    def _batch_ingest_from_on_prem(self, config: SourceConfig, batch_size: int) -> pd.DataFrame:
        """
        Batch ingest logs from on-premises file system with incremental processing.
        
        Args:
            config: Source configuration
            batch_size: Number of logs to process in each batch
            
        Returns:
            DataFrame containing all processed logs
        """
        # Implementation would use file scanning with glob and file modification dates
        # Using mock data for now
        return self._generate_mock_logs(config.name, 'on_prem', batch_size)
