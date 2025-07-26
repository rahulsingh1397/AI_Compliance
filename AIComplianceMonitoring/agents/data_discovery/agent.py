"""
Enhanced Data Discovery Agent for AI-Enhanced Data Privacy and Compliance Monitoring.

Key Improvements:
1. Dependency injection for components
2. Configuration management with dataclass
3. Resource management via context manager
4. Enhanced error handling and validation
5. Parallel processing improvements
6. Type hints and documentation
7. Database connection pooling
"""

import os
import json
import logging
import time
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlalchemy as sa
from sqlalchemy.pool import QueuePool
from dataclasses import dataclass
from pydantic import validate_arguments, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
import humanize

# Configure structured logging
logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for Data Discovery Agent"""
    db_uri: Optional[str] = None
    # NLP Model configs
    bert_classifier_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    bert_ner_model_name: str = "dslim/bert-base-NER"
    spacy_model_name: str = "en_core_web_sm"
    # ML Classifier config
    ml_classifier_type: str = "lightgbm" # Defaulting to the enhanced classifier
    # General configs
    max_workers: int = 10
    db_pool_size: int = 5
    db_max_overflow: int = 10
    sample_size: int = 1000
    default_file_extensions: Tuple[str, ...] = ('.txt', '.csv', '.json', '.xml', '.html', '.md')


class DataDiscoveryAgent:
    """
    Enhanced Data Discovery Agent for sensitive data identification.
    
    Features:
    - Context manager for resource management
    - Dependency injection for components
    - Configurable via AgentConfig
    - Robust error handling and retry mechanisms
    """
    
    @validate_arguments
    def __init__(self, 
                 config: Optional[AgentConfig] = None,
                 nlp_model: Optional[Any] = None,
                 ml_classifier: Optional[Any] = None,
                 metadata_handler: Optional[Any] = None):
        """
        Initialize the Data Discovery Agent with dependency injection.
        
        Args:
            config: Agent configuration
            nlp_model: Pre-initialized NLP model
            ml_classifier: Pre-initialized ML classifier
            metadata_handler: Pre-initialized metadata handler
        """
        logger.debug("DataDiscoveryAgent.__init__ called.")

        logger.debug("Attempting to import NLPModel...")
        from .nlp_model import NLPModel
        logger.debug("NLPModel imported successfully.")

        logger.debug("Attempting to import MLClassifier...")
        from .ml_classifier import MLClassifier
        logger.debug("MLClassifier imported successfully.")

        logger.debug("Attempting to import MetadataHandler...")
        from .metadata_handler import MetadataHandler
        logger.debug("MetadataHandler imported successfully.")
        
        logger.debug("Initializing agent configuration...")
        self.config = config or AgentConfig()
        logger.info("Initializing Data Discovery Agent with config: %s", self.config)
        
        # Initialize components with dependency injection
        try:
            self.nlp_model = nlp_model or NLPModel(
                bert_classifier_model_name=self.config.bert_classifier_model_name,
                bert_ner_model_name=self.config.bert_ner_model_name,
                spacy_model_name=self.config.spacy_model_name
            )
            self.ml_classifier = ml_classifier or MLClassifier(
                classifier_type=self.config.ml_classifier_type
            )
            self.metadata_handler = metadata_handler or MetadataHandler(
                db_uri=self.config.db_uri
            )
            
            # Database connection pool
            self.engine = None
            if self.config.db_uri:
                self.engine = sa.create_engine(
                    self.config.db_uri,
                    poolclass=QueuePool,
                    pool_size=self.config.db_pool_size,
                    max_overflow=self.config.db_max_overflow,
                    pool_recycle=3600
                )
                
            logger.info("Agent components initialized successfully")
            
        except Exception as e:
            logger.exception("Unexpected initialization error")
            raise RuntimeError("Agent initialization failed") from e
            
        logger.info("Data Discovery Agent initialized successfully")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(sa.exc.OperationalError)
    )
    def scan_database(self):
        """
        Scan all tables in the configured database for sensitive data.
        """
        if not self.engine:
            logger.warning("Database URI not configured. Skipping database scan.")
            return

        logger.info("Starting database scan...")
        
        try:
            inspector = sa.inspect(self.engine)
            with self.engine.connect() as connection:
                schemas = inspector.get_schema_names()
                for schema in schemas:
                    # Skipping system schemas
                    if schema in ('information_schema', 'pg_catalog', 'sys'):
                        continue
                    
                    logger.info(f"Scanning schema: {schema}")
                    tables = inspector.get_table_names(schema=schema)
                    
                    for table_name in tables:
                        try:
                            logger.info(f"Scanning table: {schema}.{table_name}")
                            
                            query = f'SELECT * FROM "{schema}"."{table_name}"'
                            # Add LIMIT clause for databases that support it
                            if self.engine.dialect.name in ('postgresql', 'mysql', 'sqlite'):
                                query += f" LIMIT {self.config.sample_size}"

                            df = pd.read_sql(query, connection)
                            
                            if df.empty:
                                logger.info(f"Table {schema}.{table_name} is empty. Skipping.")
                                continue

                            predictions = self.ml_classifier.predict(df)
                            
                            sensitive_columns = [
                                col for col, result in predictions.items() if result.get('prediction') == 1
                            ]
                            
                            logger.info(f"Found {len(sensitive_columns)} sensitive columns in {schema}.{table_name}: {sensitive_columns}")

                            self.metadata_handler.save_metadata({
                                "source_type": "database",
                                "source_name": f"{self.config.db_uri.split('/')[-1]}.{schema}.{table_name}",
                                "scan_date": time.time(),
                                "classification": "sensitive" if sensitive_columns else "non-sensitive",
                                "details": {
                                    "sensitive_columns": predictions
                                }
                            })

                        except Exception as e:
                            logger.error(f"Failed to scan table {schema}.{table_name}: {e}", exc_info=True)
                            
        except sa.exc.OperationalError as e:
            logger.error(f"Database connection failed: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during database scan: {e}", exc_info=True)

        logger.info("Database scan completed.")

    def __enter__(self):
        """Context manager entry"""
        self.connect_resources()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit - resource cleanup"""
        self.cleanup_resources()
    
    def connect_resources(self):
        """Connect to external resources"""
        if self.engine:
            self.engine.connect()
        if hasattr(self.metadata_handler, 'connect'):
            self.metadata_handler.connect()
    
    def cleanup_resources(self):
        """Clean up resources"""
        if self.engine:
            self.engine.dispose()
        if hasattr(self.metadata_handler, 'close'):
            self.metadata_handler.close()


    @retry(stop=stop_after_attempt(3),
          wait=wait_exponential(multiplier=1, min=4, max=10),
          retry=retry_if_exception_type((sa.exc.OperationalError, OSError, pd.errors.EmptyDataError)))
    def _chunk_text(self, text: str, chunk_size: int = 512, stride: int = 256) -> List[str]:
        """
        Split text into overlapping chunks of specified size.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum number of tokens per chunk
            stride: Number of tokens to overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Simple whitespace tokenization - in practice, use the same tokenizer as your model
        tokens = text.split()
        chunks = []
        
        for i in range(0, len(tokens), stride):
            chunk = ' '.join(tokens[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks

    def _process_excel_file(self, file_path: str, chunk_size: int = 512, stride: int = 256) -> Dict[str, Any]:
        """
        Process Excel file by analyzing each column separately.
        
        Args:
            file_path: Path to the Excel file
            chunk_size: Maximum number of tokens per chunk for BERT
            stride: Number of tokens to overlap between chunks
            
        Returns:
            Dictionary with scan results
        """
        results = {
            'content': [],
            'sensitive_columns': [],
            'column_analysis': {}
        }
        
        try:
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl')
                sheet_results = {'sheet_name': sheet_name, 'columns': {}}
                
                for column in df.columns:
                    # Convert column to string and drop NA values
                    col_data = df[column].astype(str).dropna()
                    col_text = ' '.join(col_data)
                    
                    # Skip empty columns
                    if not col_text.strip():
                        continue
                        
                    # Analyze column content with chunking parameters
                    analysis = self.nlp_model.analyze_document(
                        col_text,
                        chunk_size=chunk_size,
                        stride=stride
                    )
                    sheet_results['columns'][column] = {
                        'contains_sensitive_data': analysis.get('contains_sensitive_data', False),
                        'sensitive_data_types': analysis.get('sensitive_data_types', []),
                        'sample_data': col_data.head(3).tolist()  # Store sample data for verification
                    }
                    
                    if analysis.get('contains_sensitive_data', False):
                        results['sensitive_columns'].append(f"{sheet_name}.{column}")
                
                results['column_analysis'][sheet_name] = sheet_results
                
                # Store sheet content as string for full document analysis
                results['content'].append(f"Sheet: {sheet_name}\n{df.head(100).to_string()}")
                
        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {str(e)}")
            raise
            
        return results

    def scan_unstructured_data(self, 
                             file_path: str,
                             document_identifier: Optional[str] = None,
                             chunk_size: int = 512,
                             stride: int = 256) -> Dict[str, Any]:
        """
        Scan unstructured data file for sensitive information with enhanced processing.
        
        Args:
            file_path: Path to the file to scan (supports .txt, .csv, .json, .xlsx, .xls)
            document_identifier: Optional document identifier
            chunk_size: Maximum number of tokens per chunk for BERT
            stride: Number of tokens to overlap between chunks
            
        Returns:
            Dictionary with scan results including column-level analysis for structured data
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        
        logger.info(f"Scanning file: {file_name} ({file_size/1024:.2f} KB)")
        
        try:
            analysis_results = []
            sensitive_data_found = False
            sensitive_data_types = set()
            
            # Handle Excel files with column-wise scanning
            if file_ext in ('.xlsx', '.xls'):
                excel_results = self._process_excel_file(
                    file_path=file_path,
                    chunk_size=chunk_size,
                    stride=stride
                )
                
                # Analyze aggregated content
                full_content = '\n\n'.join(excel_results['content'])
                full_analysis_result = self.nlp_model.analyze_document(
                    full_content,
                    chunk_size=chunk_size,
                    stride=stride
                )
                analysis_results.append(full_analysis_result) # Store the full analysis

                # Extract results from the full analysis
                sensitive_data_found = full_analysis_result['classification']['contains_sensitive_data']
                pii_entities = full_analysis_result.get('pii_entities', [])
                sensitive_data_types = {entity['entity_type'] for entity in pii_entities}
                
                # Store metadata with column-level analysis
                doc_id = document_identifier or file_name
                record_id = self.metadata_handler.store_unstructured_data_metadata(
                    source_location=file_path,
                    document_identifier=doc_id,
                    analysis_result=full_analysis_result
                )
                
                return {
                    'document_id': doc_id,
                    'file_path': file_path,
                    'file_name': file_name,
                    'file_size': file_size,
                    'content_type': file_ext.lstrip('.'),
                    'record_id': record_id,
                    'contains_sensitive_data': sensitive_data_found,
                    'sensitive_data_types': list(sensitive_data_types),
                    'sensitive_columns': excel_results['sensitive_columns'],
                    'column_analysis': excel_results['column_analysis'],
                    'status': 'success',
                    'processing_notes': 'Processed with column-wise analysis and full document analysis'
                }
                
            # Handle CSV files with chunking
            elif file_ext == '.csv':
                import pandas as pd
                import chardet
                
                logger.info(f"Processing CSV file {file_name} with chunking.")
                overall_sensitive_found = False
                overall_sensitive_types = set()
                column_analysis_details = {}
                total_nlp_chunks_processed = 0
                pandas_chunks_count = 0

                # Detect encoding from a sample first
                with open(file_path, 'rb') as f_sample:
                    sample_data = f_sample.read(10000) # Read 10KB for encoding detection
                    detected_encoding = chardet.detect(sample_data)['encoding'] or 'utf-8'
                
                try:
                    for i, chunk_df in enumerate(pd.read_csv(file_path, chunksize=10000, encoding=detected_encoding, low_memory=False)):
                        pandas_chunks_count += 1
                        logger.debug(f"Processing pandas chunk {i+1} for {file_name}")
                        for column_name in chunk_df.columns:
                            # Initialize column analysis if not present for this chunk
                            if column_name not in column_analysis_details:
                                column_analysis_details[column_name] = {
                                    'contains_sensitive_data': False, 
                                    'sensitive_data_types': set(), 
                                    'pii_entities_found': 0
                                }

                            # Ensure column data is not all NaN, convert to string
                            column_data_series = chunk_df[column_name].dropna()
                            if column_data_series.empty:
                                continue
                            
                            # To avoid token limit errors for very long columns, we can truncate the text.
                            # A more robust solution might involve chunking the column text itself.
                            # For now, we'll limit the number of rows processed per column chunk.
                            MAX_ROWS_PER_COLUMN_CHUNK = 1000
                            if len(column_data_series) > MAX_ROWS_PER_COLUMN_CHUNK:
                                logger.warning(f"Column '{column_name}' in pandas chunk {i+1} has {len(column_data_series)} rows. "
                                               f"Truncating to {MAX_ROWS_PER_COLUMN_CHUNK} rows for analysis to avoid memory issues.")
                                column_data_series = column_data_series.head(MAX_ROWS_PER_COLUMN_CHUNK)

                            full_column_text = "\n".join(column_data_series.astype(str))

                            if not full_column_text.strip():
                                continue
                            
                            # Analyze the full column text
                            analysis_result = self.nlp_model.analyze_document(
                                full_column_text,
                                chunk_size=chunk_size,
                                stride=stride
                            )
                            
                            total_nlp_chunks_processed += analysis_result.get('chunk_count', 1)

                            col_sensitive_found = analysis_result['classification']['contains_sensitive_data']
                            col_pii_entities = analysis_result.get('pii_entities', [])
                            
                            if col_sensitive_found:
                                overall_sensitive_found = True
                                column_analysis_details[column_name]['contains_sensitive_data'] = True
                            
                            if col_pii_entities:
                                pii_types = {entity['entity_type'] for entity in col_pii_entities}
                                overall_sensitive_types.update(pii_types)
                                column_analysis_details[column_name]['sensitive_data_types'].update(pii_types)
                                column_analysis_details[column_name]['pii_entities_found'] += len(col_pii_entities)

                except pd.errors.EmptyDataError:
                    logger.warning(f"CSV file {file_name} is empty or contains no data.")
                    # Proceed to store metadata for an empty file if necessary, or return specific status
                except Exception as e_csv:
                    logger.error(f"Error processing CSV file {file_name} with pandas: {str(e_csv)}", exc_info=True)
                    # Fallback or re-raise, for now, let's store error in metadata
                    doc_id = document_identifier or file_name
                    self.metadata_handler.store_unstructured_data_metadata(
                        source_location=file_path,
                        document_identifier=doc_id,
                        analysis_result={'error': f'CSV processing error: {str(e_csv)}', 'status': 'failed'}
                    )
                    raise # Re-raise to be caught by the outer try-except

                # Convert sets to lists for JSON serialization in metadata
                for col_name in column_analysis_details:
                    column_analysis_details[col_name]['sensitive_data_types'] = list(column_analysis_details[col_name]['sensitive_data_types'])

                doc_id = document_identifier or file_name
                record_id = self.metadata_handler.store_unstructured_data_metadata(
                    source_location=file_path,
                    document_identifier=doc_id,
                    analysis_result={
                        'classification': {'contains_sensitive_data': overall_sensitive_found},
                        'sensitive_data_types': list(overall_sensitive_types),
                        'pandas_chunk_count': pandas_chunks_count,
                        'total_nlp_chunk_count': total_nlp_chunks_processed,
                        'column_analysis': column_analysis_details
                    }
                )
                
                return {
                    'document_id': doc_id,
                    'file_path': file_path,
                    'file_name': file_name,
                    'file_size': file_size,
                    'content_type': file_ext.lstrip('.'),
                    'record_id': record_id,
                    'contains_sensitive_data': overall_sensitive_found,
                    'sensitive_data_types': list(overall_sensitive_types),
                    'pandas_chunk_count': pandas_chunks_count,
                    'total_nlp_chunk_count': total_nlp_chunks_processed,
                    'column_analysis': column_analysis_details,
                    'status': 'success',
                    'processing_notes': f'Processed CSV with {pandas_chunks_count} pandas chunks.'
                }

            # Handle other text-based files (txt, json, log)
            elif file_ext in ('.txt', '.json', '.log'):
                import chardet
                # Read only first 10KB for encoding detection to avoid MemoryError with large files
                with open(file_path, 'rb') as f:
                    sample = f.read(10000)
                    encoding = chardet.detect(sample)['encoding'] or 'utf-8'
                
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
            
                analysis_result = self.nlp_model.analyze_document(
                    content,
                    chunk_size=chunk_size,
                    stride=stride
                )
                analysis_results.append(analysis_result) # Store the full analysis

                # Extract results from the full analysis
                sensitive_data_found = analysis_result['classification']['contains_sensitive_data']
                pii_entities = analysis_result.get('pii_entities', [])
                sensitive_data_types = {entity['entity_type'] for entity in pii_entities}
                
                # Store metadata
                doc_id = document_identifier or file_name
                record_id = self.metadata_handler.store_unstructured_data_metadata(
                    source_location=file_path,
                    document_identifier=doc_id,
                    analysis_result=analysis_result
                )
                
                return {
                    'document_id': doc_id,
                    'file_path': file_path,
                    'file_name': file_name,
                    'file_size': file_size,
                    'content_type': file_ext.lstrip('.'),
                    'record_id': record_id,
                    'contains_sensitive_data': sensitive_data_found,
                    'sensitive_data_types': list(sensitive_data_types),
                    'chunk_count': analysis_result.get('chunk_count', 1),
                    'status': 'success',
                    'processing_notes': 'Processed with chunked BERT processing'
                }
                
            else:
                return {
                    'document_id': document_identifier or file_name,
                    'file_path': file_path,
                    'file_name': file_name,
                    'error': f'Unsupported file type: {file_ext}',
                    'status': 'failed'
                }
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {str(e)}", exc_info=True)
            return {
                'document_id': document_identifier or file_name,
                'file_path': file_path,
                'file_name': file_name,
                'error': str(e),
                'status': 'failed',
                'processing_notes': f'Error during processing: {str(e)}'
            }
    
    @retry(stop=stop_after_attempt(3),
          wait=wait_exponential(multiplier=1, min=4, max=10))
    def scan_structured_data(self, 
                            connection_string: str,
                            table_name: str,
                            sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Scan structured data (database table) for sensitive information.
        
        Args:
            connection_string: Database connection string
            table_name: Name of the table to scan
            sample_size: Number of rows to sample
            
        Returns:
            Dictionary with scan results
        """
        logger.info("Scanning structured table: %s", table_name)
        sample_size = sample_size or self.config.sample_size
        
        try:
            # Create database connection with connection pool
            engine = sa.create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10
            )
            
            # Get column metadata
            inspector = sa.inspect(engine)
            columns = inspector.get_columns(table_name)
            column_names = [col['name'] for col in columns]
            
            # Sample data from the table
            query = sa.text(f"SELECT * FROM {table_name} TABLESAMPLE SYSTEM (1) LIMIT {sample_size}")
            df = pd.read_sql(query, engine)
            
            # Analyze each column in parallel
            results = {}
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self._analyze_column, df[col], col): col
                    for col in column_names if col in df.columns
                }
                
                for future in as_completed(futures):
                    col = futures[future]
                    try:
                        results[col] = future.result()
                    except Exception as e:
                        logger.error("Error analyzing column %s: %s", col, str(e))
                        results[col] = {
                            "error": str(e),
                            "column_name": col,
                            "status": "failed"
                        }
            
            # Process results
            return self._process_structured_results(
                connection_string, 
                table_name, 
                results
            )
            
        except Exception as e:
            logger.exception("Table scan failed for %s", table_name)
            return {
                "error": str(e),
                "table_name": table_name,
                "status": "failed"
            }
    
    def _analyze_column(self, data: pd.Series, column_name: str) -> Dict[str, Any]:
        """Analyze single column with error handling"""
        try:
            # Handle empty columns
            if data.isnull().all():
                return {
                    "column_name": column_name,
                    "classification": "non-sensitive",
                    "confidence": 1.0,
                    "contains_sensitive_data": False,
                    "reason": "All null values"
                }
                
            # Analyze column
            return self.ml_classifier.analyze_column(data, column_name)
        except Exception as e:
            logger.error("Column analysis error: %s", column_name)
            return {
                "column_name": column_name,
                "error": str(e),
                "status": "failed"
            }
    
    def _process_structured_results(self, 
                                   connection_string: str,
                                   table_name: str,
                                   results: Dict[str, Any]) -> Dict[str, Any]:
        """Process and store structured data results"""
        sensitive_columns = []
        for col, result in results.items():
            if result.get("contains_sensitive_data", False):
                sensitive_columns.append(col)
                # Store metadata if sensitive
                record_id = self.metadata_handler.store_structured_data_metadata(
                    source_location=connection_string,
                    table_name=table_name,
                    column_name=col,
                    classification_result=result
                )
                result["record_id"] = record_id
        
        return {
            "table_name": table_name,
            "total_columns": len(results),
            "sensitive_columns": len(sensitive_columns),
            "sensitive_column_names": sensitive_columns,
            "column_results": results,
            "status": "success"
        }


    def batch_scan_files(self, 
                      directory_path: str,
                      file_extensions: Optional[List[str]] = None,
                      max_workers: Optional[int] = None,
                      file_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Scan multiple files in a directory in parallel with progress tracking.
        
        Args:
            directory_path: Base directory path (for relative paths in file_paths)
            file_extensions: List of file extensions to include (None for all)
            max_workers: Maximum number of worker threads
            file_paths: Specific list of file paths to process (overrides directory scan)
            
        Returns:
            Dictionary with scan results summary including statistics and sensitive files
        """
        # Use config values if not overridden
        workers = max_workers or self.config.max_workers
        extensions = tuple(file_extensions or self.config.default_file_extensions)
        
        # If specific file paths are provided, use those
        if file_paths is not None:
            file_paths = [os.path.abspath(f) for f in file_paths]
            logger.info(f"Processing {len(file_paths)} specified files")
        else:
            # Otherwise, scan the directory
            if not os.path.isdir(directory_path):
                raise ValueError(f"Directory not found: {directory_path}")
                
            # Collect all matching files with progress
            logger.info(f"Scanning directory: {directory_path} for file types: {extensions}")
            file_paths = []
            file_sizes = []
            
            # First pass: collect all files and their sizes for better progress estimation
            with tqdm(unit='files', desc='Discovering files') as pbar:
                for root, _, files in os.walk(directory_path):
                    for file in files:
                        if file.lower().endswith(extensions):
                            full_path = os.path.join(root, file)
                            file_paths.append(full_path)
                            try:
                                file_sizes.append(os.path.getsize(full_path))
                            except OSError:
                                file_sizes.append(0)
                            pbar.update(1)
            
            if not file_paths:
                logger.warning(f"No files found with extensions {extensions} in {directory_path}")
                return {
                    'directory_path': directory_path,
                    'total_files_processed': 0,
                    'total_size_processed': 0,
                    'successful_scans': 0,
                    'failed_scans': 0,
                    'sensitive_files_found': 0,
                    'file_results': {},
                    'sensitive_files': [],
                    'processing_time_seconds': 0,
                    'processing_rate': 0
                }
            
            total_size = sum(file_sizes)
            logger.info(f"Found {len(file_paths):,} files ({humanize.naturalsize(total_size)}) to scan in {directory_path}")
        
        # Process files in parallel with progress tracking
        results = {}
        sensitive_files = []
        processed_count = 0
        processed_size = 0
        total_files = len(file_paths)
        start_time = time.time()
        
        # Create a lock for thread-safe progress updates
        from threading import Lock
        progress_lock = Lock()
        
        # Initialize progress bar
        progress_bar = tqdm(
            total=total_files,
            desc='Scanning files',
            unit='file',
            dynamic_ncols=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
        
        def update_progress(file_path: str, result: Dict[str, Any]):
            """Thread-safe progress update"""
            nonlocal processed_count, processed_size
            with progress_lock:
                processed_count += 1
                if 'file_size' in result:
                    processed_size += result['file_size']
                
                # Update progress bar
                elapsed = time.time() - start_time
                files_per_sec = processed_count / elapsed if elapsed > 0 else 0
                mb_per_sec = (processed_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                
                progress_bar.set_postfix({
                    'rate': f"{files_per_sec:.1f} files/s, {mb_per_sec:.1f} MB/s",
                    'sensitive': len(sensitive_files)
                })
                progress_bar.update(1)
                
                # Log every 10 files or if sensitive data is found
                if processed_count % 10 == 0 or processed_count == total_files or result.get('contains_sensitive_data', False):
                    status = "sensitive" if result.get('contains_sensitive_data', False) else "clean"
                    logger.debug(f"Processed {processed_count}/{total_files} files - {os.path.basename(file_path)}: {status}")
        
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks
                future_to_path = {
                    executor.submit(self.scan_unstructured_data, file_path): file_path
                    for file_path in file_paths
                }
                
                # Process results as they complete
                for future in as_completed(future_to_path):
                    file_path = future_to_path[future]
                    relative_path = os.path.relpath(file_path, directory_path)
                    
                    try:
                        result = future.result()
                        results[relative_path] = result
                        
                        # Track sensitive files
                        # A file is sensitive if it contains any PII, a more reliable metric.
                        if result.get('pii_count', 0) > 0:
                            sensitive_files.append({
                                'path': relative_path,
                                'sensitive_types': result.get('sensitive_data_types', []),
                                'file_size': result.get('file_size', 0)
                            })
                            logger.warning(f"Sensitive data found in: {relative_path} - Types: {result.get('sensitive_data_types', [])}")
                        
                        # Update progress
                        update_progress(relative_path, result)
                        
                    except Exception as e:
                        logger.error(f"Error processing {relative_path}: {str(e)}", exc_info=True)
                        results[relative_path] = {
                            'error': str(e),
                            'status': 'failed',
                            'file_path': relative_path
                        }
                        update_progress(relative_path, {'file_size': 0})
        finally:
            progress_bar.close()
            
        # Calculate final statistics
        elapsed = time.time() - start_time
        files_per_sec = processed_count / elapsed if elapsed > 0 else 0
        mb_per_sec = (processed_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        
        # Generate summary
        summary = self._generate_scan_summary(directory_path, results, sensitive_files)
        
        # Add timing and rate information
        summary.update({
            'total_files_processed': processed_count,
            'total_size_processed': processed_size,
            'sensitive_files_found': len(sensitive_files),
            'processing_time_seconds': round(elapsed, 2),
            'processing_rate_files_sec': round(files_per_sec, 2),
            'processing_rate_mb_sec': round(mb_per_sec, 2),
            'sensitive_files': sensitive_files,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
            'end_time': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Log completion
        logger.info(f"Scan completed: {processed_count} files processed in {elapsed:.1f} seconds "
                  f"({files_per_sec:.1f} files/s, {mb_per_sec:.1f} MB/s)")
        if sensitive_files:
            logger.warning(f"Found {len(sensitive_files)} files with sensitive data")
        
        return summary
    
    def _collect_files(self, directory_path: str, extensions: Tuple[str]) -> List[str]:
        """Collect files with given extensions"""
        if not os.path.isdir(directory_path):
            raise ValueError(f"Invalid directory: {directory_path}")
            
        files = []
        for root, _, filenames in os.walk(directory_path):
            for filename in filenames:
                if filename.lower().endswith(extensions):
                    files.append(os.path.join(root, filename))
        logger.info("Found %d files to scan", len(files))
        return files
    
    def _generate_scan_summary(self, 
                              directory_path: str, 
                              results: Dict[str, Any],
                              sensitive_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary report from scan results"""
        # Extract just the file paths from the sensitive_files list for backward compatibility
        sensitive_file_paths = [sf['path'] for sf in sensitive_files]
        
        success_count = sum(1 for r in results.values() if r.get("status") == "success")
        failure_count = len(results) - success_count
        
        return {
            "directory_path": directory_path,
            "total_files_scanned": len(results),
            "successful_scans": success_count,
            "failed_scans": failure_count,
            "sensitive_files_found": len(sensitive_files),
            "sensitive_file_paths": sensitive_file_paths,
            "scan_details": results
        }
    
    def get_scan_statistics(self) -> Dict[str, Any]:
        """Get statistics with error handling"""
        try:
            structured = self.metadata_handler.search_records(source_type="structured")
            unstructured = self.metadata_handler.search_records(source_type="unstructured")
            all_records = structured + unstructured
            
            return self._calculate_statistics(all_records)
        except Exception as e:
            logger.error("Error getting statistics: %s", e)
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def _calculate_statistics(self, records: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics from records"""
        if not records:
            return {"message": "No records found", "status": "success"}
        
        # Categorize records
        categories = {
            "PII": 0,
            "Financial": 0,
            "Health": 0,
            "Other": 0
        }
        
        for record in records:
            data_type = record.get("data_type", "Other")
            if data_type.startswith("PII"):
                categories["PII"] += 1
            elif data_type in categories:
                categories[data_type] += 1
            else:
                categories["Other"] += 1
        
        # Calculate confidence
        confidences = [r.get("confidence_score", 0) for r in records]
        avg_confidence = sum(confidences) / len(confidences)
        
        return {
            "total_records": len(records),
            "structured_records": len([r for r in records if r.get("source_type") == "structured"]),
            "unstructured_records": len([r for r in records if r.get("source_type") == "unstructured"]),
            "record_categories": categories,
            "average_confidence": round(avg_confidence, 4),
            "status": "success"
        }
    
    @retry(stop=stop_after_attempt(3))
    def train_ml_classifier(self, 
                          training_data: pd.DataFrame, 
                          target_column: str) -> Dict[str, float]:
        """Train ML classifier with validation"""
        if target_column not in training_data.columns:
            raise ValueError(f"Target column {target_column} not found in training data")
        
        logger.info("Training ML classifier with %d samples", len(training_data))
        X = training_data.drop(columns=[target_column])
        y = training_data[target_column]
        
        metrics = self.ml_classifier.train(X, y)
        logger.info("Training completed: Accuracy=%.4f", metrics.get('accuracy', 0))
        
        # Store training metadata
        self.metadata_handler.store_training_metadata(
            features=list(X.columns),
            target=target_column,
            metrics=metrics,
            classifier_type=self.config.ml_classifier_type
        )
        
        return metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Check health status of components"""
        status = {
            "nlp_model": self.nlp_model.is_ready() if hasattr(self.nlp_model, 'is_ready') else "unknown",
            "ml_classifier": self.ml_classifier.is_ready() if hasattr(self.ml_classifier, 'is_ready') else "unknown",
            "metadata_handler": self.metadata_handler.is_connected() if hasattr(self.metadata_handler, 'is_connected') else "unknown",
            "status": "operational"
        }
        
        # Check for any failed components
        if any(v == "unavailable" for v in status.values() if isinstance(v, str)):
            status["status"] = "degraded"
            
        return status


