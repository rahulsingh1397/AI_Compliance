import sys
import os
import logging
import json
import traceback
import argparse
import glob
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# --- Configuration & Setup ---
project_root = os.path.dirname(os.path.abspath(__file__))

# Define a basic configuration class similar to the original AgentConfig
class AgentConfig:
    def __init__(self, db_uri=None, bert_classifier_model_name='distilbert-base-uncased-finetuned-sst-2-english',
                 bert_ner_model_name='dslim/bert-base-NER', spacy_model_name='en_core_web_sm',
                 ml_classifier_type='lightgbm', max_workers=10, db_pool_size=5, db_max_overflow=10,
                 sample_size=1000, default_file_extensions=('.txt', '.csv', '.json', '.xml', '.html', '.md')):
        self.db_uri = db_uri
        self.bert_classifier_model_name = bert_classifier_model_name
        self.bert_ner_model_name = bert_ner_model_name
        self.spacy_model_name = spacy_model_name
        self.ml_classifier_type = ml_classifier_type
        self.max_workers = max_workers
        self.db_pool_size = db_pool_size
        self.db_max_overflow = db_max_overflow
        self.sample_size = sample_size
        self.default_file_extensions = default_file_extensions
    
    def __str__(self):
        return (f"AgentConfig(db_uri={self.db_uri}, bert_classifier_model_name='{self.bert_classifier_model_name}', "
                f"bert_ner_model_name='{self.bert_ner_model_name}', spacy_model_name='{self.spacy_model_name}', "
                f"ml_classifier_type='{self.ml_classifier_type}', max_workers={self.max_workers}, "
                f"db_pool_size={self.db_pool_size}, db_max_overflow={self.db_max_overflow}, "
                f"sample_size={self.sample_size}, default_file_extensions={self.default_file_extensions})")

# Simplified mock for metadata extraction 
class MetadataHandler:
    def __init__(self):
        logging.debug("Initialized MetadataHandler")
    
    def extract_metadata(self, file_path):
        try:
            logging.debug(f"Extracting metadata for: {file_path}")
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            extension = os.path.splitext(file_path)[1].lower()
            
            # Try to read a sample of the file to determine content type 
            mime_type = "application/octet-stream"
            with open(file_path, 'rb') as f:
                header = f.read(512)
                if header.startswith(b'%PDF-'):
                    mime_type = "application/pdf"
                elif header.startswith(b'PK\x03\x04'):
                    if extension == ".xlsx":
                        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    elif extension == ".docx":
                        mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    else:
                        mime_type = "application/zip"
                else:
                    # Check for text files
                    try:
                        header.decode('utf-8')
                        if extension == ".csv":
                            mime_type = "text/csv"
                        elif extension == ".json":
                            mime_type = "application/json"
                        elif extension in [".txt", ".md"]:
                            mime_type = "text/plain"
                        else:
                            mime_type = "text/plain"
                    except:
                        pass
            
            # Get file stats 
            stat_info = os.stat(file_path)
            created_date = datetime.fromtimestamp(stat_info.st_ctime).isoformat()
            modified_date = datetime.fromtimestamp(stat_info.st_mtime).isoformat()
            
            return {
                "file_name": file_name,
                "file_size": file_size,
                "created_date": created_date,
                "modified_date": modified_date,
                "file_extension": extension,
                "mime_type": mime_type
            }
        except Exception as e:
            logging.error(f"Error extracting metadata from {file_path}: {e}")
            return {
                "file_name": os.path.basename(file_path),
                "error": str(e)
            }

# Enhanced text classifier that uses regex patterns to detect sensitive data
import re

class SimpleClassifier:
    # Keywords for general sensitive content
    SENSITIVE_KEYWORDS = [
        'ssn', 'social security', 'password', 'credit card', 'bank account',
        'address', 'phone', 'email', 'birthdate', 'birth date', 'license',
        'passport', 'confidential', 'private', 'secret', 'classified', 
        'sensitive', 'personal', 'proprietary', 'restricted', 'hra', 'health record',
        'medical', 'insurance', 'patient', 'diagnosis', 'treatment'
    ]
    
    # Regex patterns for specific PII types
    PII_PATTERNS = {
        # Credit card numbers (major card types)
        'credit_card': r'\b(?:\d[ -]*?){13,16}\b',  # Basic pattern to catch most formats
        
        # Social Security Numbers (with or without dashes)
        'ssn': r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        
        # Email addresses
        'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        
        # Phone numbers (various formats)
        'phone': r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        
        # IP addresses
        'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        
        # Dates (various formats)
        'date': r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b',
        
        # ZIP/Postal codes
        'zipcode': r'\b\d{5}(?:-\d{4})?\b',
    }
    
    def __init__(self):
        logging.debug("Initialized Enhanced SimpleClassifier with regex patterns")
        # Compile patterns for efficiency
        self.compiled_patterns = {name: re.compile(pattern) for name, pattern in self.PII_PATTERNS.items()}
    
    def detect_pii(self, text):
        """Find PII entities using regex patterns"""
        if not text:
            return []
        
        pii_entities = []
        
        # Detect patterns
        for pii_type, pattern in self.compiled_patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                # Mask the match for display - show only partial content
                if pii_type == 'credit_card' and len(match) > 8:
                    masked = match[:4] + "****" + match[-4:]
                elif pii_type == 'ssn' and len(match) > 5:
                    masked = "***-**-" + match[-4:]
                elif pii_type == 'email':
                    parts = match.split('@')
                    if len(parts) > 1:
                        masked = parts[0][:3] + "***@" + parts[1]
                    else:
                        masked = match[:3] + "***"
                else:
                    # Generic masking for other types
                    if len(match) > 4:
                        masked = match[:2] + "***" + match[-2:]
                    else:
                        masked = match[:1] + "***"
                
                pii_entities.append({
                    "type": pii_type,
                    "masked_value": masked,
                    "confidence": 0.9  # High confidence for regex matches
                })
        
        return pii_entities
    
    def analyze_document(self, text):
        if not text:
            return {
                "classification": {
                    "contains_sensitive_data": False,
                    "confidence": 1.0,
                    "classification": "NON_SENSITIVE"
                },
                "pii_entities": [],
                "pii_count": 0,
                "sensitivity_score": 0.0
            }
        
        # Convert to lowercase for case-insensitive keyword matching
        text_lower = text.lower()
        
        # Count keyword matches
        keyword_matches = [kw for kw in self.SENSITIVE_KEYWORDS if kw in text_lower]
        
        # Find PII using regex patterns
        pii_entities = self.detect_pii(text)
        pii_count = len(pii_entities)
        
        # Calculate sensitivity score based on both keywords and PII entities
        # Weight PII entities higher than keywords
        keyword_score = len(keyword_matches) / max(1, min(1000, len(text)/100))
        pii_score = min(1.0, pii_count * 0.2)  # Each PII entity increases score significantly
        
        # Combined score with higher weight for PII
        sensitivity_score = min(1.0, (keyword_score * 0.3) + (pii_score * 0.7))
        
        # Classification logic
        is_sensitive = sensitivity_score > 0.1 or pii_count > 0
        classification = "SENSITIVE" if is_sensitive else "NON_SENSITIVE"
        
        # Confidence based on score and PII presence
        confidence = 0.6 + (sensitivity_score * 0.4)  # Range from 0.6 to 1.0
        if pii_count > 0:
            confidence = max(confidence, 0.9)  # High confidence if PII detected
        
        logging.debug(f"Classification: {classification}, Score: {sensitivity_score:.2f}, Confidence: {confidence:.2f}, PII count: {pii_count}")
        
        return {
            "classification": {
                "contains_sensitive_data": is_sensitive,
                "confidence": confidence,
                "classification": classification
            },
            "pii_entities": pii_entities,
            "keyword_matches": keyword_matches[:10],
            "pii_count": pii_count,
            "sensitivity_score": sensitivity_score
        }

# Simplified version of the DataDiscoveryAgent
class DataDiscoveryAgent:
    def __init__(self, config=None):
        logging.debug("Initializing simplified DataDiscoveryAgent")
        self.config = config or AgentConfig()
        self.metadata_handler = MetadataHandler()
        self.classifier = SimpleClassifier()
        logging.info(f"DataDiscoveryAgent initialized with config: {self.config}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def scan_file(self, file_path):
        try:
            logging.info(f"Scanning file: {file_path}")
            
            # Extract metadata
            metadata = self.metadata_handler.extract_metadata(file_path)
            extension = metadata.get("file_extension", "").lower()
            
            # Basic content extraction based on file type
            file_content = ""
            if extension in [".txt", ".csv", ".md", ".json", ".xml", ".html"]:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        # Read up to ~100KB of text to avoid memory issues with large files
                        file_content = f.read(102400)
                except Exception as e:
                    logging.error(f"Error reading text file {file_path}: {e}")
            else:
                # For non-text files, we just note the file type
                file_content = f"Binary file of type {metadata.get('mime_type', 'unknown')}"
            
            # Analyze content
            analysis_result = self.classifier.analyze_document(file_content)
            
            result = {
                "file_path": file_path,
                "metadata": metadata,
                "analysis": analysis_result,
                "scan_timestamp": datetime.now().isoformat(),
                "scan_status": "SUCCESS"
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error scanning file {file_path}: {e}", exc_info=True)
            return {
                "file_path": file_path,
                "scan_status": "ERROR",
                "error": str(e),
                "scan_timestamp": datetime.now().isoformat()
            }
    
    def batch_scan_files(self, directory_path, extensions=None):
        extensions = extensions or self.config.default_file_extensions
        start_time = datetime.now()
        
        logging.info(f"Starting batch scan of {directory_path} with extensions: {extensions}")
        
        # Find all files with the specified extensions
        all_files = []
        for ext in extensions:
            pattern = os.path.join(directory_path, f"**/*{ext}")
            all_files.extend(glob.glob(pattern, recursive=True))
        
        logging.info(f"Found {len(all_files)} files to scan")
        
        # Process each file
        results = []
        successful_scans = 0
        failed_scans = 0
        sensitive_files_found = 0
        
        for file_path in all_files:
            try:
                result = self.scan_file(file_path)
                results.append(result)
                
                if result.get("scan_status") == "SUCCESS":
                    successful_scans += 1
                    if result.get("analysis", {}).get("classification", {}).get("contains_sensitive_data", False):
                        sensitive_files_found += 1
                else:
                    failed_scans += 1
            except Exception as e:
                logging.error(f"Unexpected error scanning {file_path}: {e}", exc_info=True)
                failed_scans += 1
                results.append({
                    "file_path": file_path,
                    "scan_status": "ERROR",
                    "error": str(e),
                    "scan_timestamp": datetime.now().isoformat()
                })
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            "directory_scanned": directory_path,
            "scan_started": start_time.isoformat(),
            "scan_completed": end_time.isoformat(),
            "scan_duration_seconds": duration,
            "total_files_scanned": len(all_files),
            "successful_scans": successful_scans,
            "failed_scans": failed_scans,
            "sensitive_files_found": sensitive_files_found,
            "file_results": results
        }
        
        return summary

def setup_logging(log_file_path):
    """Configures logging to file and console."""
    if os.path.exists(log_file_path):
        try:
            os.remove(log_file_path) # Clear previous log file
        except OSError as e:
            print(f"Warning: Could not remove old log file '{log_file_path}': {e}")

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# --- Main execution ---
def run_data_scan(data_dir_path, results_file_path, current_log_file_path):
    """
    Initializes and runs the DataDiscoveryAgent to scan the specified data directory.
    """
    logger = logging.getLogger(__name__) # Get logger instance for this function
    logger.info("="*45)
    logger.info("      STARTING DATA DISCOVERY SCAN TEST      ")
    logger.info("="*45)

    if not os.path.isdir(data_dir_path):
        logger.error(f"FATAL: Data directory not found at: {data_dir_path}")
        return

    logger.info(f"Target directory for scanning: {data_dir_path}")
    logger.info(f"Results will be saved to: {results_file_path}")
    logger.info(f"Logging to: {current_log_file_path}")

    try:
        agent_config = AgentConfig()
        logger.info(f"Using AgentConfig: {agent_config}")

        logger.info("Initializing DataDiscoveryAgent...")
        with DataDiscoveryAgent(config=agent_config) as agent:
            logger.info("DataDiscoveryAgent initialized. Starting file scan...")
            scan_results = agent.batch_scan_files(directory_path=data_dir_path)
            logger.info("Scan completed successfully.")

            logger.info(f"Saving scan results to {results_file_path}...")
            with open(results_file_path, 'w', encoding='utf-8') as f:
                json.dump(scan_results, f, indent=4, ensure_ascii=False)
            logger.info("Results saved.")

            print("\n--- Scan Summary ---")
            summary = {
                "total_files_scanned": scan_results.get("total_files_scanned"),
                "successful_scans": scan_results.get("successful_scans"),
                "failed_scans": scan_results.get("failed_scans"),
                "sensitive_files_found": scan_results.get("sensitive_files_found"),
            }
            print(json.dumps(summary, indent=2))
            print(f"\nDetailed results are in {results_file_path}")
            print(f"Full log is in {current_log_file_path}")
            print("--- End of Summary ---\n")

    except Exception as e:
        logger.error("An unexpected error occurred during the data discovery scan.", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Data Discovery Agent Scan.")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=os.path.join(project_root, 'AIComplianceMonitoring', 'data'),
        help="Directory containing data to scan."
    )
    parser.add_argument(
        "--results_file", 
        type=str, 
        default=os.path.join(project_root, 'scan_results.json'),
        help="File to save scan results (JSON)."
    )
    parser.add_argument(
        "--log_file", 
        type=str, 
        default=os.path.join(project_root, 'test_scan.log'),
        help="File to write logs to."
    )
    args = parser.parse_args()

    # Setup logging with the potentially overridden log_file path
    logger = setup_logging(args.log_file)

    run_data_scan(args.data_dir, args.results_file, args.log_file)
    
    logger.info("="*45)
    logger.info("      DATA DISCOVERY SCAN TEST FINISHED      ")
    logger.info("="*45)
