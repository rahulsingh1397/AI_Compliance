"""
NLP Model for Data Discovery Agent.

This module implements NLP models (BERT, spaCy) for unstructured data classification
to identify sensitive information like PII in emails, documents, etc.

Features:
- Chunked text processing for long documents
- Enhanced PII detection with regex patterns and spaCy
- BERT-based classification with configurable chunk size and stride
"""

import os
import re
import spacy
import numpy as np
import logging
import warnings
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import torch
from pydantic import BaseModel, Field
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)
try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    warnings.warn("'datasets' package not found, falling back to standard processing. Install with: pip install datasets")

# Suppress HuggingFace warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings('ignore', message=".*Using a non-full backward hook*")
warnings.filterwarnings('ignore', message=".*You seem to be using the pipelines sequentially on GPU.*")
warnings.filterwarnings("ignore", message="`return_all_scores` is now deprecated")

# Configure logging
logger = logging.getLogger(__name__)

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 512  # BERT's maximum sequence length
DEFAULT_STRIDE = 256      # Overlap between chunks

class NLPModel:
    """
    NLP Model for sensitive data identification and classification.
    Uses BERT for document-level classification and spaCy for entity recognition.
    """
    
    def __init__(self, 
                 bert_classifier_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
                 bert_ner_model_name: str = "dslim/bert-base-NER", 
                 spacy_model_name: str = "en_core_web_sm",
                 device: Optional[Union[int, str]] = None,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 stride: int = DEFAULT_STRIDE):
        """
        Initialize NLP models for data classification with enhanced PII detection.
        
        Args:
            bert_classifier_model_name: Name or path of the BERT model for text classification.
            bert_ner_model_name: Name or path of the BERT model for Named Entity Recognition (NER).
            spacy_model_name: Name of the spaCy model to use for basic NLP tasks.
            device: Device to run models on (None for auto, 'cuda' for GPU, 'cpu' for CPU, or device index e.g., 0).
            chunk_size: Maximum number of tokens per chunk for BERT processing.
            stride: Number of overlapping tokens between chunks.
        """
        self.chunk_size = chunk_size
        self.stride = stride
        self.gpu_batch_size = 4 if torch.cuda.is_available() else 2

        self._initialize_device(device)
        self._initialize_classifier(bert_classifier_model_name)
        self._initialize_ner_pipelines(bert_ner_model_name, spacy_model_name)

    def _initialize_device(self, device: Optional[Union[int, str]]):
        """Determines and sets the computational device (CPU/GPU)."""
        if device is None:
            self.device_id = 0 if torch.cuda.is_available() else -1
        elif isinstance(device, str) and device.lower() == 'cuda':
            self.device_id = 0 if torch.cuda.is_available() else -1
        elif isinstance(device, str) and device.lower() == 'cpu':
            self.device_id = -1
        elif isinstance(device, int):
            self.device_id = device
        else:
            logger.warning(f"Invalid device specified: {device}. Defaulting to auto-detection.")
            self.device_id = 0 if torch.cuda.is_available() else -1
        logger.info(f"Initializing NLPModel on device: {'cuda:' + str(self.device_id) if self.device_id != -1 else 'cpu'}")

    def _initialize_classifier(self, model_name: str):
        """Initializes the BERT model and pipeline for text classification."""
        logger.info(f"Loading BERT classification model: {model_name}")
        use_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
        model_kwargs = {"torch_dtype": torch.float16} if use_fp16 else {}

        try:
            self.classifier_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.classifier_model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
            self.classifier_pipeline = pipeline(
                "text-classification",
                model=self.classifier_model,
                tokenizer=self.classifier_tokenizer,
                device=self.device_id,
                batch_size=self.gpu_batch_size,
                return_all_scores=True
            )
            logger.info("BERT classification pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to load BERT classification model '{model_name}': {e}")
            raise

    def _initialize_ner_pipelines(self, bert_ner_model_name: str, spacy_model_name: str):
        """Initializes BERT and spaCy models and pipelines for Named Entity Recognition."""
        # Initialize BERT NER Pipeline
        logger.info(f"Loading BERT NER model: {bert_ner_model_name}")
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model=bert_ner_model_name,
                tokenizer=bert_ner_model_name,
                device=self.device_id,
                batch_size=self.gpu_batch_size,
                grouped_entities=True
            )
            logger.info("BERT NER pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to load BERT NER model '{bert_ner_model_name}': {e}", exc_info=True)
            self.ner_pipeline = None

        # Initialize spaCy
        logger.info(f"Loading spaCy model: {spacy_model_name}")
        try:
            self.nlp = spacy.load(spacy_model_name)
            if 'sentencizer' not in self.nlp.pipe_names:
                self.nlp.add_pipe('sentencizer', first=True)
            
            pipes_to_disable = [p for p in ["parser", "textcat"] if p in self.nlp.pipe_names]
            if pipes_to_disable:
                self.nlp.disable_pipes(*pipes_to_disable)

            self._add_custom_patterns_to_spacy()
            logger.info("spaCy pipeline initialized and configured successfully.")
        except (OSError, ValueError):
            logger.warning(f"spaCy model '{spacy_model_name}' not found. Attempting to download...")
            from spacy.cli import download as spacy_download
            try:
                spacy_download(spacy_model_name)
                self.nlp = spacy.load(spacy_model_name)
                logger.info(f"Successfully downloaded and loaded spaCy model: {spacy_model_name}")
                # Recurse to configure the newly downloaded model
                self._initialize_ner_pipelines(bert_ner_model_name, spacy_model_name)
            except Exception as e_dl:
                logger.error(f"Failed to download or initialize spaCy model after download attempt: {e_dl}", exc_info=True)
                self.nlp = None
        except Exception as e:
            logger.error(f"An unexpected error occurred during spaCy initialization: {e}", exc_info=True)
            self.nlp = None
            
        # If spaCy still couldn't be loaded, log a warning and proceed without it
        if not self.nlp:
            logger.warning("spaCy could not be loaded. Some features may be limited.")

        # Define PII entity types and their regex patterns (if applicable)
        # Confidence scores here are base scores, can be adjusted by source (BERT, spaCy, regex)
        self.pii_definitions = {
            "PERSON": {"label": "Person Name", "regex": None, "confidence": 0.85},
            "EMAIL": {"label": "Email Address", "regex": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "confidence": 0.98},
            "PHONE_NUMBER": {"label": "Phone Number", "regex": r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b", "confidence": 0.95},
            "US_SSN": {"label": "US Social Security Number", "regex": r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b", "confidence": 0.99},
            "CREDIT_CARD": {"label": "Credit Card Number", "regex": r"\b(?:\d[ -]*?){13,19}\b", "confidence": 0.97}, # General pattern, specific cards below
            "VISA": {"label": "Credit Card (Visa)", "regex": r"\b4[0-9]{12}(?:[0-9]{3})?\b", "confidence": 0.98},
            "MASTERCARD": {"label": "Credit Card (Mastercard)", "regex": r"\b(?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][0-9]|2720)[0-9]{12}\b", "confidence": 0.98},
            "AMEX": {"label": "Credit Card (American Express)", "regex": r"\b3[47][0-9]{13}\b", "confidence": 0.98},
            "DISCOVER": {"label": "Credit Card (Discover)", "regex": r"\b6(?:011|5[0-9]{2})[0-9]{12}\b", "confidence": 0.98},
            "ADDRESS": {"label": "Physical Address", "regex": r"\b\d+\s+([A-Za-z0-9\s,.-]+)(\b(?:Apt|Suite|Unit|#)[\s.]*\w+)?\s*,?\s*[A-Za-z\s]+(?:,\s*[A-Z]{2})?\s*\d{5}(?:-\d{4})?\b", "confidence": 0.75},
            "DATE_OF_BIRTH": {"label": "Date of Birth", "regex": r"\b(?:(?:0?[1-9]|1[0-2])[-/\s](?:0?[1-9]|[12][0-9]|3[01])[-/\s](?:19|20)?\d{2})|(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-.\s]+\d{1,2}(?:st|nd|rd|th)?[\s.,-]+(?:19|20)?\d{2})\b", "confidence": 0.80},
            "IP_ADDRESS": {"label": "IP Address", "regex": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b", "confidence": 0.99},
            "PASSPORT_NUMBER": {"label": "Passport Number", "regex": r"\b[A-PR-WYa-pr-wy][1-9]\d{7}[A-PR-WYa-pr-wy]?\b", "confidence": 0.90}, # Common patterns, vary by country
            "DRIVERS_LICENSE": {"label": "Driver's License", "regex": r"\b[A-Za-z][0-9]{6,12}\b", "confidence": 0.85}, # Highly variable
            "BANK_ACCOUNT_NUMBER": {"label": "Bank Account Number", "regex": r"\b\d{6,17}\b", "confidence": 0.70},
            # spaCy specific labels we might want to map or use
            "ORG": {"label": "Organization", "regex": None, "confidence": 0.70},
            "GPE": {"label": "Geopolitical Entity", "regex": None, "confidence": 0.70},
            "DATE": {"label": "Date", "regex": None, "confidence": 0.70} # General date, distinct from DOB
        }

        # Initialize EntityRuler and add it to the spaCy pipeline
        if not self.nlp.has_pipe("entity_ruler"):
            # Try to add before 'ner' if 'ner' exists, otherwise add at the end.
            # overwrite_ents = True allows custom entities to overwrite existing ones.
            if self.nlp.has_pipe("ner"):
                self.ruler = self.nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
            else:
                self.ruler = self.nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
            logger.info("Added EntityRuler to spaCy pipeline.")
        else:
            self.ruler = self.nlp.get_pipe("entity_ruler") # Get existing ruler
            logger.info("EntityRuler already exists in spaCy pipeline. Using existing.")

        self._add_custom_patterns_to_spacy()
        logger.info("NLPModel initialization complete.")
    
    def _add_custom_patterns_to_spacy(self):
        """
        Adds custom PII detection patterns (from self.pii_patterns) to the spaCy EntityRuler.
        """
        if not hasattr(self, 'ruler') or self.ruler is None:
            logger.error("EntityRuler (self.ruler) not initialized. Cannot add custom patterns.")
            return

        patterns = []
        # Corrected to use self.pii_definitions
        for pii_type_key, attributes in self.pii_definitions.items(): 
            if attributes.get("regex") and attributes.get("label"):
                # spaCy's EntityRuler pattern format for regex:
                # The 'label' for the EntityRuler should be the PII type key (e.g., "EMAIL", "US_SSN")
                # so we can map it back easily using pii_definitions.
                pattern = {
                    "label": pii_type_key,  # Use the PII type key as the spaCy label
                    "pattern": [{ "TEXT": { "REGEX": attributes["regex"] } }],
                    "id": pii_type_key  # Use the original key as an ID for the pattern
                }
                patterns.append(pattern)
            elif attributes.get("regex"):
                logger.warning(f"Skipping PII pattern for '{pii_type_key}' due to missing 'label'. Regex: {attributes['regex']}")
            # We don't warn if regex is missing because some PII types are only for spaCy/BERT NER (e.g., PERSON)
        
        if patterns:
            try:
                self.ruler.add_patterns(patterns)
                logger.info(f"Added {len(patterns)} custom regex PII patterns to spaCy EntityRuler.")
            except Exception as e:
                logger.error(f"Error adding patterns to EntityRuler: {e}", exc_info=True)
        else:
            logger.info("No valid regex-based PII patterns found in pii_definitions to add to spaCy EntityRuler.")
    
    def _chunk_text_by_sentences(self, text: str, chunk_size: int, stride: int) -> List[str]:
        """
        Splits text into chunks by sentences, ensuring no chunk exceeds the max token limit.
        Args:
            text: Text to chunk
            chunk_size: The maximum size of each chunk.
            stride: The overlap between chunks (used in fallback).
            
        Returns:
            List of text chunks respecting sentence boundaries when possible
        """
        if not self.nlp or not self.nlp.has_pipe('sentencizer'):
            logger.warning("spaCy sentencizer not available or not functional. Falling back to basic chunking with stride.")
            # Fallback chunking with stride
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - stride)]
        
        # Use spaCy to split text into sentences
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        # Use the standalone tokenizer instead of pipeline.tokenizer to avoid "Already borrowed" errors
        tokenizer = self.classifier_tokenizer  # Use the dedicated tokenizer, not pipeline.tokenizer
        
        chunks = []
        current_chunk_sents = []
        current_chunk_len = 0
        
        # Build chunks by accumulating sentences until we reach token limit
        for sent in sentences:
            sent_tokens = tokenizer.encode(sent, add_special_tokens=False)
            sent_len = len(sent_tokens)
            
            if current_chunk_len + sent_len > chunk_size - 2:  # Leave room for special tokens
                if current_chunk_sents:
                    chunks.append(" ".join(current_chunk_sents))
                current_chunk_sents = [sent]
                current_chunk_len = sent_len
            else:
                current_chunk_sents.append(sent)
                current_chunk_len += sent_len
                
        # Add the last chunk if any sentences remain
        if current_chunk_sents:
            chunks.append(" ".join(current_chunk_sents))
            
        return chunks
    
    def _process_batch(self, batch: List[str]) -> List[Dict]:
        """Process a batch of text chunks through the classifier pipeline with token length control."""
        try:
            # Ensure each chunk is within token limits
            processed_batch = []
            for chunk in batch:
                tokens = self.classifier_tokenizer.encode(chunk, add_special_tokens=True)
                if len(tokens) > self.chunk_size:
                    logger.debug(f"Truncating chunk from {len(tokens)} to {self.chunk_size} tokens")
                    tokens = tokens[:self.chunk_size]
                    chunk = self.classifier_tokenizer.decode(tokens[:-1])  # Remove partial token
                processed_batch.append(chunk)
            
            logger.debug(f"Processing batch of {len(processed_batch)} chunks")
            
            # Process batch with proper padding and truncation
            return self.text_classifier_pipeline(
                processed_batch,
                truncation=True,
                padding=True,
                max_length=self.chunk_size,
                batch_size=8,
                top_k=None  # Replaces return_all_scores=True
            )
        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
            return []
    

    def _safe_truncate_text(self, text: str, max_length: int = None) -> str:
        """
        Safely truncate text to a maximum token length without
        risking 'Already borrowed' errors.

        Args:
            text: Text to truncate
            max_length: Maximum token length (defaults to self.chunk_size)

        Returns:
            Truncated text that will fit within token limit
        """
        if not text:
            return ""

        if max_length is None:
            max_length = min(self.chunk_size, 512)  # Never exceed 512 tokens
        else:
            max_length = min(max_length, 512)  # Enforce hard limit of 512

        # Simple estimation: approximate 2.2 characters per token for English
        # This is slightly less conservative to improve throughput while still being safe
        approx_chars_per_token = 2.2  # Reduced from 2.5 for better efficiency
        max_char_length = int(max_length * approx_chars_per_token * 0.92)  # 8% safety margin

        if len(text) > max_char_length:
            logger.debug(f"Truncating text from {len(text)} chars to {max_char_length}")
            return text[:max_char_length]

        return text

    def _process_chunks_in_batches(self, chunks: List[str]) -> List[Dict]:
        """
        Process all text chunks in optimized batches using multiprocessing and Hugging Face datasets.map()
        for better performance on multi-core systems.

        Args:
            chunks: List of text chunks to classify

        Returns:
            List of classification results for each chunk
        """
        if not chunks:
            return []

        logger.info(f"Processing {len(chunks)} chunks using multicore batch processing approach")

        processed_chunks = [self._safe_truncate_text(chunk) for chunk in chunks]

        try:
            if DATASETS_AVAILABLE:
                hf_dataset = Dataset.from_dict({"text": processed_chunks})

                import multiprocessing
                total_cores = multiprocessing.cpu_count()
                # Use multiprocessing only if we have more than one chunk to process
                num_proc = min(total_cores, len(chunks)) if len(chunks) > 1 else 1

                def predict_text_batch(examples):
                    # This function is defined inside to have access to self, but a new pipeline
                    # is created to avoid serialization issues with multiprocessing.
                    local_pipeline = pipeline(
                        "text-classification",
                        model=self.classifier_model,
                        tokenizer=self.classifier_tokenizer,
                        device=self.device_id,
                        truncation=True,
                        padding=True,
                        max_length=512
                    )
                    output = {"results": []}
                    try:
                        output["results"] = local_pipeline(
                            examples["text"],
                            truncation=True,
                            padding=True,
                            max_length=512,
                        )
                    except Exception as e:
                        logger.error(f"Error in batch prediction: {e}")
                    return output

                # Suppress the num_proc warning by only setting it when num_proc > 1
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if num_proc > 1:
                        logger.info(f"Using {num_proc} CPU cores for parallel processing (out of {total_cores} available)")
                        results_dataset = hf_dataset.map(
                            predict_text_batch,
                            batched=True,
                            batch_size=4,
                            num_proc=num_proc
                        )
                    else:
                        logger.info("Processing with a single core (dataset size <= 1)")
                        results_dataset = hf_dataset.map(
                            predict_text_batch,
                            batched=True,
                            batch_size=4
                        )

                if "results" in results_dataset.column_names:
                    logger.info(f"Successfully processed {len(chunks)} chunks in parallel")
                    return results_dataset["results"]
                else:
                    logger.error("No 'results' column found in processed dataset")
                    return []
            else:
                # Fallback to standard multiprocessing if datasets is not available
                logger.info("Datasets library not available, using multiprocessing.Pool for parallel processing")
                from multiprocessing import Pool, cpu_count

                optimal_cores = max(1, min(cpu_count() - 1, 4))
                batch_size = 4
                all_results = []

                batches = [processed_chunks[i:i+batch_size] for i in range(0, len(processed_chunks), batch_size)]

                def process_batch(batch):
                    try:
                        return self.text_classifier_pipeline(
                            batch,
                            truncation=True,
                            padding=True,
                            max_length=512,
                            batch_size=self.gpu_batch_size
                        )
                    except Exception as batch_err:
                        logger.error(f"Error processing batch: {batch_err}")
                        return [[{"label": "UNKNOWN", "score": 0.0}]] * len(batch)

                logger.info(f"Processing {len(batches)} batches across {optimal_cores} CPU cores")
                with Pool(processes=optimal_cores) as pool:
                    batch_results = pool.map(process_batch, batches)

                for result in batch_results:
                    all_results.extend(result)

                logger.info(f"Parallel processing complete. Processed {len(all_results)} chunks.")
                return all_results
        except Exception as e:
            logger.error(f"Critical error in batch processing: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Critical error in batch processing: {e}", exc_info=True)
            return []
    
    def classify_document(self, text: str, chunk_size: int, stride: int) -> Dict[str, Any]:
        """
        Classify document to determine if it contains sensitive information using efficient batch processing.
        
        Args:
            text: Document text to classify.
            chunk_size: Maximum size of each chunk.
            stride: Overlap between chunks.
            
        Returns:
            Dictionary with classification results (contains_sensitive_data, confidence, classification).
        """
        if not text or not text.strip():
            logger.info("Empty text provided for classification. Returning non-sensitive.")
            return {
                "contains_sensitive_data": False,
                "confidence": 0.0,
                "classification": "non-sensitive",
                "error": "Empty input text",
                "chunk_count": 0
            }

        # Chunk the document to handle long texts
        chunks = self._chunk_text_by_sentences(text, chunk_size=chunk_size, stride=stride)
        if not chunks:
            logger.warning("No text chunks were generated for classification.")
            return {
                'contains_sensitive_data': False, 
                'confidence': 0.0, 
                'classification': 'No Content',
                'chunk_count': 0
            }

        # Process all chunks in optimized batches
        all_results = self._process_chunks_in_batches(chunks)
        
        # Parse results
        classifications = []
        confidences = []
        
        for result in all_results:
            if not result or not isinstance(result, list) or not result:
                continue
            
            # Get highest scoring label for this chunk
            try:
                top_label = max(result, key=lambda x: x['score'])
                predicted_label = top_label['label'].lower()
                score = top_label['score']
                
                is_sensitive = predicted_label in self.sensitive_label_keywords
                classifications.append(is_sensitive)
                confidences.append(score)
            except Exception as e:
                logger.warning(f"Could not determine classification from result: {e}")
                continue

        # Determine overall classification based on chunk results
        if not classifications:
            logger.warning("No chunks were successfully classified. Returning default non-sensitive.")
            return {
                "contains_sensitive_data": False,
                "confidence": 0.0,
                "classification": "non-sensitive",
                "chunk_count": len(chunks),
                "error": "No chunks successfully classified"
            }

        contains_sensitive_data = any(classifications)
        
        # Calculate confidence based on classification result
        if contains_sensitive_data:
            # Use maximum confidence from sensitive chunks
            relevant_confidences = [conf for i, conf in enumerate(confidences) if classifications[i]]
            final_confidence = max(relevant_confidences) if relevant_confidences else 0.0
        else:
            # Use minimum confidence from all chunks (weakest link)
            final_confidence = min(confidences) if confidences else 0.0
            
        return {
            "contains_sensitive_data": contains_sensitive_data,
            "confidence": float(final_confidence),
            "classification": "sensitive" if contains_sensitive_data else "non-sensitive",
            "chunk_count": len(chunks)
        }
        
    def _run_bert_ner(self, chunks: List[str], chunk_size: int, stride: int) -> List[Dict[str, Any]]:
        """Runs the BERT NER pipeline on text chunks and returns found entities."""
        bert_pii_entities = []
        if not self.ner_pipeline:
            return bert_pii_entities

        logger.info(f"Running BERT NER on {len(chunks)} chunks...")
        try:
            bert_results = self.ner_pipeline(chunks)
            for i, chunk_result in enumerate(bert_results):
                chunk_start_offset = i * (chunk_size - stride)
                for entity in chunk_result:
                    bert_pii_entities.append({
                        "entity_type": entity["entity_group"],
                        "value": entity["word"],
                        "start": entity["start"] + chunk_start_offset,
                        "end": entity["end"] + chunk_start_offset,
                        "confidence": entity["score"],
                        "source": "bert_ner"
                    })
            logger.info(f"BERT NER found {len(bert_pii_entities)} potential PII entities.")
        except Exception as e:
            logger.error(f"Error during BERT NER processing: {e}")
        return bert_pii_entities

    def _run_spacy_ner(self, text: str) -> List[Dict[str, Any]]:
        """Runs the spaCy NER pipeline and returns found entities."""
        spacy_pii_entities = []
        if not self.nlp:
            return spacy_pii_entities

        logger.info("Running spaCy NER...")
    def _run_bert_ner(self, chunks: List[str], original_text: str) -> List[Dict[str, Any]]:
        """Runs the BERT NER pipeline on chunks and maps entities back to original text."""
        if not self.ner_pipeline:
            logger.warning("BERT NER pipeline not available. Skipping BERT NER.")
            return []

        bert_pii_entities = []
        logger.info(f"Running BERT NER on {len(chunks)} chunks.")
        
        # Find chunk start positions in the original text to calculate absolute offsets
        chunk_offsets = []
        last_pos = 0
        for chunk in chunks:
            try:
                # Find the start of the chunk in the original text, starting from the last position
                start_pos = original_text.index(chunk, last_pos)
                chunk_offsets.append(start_pos)
                last_pos = start_pos + 1 # Move search start to after the beginning of the found chunk
            except ValueError:
                # This can happen if chunking logic modifies the text (e.g. joins sentences with space)
                logger.warning("Could not perfectly align a chunk with original text. Entity offsets may be approximate for this chunk.")
                chunk_offsets.append(last_pos) # Fallback to last known position

        try:
            # Process all chunks in a batch
            ner_results = self.ner_pipeline(chunks)

            for i, chunk_result in enumerate(ner_results):
                chunk_offset = chunk_offsets[i]
                for entity in chunk_result:
                    bert_pii_entities.append({
                        "entity_type": self.pii_definitions.get(entity['entity_group'], {}).get("label", entity['entity_group']),
                        "value": entity['word'],
                        "start": chunk_offset + entity['start'],
                        "end": chunk_offset + entity['end'],
                        "confidence": round(entity['score'], 4),
                        "source": "bert_ner"
                    })
            logger.info(f"BERT NER found {len(bert_pii_entities)} potential PII entities.")
        except Exception as e:
            logger.error(f"Error during BERT NER processing: {e}", exc_info=True)
            
        return bert_pii_entities

    def _run_spacy_ner(self, text: str) -> List[Dict[str, Any]]:
        """Runs the spaCy NER pipeline on the full text to find entities."""
        if not self.nlp:
            logger.warning("spaCy pipeline not available. Skipping spaCy NER.")
            return []

        spacy_pii_entities = []
        logger.info("Running spaCy NER...")
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                # Use a broader set of labels from spaCy's default NER
                if ent.label_ in self.pii_definitions or ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "MONEY", "NORP", "FAC", "LOC"]:
                    spacy_pii_entities.append({
                        "entity_type": self.pii_definitions.get(ent.label_, {}).get("label", ent.label_),
                        "value": ent.text,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": 1.0,  # spaCy rule-based/statistical entities are given high confidence
                        "source": "spacy_ner"
                    })
            logger.info(f"spaCy NER found {len(spacy_pii_entities)} potential PII entities.")
        except Exception as e:
            logger.error(f"Error during spaCy NER processing: {e}", exc_info=True)
        return spacy_pii_entities

    def detect_pii(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, stride: int = DEFAULT_STRIDE) -> List[Dict[str, Any]]:
        """
        Detect PII entities in text using multiple NER approaches (BERT, spaCy).
        
        Args:
            text: Input text to analyze for PII entities.
            chunk_size: Maximum size of each chunk.
            stride: Overlap between chunks.
            
        Returns:
            List of PII entities with type, value, position, and confidence.
        """
        if not text.strip():
            return []

        logger.info(f"Starting PII detection on text of length {len(text)}.")
        start_time = time.time()

        chunks = self._chunk_text_by_sentences(text, chunk_size, stride)

        # --- Run NER Pipelines ---
        bert_pii_entities = self._run_bert_ner(chunks, text)
        spacy_pii_entities = self._run_spacy_ner(text)

        # --- Merge and Filter PII ---
        all_pii_entities = bert_pii_entities + spacy_pii_entities
        logger.info(f"Collected {len(all_pii_entities)} raw entities from all NER sources.")

        merged_entities = self._merge_and_deduplicate_pii(all_pii_entities)
        logger.info(f"Found {len(merged_entities)} entities after merging and deduplication.")

        final_entities = self._filter_false_positives(merged_entities)
        logger.info(f"PII detection finished. Found {len(final_entities)} unique PII entities after filtering.")

        processing_time = time.time() - start_time
        logger.info(f"PII detection completed in {processing_time:.2f} seconds.")

        return final_entities

    def _merge_and_deduplicate_pii(self, pii_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sorts and deduplicates overlapping PII entities based on position and confidence."""
        if not pii_entities:
            return []

        # Sort by start position, then by confidence score descending for tie-breaking
        pii_entities.sort(key=lambda x: (x['start'], -x.get('confidence', 0.0)))

        deduplicated_entities = []
        if not pii_entities:
            return []

        current_entity = pii_entities[0]

        for next_entity in pii_entities[1:]:
            # Check for overlap: if next entity starts before current one ends
            if next_entity['start'] < current_entity['end']:
                # Overlap detected, decide which one to keep
                # Prioritize higher confidence
                if next_entity.get('confidence', 0.0) > current_entity.get('confidence', 0.0):
                    current_entity = next_entity  # Replace with higher confidence entity
                # If confidence is same, prioritize the longer entity
                elif (next_entity.get('confidence', 0.0) == current_entity.get('confidence', 0.0) and
                      (next_entity['end'] - next_entity['start']) > (current_entity['end'] - current_entity['start'])):
                    current_entity = next_entity  # Replace with longer entity
            else:
                # No overlap, add the current entity to our list and move to the next one
                deduplicated_entities.append(current_entity)
                current_entity = next_entity
        
        # Add the last processed entity
        deduplicated_entities.append(current_entity)

        return deduplicated_entities

    def _filter_false_positives(self, pii_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters out likely false positive PII detections.
        (Placeholder for more advanced filtering logic)
        """
        # Example: remove any detected 'PERSON' that is only one character long
        return [
            entity for entity in pii_entities
            if not (entity.get('entity_type') == 'Person Name' and len(entity.get('value', '')) <= 2)
        ]

    def analyze_document(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, stride: int = DEFAULT_STRIDE) -> Dict[str, Any]:
        """
        Performs a full analysis of a document, including classification and PII detection.
        
        Args:
            text: The document text to analyze.
            chunk_size: Maximum size of each chunk.
            stride: Overlap between chunks.
            
        Returns:
            A dictionary containing:
            - classification: Dictionary with classification results
            - pii_entities: List of detected PII entities
            - pii_count: Number of PII entities found
            - sensitivity_score: Confidence score of the classification
            - chunk_count: Number of chunks processed
            - processing_stats: Dictionary with processing statistics
        """
        import time
        start_time = time.time()
        
        # For smaller texts, sequential is more efficient
        classification = self.classify_document(text, chunk_size=chunk_size, stride=stride)
        pii_entities = self.detect_pii(text, chunk_size=chunk_size, stride=stride)
        
        # Calculate processing statistics
        processing_time = time.time() - start_time
        chars_per_second = len(text) / processing_time if processing_time > 0 else 0
        
        # Determine if document is sensitive based on BOTH classification AND detected PII
        contains_sensitive_data = classification["contains_sensitive_data"] or len(pii_entities) > 0
        
        # Adjust sensitivity score if PII is detected but classifier confidence is low
        sensitivity_score = classification["confidence"]
        if len(pii_entities) > 0 and sensitivity_score < 0.5:
            # Calculate new score based on number of PII entities (more PII = higher score)
            pii_based_score = min(0.95, 0.5 + (len(pii_entities) / 200))  # Cap at 0.95
            sensitivity_score = max(sensitivity_score, pii_based_score)  # Use the higher score
        
        # Enhanced return information including stats
        result = {
            "classification": {
                **classification,
                "contains_sensitive_data": contains_sensitive_data  # Override with combined assessment
            },
            "pii_entities": pii_entities,
            "pii_count": len(pii_entities),
            "sensitivity_score": sensitivity_score,  # Use adjusted score
            "chunk_count": classification.get("chunk_count", 1),
            "processing_stats": {
                "text_length": len(text),
                "processing_time_seconds": round(processing_time, 3),
                "chars_per_second": int(chars_per_second),
                "parallel_processing": False # Disabled parallel processing for simplicity for now
            }
        }
        
        # Log detailed results
        logger.info(f"Document analysis complete in {processing_time:.2f}s. "
                   f"Found {len(pii_entities)} PII entities with sensitivity score {sensitivity_score:.2f}")
        
        return result


if __name__ == '__main__':
    # Configure logging for standalone testing
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- Running NLPModel Standalone Test ---")
    
    # Initialize the model
    # This might take a moment to download models on first run
    try:
        nlp_model = NLPModel()
        
        # --- Test Case 1: Text with various PII ---
        test_text_1 = (
            "John Doe, resident of 123 Main Street, Anytown, USA, can be reached at john.doe@email.com. "
            "His phone number is (555) 123-4567. Please send the invoice to Jane Smith at "
            "jane.smith@company.org. Her social security number is 987-65-4321. "
            "The meeting is on 2023-10-27. My IP is 192.168.1.1."
        )
        
        logger.info("\n--- Analyzing Test Case 1 ---")
        analysis_result = nlp_model.analyze_document(test_text_1)
        
        import json
        import numpy as np

        # Helper function to convert NumPy types to native Python types
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            return obj
            
        print("\n--- Analysis Results ---")
        # Convert NumPy types before serializing to JSON
        serializable_result = convert_numpy_types(analysis_result)
        print(json.dumps(serializable_result, indent=2))

        # --- Test Case 2: Simple text with no PII ---
        test_text_2 = "The quick brown fox jumps over the lazy dog."
        logger.info("\n--- Analyzing Test Case 2 ---")
        analysis_result_2 = nlp_model.analyze_document(test_text_2)
        print("\n--- Analysis Results (No PII) ---")
        print(json.dumps(analysis_result_2, indent=2))
        assert analysis_result_2['pii_count'] == 0, "Should not find PII in simple text"
        print("\nTest Case 2 passed: No PII found as expected.")

    except Exception as e:
        logger.error(f"An error occurred during the standalone test: {e}", exc_info=True)

    logger.info("--- NLPModel Standalone Test Finished ---")
