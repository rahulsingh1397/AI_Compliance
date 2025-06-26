"""
Metadata Handler for Data Discovery Agent.

This module handles the storage and management of metadata about discovered
sensitive data, including classification results and data locations.
"""

import os
import json
import datetime
import uuid
from typing import List, Dict, Any, Optional
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON

# Define SQLAlchemy Base
Base = declarative_base()

class SensitiveDataRecord(Base):
    """SQLAlchemy model for sensitive data records."""
    __tablename__ = 'sensitive_data_records'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_type = Column(String(50))  # 'structured' or 'unstructured'
    source_location = Column(String(255))  # File path, database connection string, etc.
    source_identifier = Column(String(255))  # Table/column name, file name, etc.
    data_type = Column(String(50))  # PII, financial, health, etc.
    classification = Column(String(50))  # 'sensitive' or 'non-sensitive'
    confidence_score = Column(Float)
    discovery_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    details = Column(JSON)  # Additional details as JSON
    
    # Relationships
    pii_entities = relationship("PIIEntity", back_populates="record", cascade="all, delete-orphan")

class PIIEntity(Base):
    """SQLAlchemy model for individual PII entities found in a record."""
    __tablename__ = 'pii_entities'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    record_id = Column(String(36), ForeignKey('sensitive_data_records.id'))
    entity_type = Column(String(50))  # Name, Email, SSN, etc.
    entity_value = Column(String(255))  # Actual value or hash/masked value
    start_position = Column(Integer, nullable=True)  # For unstructured data
    end_position = Column(Integer, nullable=True)  # For unstructured data
    confidence_score = Column(Float)
    is_masked = Column(Boolean, default=False)
    
    # Relationships
    record = relationship("SensitiveDataRecord", back_populates="pii_entities")

class MetadataHandler:
    """
    Handler for storing and retrieving metadata about discovered sensitive data.
    Uses SQLAlchemy with PostgreSQL for persistent storage.
    """
    
    def __init__(self, db_uri: str = None):
        """
        Initialize the metadata handler with database connection.
        
        Args:
            db_uri: SQLAlchemy database URI (defaults to SQLite for development)
        """
        # Default to SQLite for development if no URI provided
        self.db_uri = db_uri or "sqlite:///sensitive_data_metadata.db"
        self.engine = create_engine(self.db_uri)
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
    
    def store_structured_data_metadata(self, 
                                      source_location: str,
                                      table_name: str,
                                      column_name: str,
                                      classification_result: Dict[str, Any]) -> str:
        """
        Store metadata for sensitive data found in structured data sources.
        
        Args:
            source_location: Database connection string or identifier
            table_name: Name of the database table
            column_name: Name of the column containing sensitive data
            classification_result: Result from the ML classifier
            
        Returns:
            ID of the created record
        """
        session = self.Session()
        
        try:
            # Create record
            record = SensitiveDataRecord(
                source_type="structured",
                source_location=source_location,
                source_identifier=f"{table_name}.{column_name}",
                data_type=self._determine_data_type(classification_result),
                classification=classification_result.get("classification", "unknown"),
                confidence_score=classification_result.get("confidence", 0.0),
                details={
                    "table_name": table_name,
                    "column_name": column_name,
                    "data_type": str(classification_result.get("data_type", "")),
                    "sample_values": classification_result.get("sample_values", [])
                }
            )
            
            session.add(record)
            session.commit()
            
            return record.id
            
        finally:
            session.close()
    
    def store_unstructured_data_metadata(self,
                                        source_location: str,
                                        analysis_result: Dict[str, Any],
                                        document_identifier: Optional[str] = None) -> str:
        """
        Store metadata for sensitive data found in unstructured data sources.
        
        Args:
            source_location: File path or document store identifier
            analysis_result: Result from the NLP model analysis
            document_identifier: Optional document name or ID for traceability
            
        Returns:
            ID of the created record
        """
        session = self.Session()
        
        try:
            # Create record
            record = SensitiveDataRecord(
                source_type="unstructured",
                source_location=source_location,
                source_identifier=document_identifier or source_location,  # Fallback to source_location if no document_id
                data_type=self._determine_data_type(analysis_result),
                classification=analysis_result.get("classification", {}).get("classification", "unknown"),
                confidence_score=analysis_result.get("classification", {}).get("confidence", 0.0),
                details={
                    "document_identifier": document_identifier,
                    "pii_count": analysis_result.get("pii_count", 0),
                    "sensitivity_score": analysis_result.get("sensitivity_score", 0.0)
                }
            )
            
            # Add PII entities
            for entity in analysis_result.get("pii_entities", []):
                pii_entity = PIIEntity(
                    entity_type=entity.get("type", "unknown"),
                    entity_value=self._mask_pii_value(entity.get("value", ""), entity.get("type", "unknown")),
                    start_position=entity.get("start"),
                    end_position=entity.get("end"),
                    confidence_score=entity.get("confidence", 0.0),
                    is_masked=True
                )
                record.pii_entities.append(pii_entity)
            
            session.add(record)
            session.commit()
            
            return record.id
            
        finally:
            session.close()
    
    def get_record_by_id(self, record_id: str) -> Dict[str, Any]:
        """
        Retrieve a sensitive data record by ID.
        
        Args:
            record_id: ID of the record to retrieve
            
        Returns:
            Dictionary with record data
        """
        session = self.Session()
        
        try:
            record = session.query(SensitiveDataRecord).filter_by(id=record_id).first()
            
            if not record:
                return None
            
            # Convert to dictionary
            result = {
                "id": record.id,
                "source_type": record.source_type,
                "source_location": record.source_location,
                "source_identifier": record.source_identifier,
                "data_type": record.data_type,
                "classification": record.classification,
                "confidence_score": record.confidence_score,
                "discovery_timestamp": record.discovery_timestamp.isoformat(),
                "last_updated": record.last_updated.isoformat(),
                "details": record.details,
                "pii_entities": []
            }
            
            # Add PII entities
            for entity in record.pii_entities:
                result["pii_entities"].append({
                    "id": entity.id,
                    "entity_type": entity.entity_type,
                    "entity_value": entity.entity_value,
                    "start_position": entity.start_position,
                    "end_position": entity.end_position,
                    "confidence_score": entity.confidence_score,
                    "is_masked": entity.is_masked
                })
            
            return result
            
        finally:
            session.close()
    
    def search_records(self, 
                      source_type: Optional[str] = None,
                      data_type: Optional[str] = None,
                      classification: Optional[str] = None,
                      min_confidence: float = 0.0,
                      limit: int = 100,
                      offset: int = 0) -> List[Dict[str, Any]]:
        """
        Search for sensitive data records based on criteria.
        
        Args:
            source_type: Filter by source type ('structured' or 'unstructured')
            data_type: Filter by data type (PII, financial, health, etc.)
            classification: Filter by classification ('sensitive' or 'non-sensitive')
            min_confidence: Minimum confidence score
            limit: Maximum number of records to return
            offset: Offset for pagination
            
        Returns:
            List of matching records
        """
        session = self.Session()
        
        try:
            query = session.query(SensitiveDataRecord)
            
            # Apply filters
            if source_type:
                query = query.filter(SensitiveDataRecord.source_type == source_type)
            
            if data_type:
                query = query.filter(SensitiveDataRecord.data_type == data_type)
            
            if classification:
                query = query.filter(SensitiveDataRecord.classification == classification)
            
            if min_confidence > 0:
                query = query.filter(SensitiveDataRecord.confidence_score >= min_confidence)
            
            # Apply pagination
            query = query.order_by(SensitiveDataRecord.discovery_timestamp.desc())
            query = query.limit(limit).offset(offset)
            
            # Convert to list of dictionaries
            results = []
            for record in query.all():
                results.append({
                    "id": record.id,
                    "source_type": record.source_type,
                    "source_location": record.source_location,
                    "source_identifier": record.source_identifier,
                    "data_type": record.data_type,
                    "classification": record.classification,
                    "confidence_score": record.confidence_score,
                    "discovery_timestamp": record.discovery_timestamp.isoformat(),
                    "pii_count": len(record.pii_entities)
                })
            
            return results
            
        finally:
            session.close()
    
    def _determine_data_type(self, result: Dict[str, Any]) -> str:
        """
        Determine the data type based on classification result.
        
        Args:
            result: Classification or analysis result
            
        Returns:
            Data type string
        """
        # Check for PII entities
        pii_entities = result.get("pii_entities", [])
        if pii_entities:
            entity_types = [entity.get("type", "").lower() for entity in pii_entities]
            
            if any("ssn" in t or "social security" in t for t in entity_types):
                return "PII-SSN"
            elif any("credit card" in t for t in entity_types):
                return "PII-Financial"
            elif any("address" in t for t in entity_types):
                return "PII-Address"
            elif any("email" in t for t in entity_types):
                return "PII-Contact"
            else:
                return "PII-General"
        
        # For structured data
        if "column_name" in result:
            column_name = result.get("column_name", "").lower()
            
            if any(term in column_name for term in ["ssn", "social", "security"]):
                return "PII-SSN"
            elif any(term in column_name for term in ["credit", "card", "payment", "account"]):
                return "PII-Financial"
            elif any(term in column_name for term in ["address", "street", "city", "zip"]):
                return "PII-Address"
            elif any(term in column_name for term in ["email", "phone", "contact"]):
                return "PII-Contact"
            elif any(term in column_name for term in ["health", "medical", "diagnosis"]):
                return "Health"
            elif any(term in column_name for term in ["salary", "income", "revenue"]):
                return "Financial"
        
        # Default
        return "General"
    
    def _mask_pii_value(self, value: str, entity_type: str) -> str:
        """
        Mask PII values for secure storage.
        
        Args:
            value: Original PII value
            entity_type: Type of PII entity
            
        Returns:
            Masked value
        """
        # Different masking strategies based on entity type
        if "SSN" in entity_type:
            # Mask SSN: XXX-XX-1234 -> XXX-XX-****
            if len(value) >= 4:
                return value[:-4] + "****"
            return "****"
        
        elif "Credit Card" in entity_type:
            # Mask credit card: XXXX-XXXX-XXXX-1234 -> XXXX-XXXX-XXXX-****
            if len(value) >= 4:
                return value[:-4] + "****"
            return "****"
        
        elif "Email" in entity_type:
            # Mask email: user@example.com -> u***@example.com
            if "@" in value:
                username, domain = value.split("@", 1)
                if len(username) > 1:
                    masked_username = username[0] + "***"
                    return f"{masked_username}@{domain}"
            return "****@****.com"
        
        elif "Phone" in entity_type:
            # Mask phone: XXX-XXX-1234 -> XXX-XXX-****
            if len(value) >= 4:
                return value[:-4] + "****"
            return "****"
        
        elif "Address" in entity_type:
            # Mask address: Return only general area
            parts = value.split(",")
            if len(parts) > 1:
                return "****," + ",".join(parts[1:])
            return "****"
        
        elif "Name" in entity_type:
            # Mask name: John Doe -> J*** D***
            parts = value.split()
            masked_parts = []
            for part in parts:
                if len(part) > 1:
                    masked_parts.append(part[0] + "***")
                else:
                    masked_parts.append("*")
            return " ".join(masked_parts)
        
        # Default masking
        if len(value) > 2:
            return value[0] + "****" + value[-1]
        return "****"
