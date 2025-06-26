"""
Reporting Agent for AI-Enhanced Data Privacy and Compliance Monitoring.

Key Features:
1. Automated compliance reports for GDPR Article 30, CCPA, and HIPAA (FR3.1)
2. Report scheduling and export in PDF/CSV formats (FR3.3)
3. Tamper-proof report storage in PostgreSQL for 3 years (FR3.4)
"""

import os
import logging
import time
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pydantic import validate_arguments

# Import base agent
import sys
import os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent, BaseAgentConfig

# Configure structured logging
logger = logging.getLogger(__name__)

@dataclass
class ReportingAgentConfig(BaseAgentConfig):
    """Configuration for Reporting Agent"""
    agent_name: str = "reporting_agent"
    
    # PostgreSQL configuration for tamper-proof storage
    postgres_uri: Optional[str] = None
    postgres_schema: str = "compliance_reports"
    
    # Report retention configuration
    report_retention_years: int = 3
    
    # Report types and formats
    supported_regulations: List[str] = field(default_factory=lambda: [
        "gdpr_article30", "ccpa", "hipaa"
    ])
    supported_formats: List[str] = field(default_factory=lambda: [
        "pdf", "csv"
    ])
    
    # Scheduling configuration
    default_schedule: Dict[str, str] = field(default_factory=lambda: {
        "gdpr_article30": "monthly",
        "ccpa": "quarterly",
        "hipaa": "monthly"
    })


class ReportingAgent(BaseAgent):
    """
    Reporting Agent for automated compliance reports and storage.
    
    Features:
    - Automated compliance report generation
    - Scheduled reporting
    - Tamper-proof storage in PostgreSQL
    - Export in multiple formats (PDF, CSV)
    """
    
    @validate_arguments
    def __init__(self, 
                config: Optional[ReportingAgentConfig] = None,
                report_generator: Optional[Any] = None,
                report_storage: Optional[Any] = None,
                report_scheduler: Optional[Any] = None):
        """
        Initialize the Reporting Agent with dependency injection.
        
        Args:
            config: Agent configuration
            report_generator: Pre-initialized report generator module
            report_storage: Pre-initialized report storage module
            report_scheduler: Pre-initialized report scheduler module
        """
        # Initialize base agent
        self.config = config or ReportingAgentConfig()
        super().__init__(config=self.config)
        
        logger.debug("Reporting agent initializing specialized components")
        
        # Initialize the component modules
        try:
            logger.debug("Importing ReportGeneratorModule...")
            from .report_generator import ReportGeneratorModule
            logger.debug("ReportGeneratorModule imported successfully")
            
            logger.debug("Importing ReportStorageModule...")
            from .report_storage import ReportStorageModule
            logger.debug("ReportStorageModule imported successfully")
            
            logger.debug("Importing ReportSchedulerModule...")
            from .report_scheduler import ReportSchedulerModule
            logger.debug("ReportSchedulerModule imported successfully")
            
            # Initialize components with dependency injection
            self.report_generator = report_generator or ReportGeneratorModule(self.config)
            self.report_storage = report_storage or ReportStorageModule(self.config)
            self.report_scheduler = report_scheduler or ReportSchedulerModule(self.config)
            
            logger.info("Reporting agent initialized successfully")
        
        except ImportError as e:
            logger.error(f"Failed to import required modules: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error initializing Reporting Agent: {str(e)}")
            raise
    
    def _initialize_resources(self):
        """Initialize resources needed by the reporting agent"""
        super()._initialize_resources()
        logger.debug("Initializing reporting-specific resources")
        
        # Initialize database connection for tamper-proof storage
        self.report_storage.initialize_storage()
        
        # Initialize report templates and generators
        self.report_generator.initialize_templates()
        
        # Initialize scheduler
        self.report_scheduler.initialize_scheduler()
    
    def _cleanup_resources(self):
        """Clean up resources used by the reporting agent"""
        logger.debug("Cleaning up reporting-specific resources")
        
        # Close storage connection
        if hasattr(self, 'report_storage'):
            self.report_storage.close_storage()
        
        # Cleanup scheduler
        if hasattr(self, 'report_scheduler'):
            self.report_scheduler.cleanup_scheduler()
        
        super()._cleanup_resources()
    
    def generate_report(self, 
                       report_type: str, 
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       format: str = "pdf") -> Dict[str, Any]:
        """
        Generate a compliance report.
        
        Args:
            report_type: Type of report to generate (e.g., gdpr_article30, ccpa, hipaa)
            start_date: Start date for the report period
            end_date: End date for the report period
            format: Output format (pdf, csv)
            
        Returns:
            Dictionary with report metadata and status
        """
        logger.info(f"Generating {report_type} report from {start_date} to {end_date} in {format} format")
        
        try:
            # Validate report type
            if report_type not in self.config.supported_regulations:
                raise ValueError(f"Unsupported report type: {report_type}")
            
            # Validate format
            if format not in self.config.supported_formats:
                raise ValueError(f"Unsupported format: {format}")
            
            # Generate the report
            report_data = self.report_generator.generate_report(
                report_type=report_type,
                start_date=start_date,
                end_date=end_date
            )
            
            # Store the report
            storage_result = self.report_storage.store_report(
                report_type=report_type,
                report_data=report_data,
                start_date=start_date,
                end_date=end_date
            )
            
            # Export the report in the requested format
            export_result = self.report_generator.export_report(
                report_data=report_data,
                format=format,
                report_id=storage_result["report_id"]
            )
            
            return {
                "status": "success",
                "report_id": storage_result["report_id"],
                "report_type": report_type,
                "format": format,
                "file_path": export_result["file_path"],
                "start_date": start_date,
                "end_date": end_date,
                "generation_date": datetime.now(),
                "expiry_date": datetime.now() + timedelta(days=365*self.config.report_retention_years)
            }
            
        except Exception as e:
            error_msg = f"Error generating {report_type} report: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    def schedule_report(self, 
                       report_type: str,
                       schedule: str,
                       format: str = "pdf") -> Dict[str, Any]:
        """
        Schedule a recurring report.
        
        Args:
            report_type: Type of report to generate (e.g., gdpr_article30, ccpa, hipaa)
            schedule: Schedule frequency (daily, weekly, monthly, quarterly)
            format: Output format (pdf, csv)
            
        Returns:
            Dictionary with schedule metadata and status
        """
        logger.info(f"Scheduling {report_type} report with {schedule} frequency in {format} format")
        
        try:
            # Schedule the report
            schedule_result = self.report_scheduler.schedule_report(
                report_type=report_type,
                schedule=schedule,
                format=format
            )
            
            return {
                "status": "success",
                "schedule_id": schedule_result["schedule_id"],
                "report_type": report_type,
                "schedule": schedule,
                "format": format,
                "next_run": schedule_result["next_run"]
            }
            
        except Exception as e:
            error_msg = f"Error scheduling {report_type} report: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    def get_reports(self, 
                  report_type: Optional[str] = None,
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None,
                  limit: int = 100,
                  offset: int = 0) -> Dict[str, Any]:
        """
        Get stored reports with optional filtering.
        
        Args:
            report_type: Filter by report type
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of reports to return
            offset: Pagination offset
            
        Returns:
            Dictionary with reports and metadata
        """
        try:
            reports = self.report_storage.get_reports(
                report_type=report_type,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                offset=offset
            )
            
            return {
                "status": "success",
                "data": reports["reports"],
                "count": reports["count"],
                "total": reports["total"],
                "limit": limit,
                "offset": offset
            }
            
        except Exception as e:
            error_msg = f"Error retrieving reports: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    def get_schedules(self, 
                    report_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get report schedules with optional filtering.
        
        Args:
            report_type: Filter by report type
            
        Returns:
            Dictionary with schedules and metadata
        """
        try:
            schedules = self.report_scheduler.get_schedules(report_type=report_type)
            
            return {
                "status": "success",
                "data": schedules,
                "count": len(schedules)
            }
            
        except Exception as e:
            error_msg = f"Error retrieving schedules: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}


# For direct testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create agent with test configuration
    config = ReportingAgentConfig()
    
    agent = ReportingAgent(config=config)
    
    # Test report generation
    result = agent.generate_report(
        report_type="gdpr_article30",
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )
    
    print(f"Report generation result: {result}")
"""
