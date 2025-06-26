"""
Base Agent for AI-Enhanced Data Privacy and Compliance Monitoring.

This module provides a common base class that all specialized agents can inherit from.
It handles shared functionality such as:
1. Configuration management
2. Logging
3. Resource management via context manager
4. Connection pooling
5. Error handling and retry logic
"""

import logging
import time
from pydantic import BaseModel
from typing import Dict, Any, Optional, Union, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import validate_arguments, ValidationError, Field

# Configure structured logging
logger = logging.getLogger(__name__)

class BaseAgentConfig(BaseModel):
    """Base configuration for all agents"""
    agent_name: str = "base_agent"
    log_level: str = "INFO"
    db_uri: Optional[str] = None
    db_pool_size: int = 5
    db_max_overflow: int = 10
    max_workers: int = 10
    retry_attempts: int = 3
    retry_backoff: float = 1.5

class BaseAgent:
    """
    Base Agent class that all specialized agents inherit from.
    
    Features:
    - Context manager for resource management
    - Dependency injection for components
    - Configurable via BaseAgentConfig
    - Robust error handling and retry mechanisms
    """
    
    @validate_arguments
    def __init__(self, config: Optional[BaseAgentConfig] = None):
        """
        Initialize the Base Agent with dependency injection.
        
        Args:
            config: Agent configuration
        """
        logger.debug(f"{self.__class__.__name__}.__init__ called.")
        
        # Initialize with default or provided config
        self.config = config or BaseAgentConfig(agent_name=self.__class__.__name__)
        
        # Configure logging based on config
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info(f"Initializing {self.config.agent_name} with config: {self.config}")
        
        # Initialize resources
        self._initialize_resources()
    
    def _initialize_resources(self):
        """Initialize resources needed by the agent"""
        logger.debug(f"{self.config.agent_name} initializing resources")
        # To be implemented by subclasses
        pass
    
    def _cleanup_resources(self):
        """Clean up resources used by the agent"""
        logger.debug(f"{self.config.agent_name} cleaning up resources")
        # To be implemented by subclasses
        pass
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with retry logic"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {self.config.agent_name}: {str(e)}")
            raise
    
    def __enter__(self):
        """Context manager entry point"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point - ensure resources are cleaned up"""
        self._cleanup_resources()
        if exc_type:
            logger.error(f"Exception in {self.config.agent_name}: {exc_type} - {exc_val}")
            return False  # Re-raise the exception
        return True
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the agent"""
        return {
            "status": "healthy",
            "agent_name": self.config.agent_name,
            "timestamp": time.time()
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent"""
        return {
            "agent_name": self.config.agent_name,
            "description": self.__doc__,
            "configuration": self.config.__dict__
        }
