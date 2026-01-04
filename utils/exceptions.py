"""
Custom Exceptions Module

Defines custom exception classes for better error handling across the project.
"""

from __future__ import annotations
from typing import Optional


class ProjectError(Exception):
    """Base exception class for all project-specific errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DatasetError(ProjectError):
    """Exception raised for dataset-related errors."""
    pass


class ModelConfigError(ProjectError):
    """Exception raised for model configuration errors."""
    pass


class TestGenerationError(ProjectError):
    """Exception raised during test case generation."""
    pass


class EvaluationError(ProjectError):
    """Exception raised during test evaluation."""
    pass


class APIError(ProjectError):
    """Exception raised for API-related errors."""
    
    def __init__(self, message: str, provider: Optional[str] = None, status_code: Optional[int] = None):
        """
        Initialize the API exception.
        
        Args:
            message: Error message
            provider: API provider name (e.g., 'anthropic', 'openai')
            status_code: HTTP status code if applicable
        """
        details = {}
        if provider:
            details['provider'] = provider
        if status_code:
            details['status_code'] = status_code
        super().__init__(message, details)
        self.provider = provider
        self.status_code = status_code

