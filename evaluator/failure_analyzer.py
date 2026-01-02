#!/usr/bin/env python3
"""
Failure Analyzer Module

Analyzes pytest output to determine the type of failure.
"""

from enum import Enum
from typing import Optional


class FailureType(str, Enum):
    """Types of test failures."""
    
    ASSERTION = "assertion"
    SYNTAX = "syntax"
    IMPORT = "import"
    NAME = "name"
    TYPE = "type"
    INDEX = "index"
    KEY = "key"
    ATTRIBUTE = "attribute"
    VALUE = "value"
    ZERODIVISION = "zerodivision"
    RECURSION = "recursion"
    TIMEOUT = "timeout"
    NONE = "none"
    UNKNOWN = "unknown"


class FailureAnalyzer:
    """Analyzes pytest output to determine failure types."""
    
    # Error patterns in order of priority
    ERROR_PATTERNS = [
        (["assertionerror", "assert "], FailureType.ASSERTION),
        (["syntaxerror"], FailureType.SYNTAX),
        (["importerror", "modulenotfounderror"], FailureType.IMPORT),
        (["nameerror"], FailureType.NAME),
        (["typeerror"], FailureType.TYPE),
        (["indexerror"], FailureType.INDEX),
        (["keyerror"], FailureType.KEY),
        (["attributeerror"], FailureType.ATTRIBUTE),
        (["valueerror"], FailureType.VALUE),
        (["zerodivisionerror"], FailureType.ZERODIVISION),
        (["recursionerror"], FailureType.RECURSION),
        (["timeoutexpired", "timeout"], FailureType.TIMEOUT),
    ]
    
    def analyze(self, error_output: str) -> FailureType:
        """
        Analyze pytest output to determine the type of failure.
        
        Args:
            error_output: The complete pytest output
            
        Returns:
            FailureType enum value indicating the type of failure
        """
        if not error_output:
            return FailureType.NONE
        
        output_lower = error_output.lower()
        
        for patterns, failure_type in self.ERROR_PATTERNS:
            for pattern in patterns:
                if pattern in output_lower:
                    return failure_type
        
        return FailureType.UNKNOWN
    
    def is_assertion_failure(self, error_output: str) -> bool:
        """
        Check if the failure is an assertion error.
        
        This is important for bug detection - only assertion failures
        indicate true bug detection (logic errors vs runtime errors).
        
        Args:
            error_output: The pytest output
            
        Returns:
            True if the failure is an assertion error
        """
        return self.analyze(error_output) == FailureType.ASSERTION
    
    def is_runtime_error(self, error_output: str) -> bool:
        """
        Check if the failure is a runtime error (not assertion).
        
        Runtime errors in bug detection are often false positives -
        the test detected an issue, but not the intended bug.
        
        Args:
            error_output: The pytest output
            
        Returns:
            True if the failure is a runtime error (not assertion)
        """
        failure_type = self.analyze(error_output)
        return failure_type not in (
            FailureType.NONE,
            FailureType.ASSERTION,
            FailureType.UNKNOWN,
        )
    
    def get_failure_description(self, failure_type: FailureType) -> str:
        """
        Get a human-readable description of the failure type.
        
        Args:
            failure_type: The FailureType enum value
            
        Returns:
            Human-readable description
        """
        descriptions = {
            FailureType.ASSERTION: "Assertion error (test logic found incorrect behavior)",
            FailureType.SYNTAX: "Syntax error in code",
            FailureType.IMPORT: "Import or module not found error",
            FailureType.NAME: "Name error (undefined variable)",
            FailureType.TYPE: "Type error (invalid operation on type)",
            FailureType.INDEX: "Index error (out of range)",
            FailureType.KEY: "Key error (missing dictionary key)",
            FailureType.ATTRIBUTE: "Attribute error (missing attribute)",
            FailureType.VALUE: "Value error (invalid value)",
            FailureType.ZERODIVISION: "Zero division error",
            FailureType.RECURSION: "Recursion error (maximum depth exceeded)",
            FailureType.TIMEOUT: "Timeout error (execution took too long)",
            FailureType.NONE: "No failure detected",
            FailureType.UNKNOWN: "Unknown error type",
        }
        return descriptions.get(failure_type, "Unknown failure type")

