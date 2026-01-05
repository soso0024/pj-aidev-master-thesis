"""
Evaluator package for test case evaluation and bug detection.

This package provides functionality for:
- Running pytest on generated test files
- Calculating code coverage (C0 and C1)
- Detecting bugs in HumanEvalPack problems
- Analyzing failure types
"""

from .test_runner import TestRunner, TestResult
from .bug_detector import BugDetector, BugDetectionResult
from .failure_analyzer import FailureAnalyzer

__all__ = [
    "TestRunner",
    "TestResult",
    "BugDetector",
    "BugDetectionResult",
    "FailureAnalyzer",
]

