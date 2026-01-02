#!/usr/bin/env python3
"""
Test Runner Module

Handles pytest execution and coverage analysis for generated test files.
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Constants
PYTEST_TIMEOUT_SECONDS = 60


@dataclass
class TestResult:
    """Result of a pytest execution."""
    
    success: bool
    output: str
    c0_coverage: float  # Statement coverage
    c1_coverage: float  # Branch coverage
    error_message: Optional[str] = None
    
    @property
    def coverage(self) -> float:
        """Alias for c0_coverage (backward compatibility)."""
        return self.c0_coverage


class TestRunner:
    """Handles pytest execution and coverage analysis."""
    
    def __init__(self, timeout: int = PYTEST_TIMEOUT_SECONDS):
        """
        Initialize the test runner.
        
        Args:
            timeout: Timeout in seconds for pytest execution
        """
        self.timeout = timeout
    
    def run_pytest(self, test_file_path: str) -> TestResult:
        """
        Run pytest on the test file and return results with coverage.
        
        Args:
            test_file_path: Path to the test file to run
            
        Returns:
            TestResult with success status, output, and coverage metrics
        """
        # Use absolute path and run from project root
        abs_path = Path(test_file_path).resolve()
        cmd = ["pytest", str(abs_path), "--cov", "--cov-branch", "-v"]

        try:
            # Run pytest and capture output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=Path.cwd(),
            )

            # Check if tests passed
            success = result.returncode == 0

            # Combine stdout and stderr for complete information
            output = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

            # Extract coverage percentages
            c0_coverage, c1_coverage = self._parse_coverage(result.stdout)

            return TestResult(
                success=success,
                output=output,
                c0_coverage=c0_coverage,
                c1_coverage=c1_coverage,
            )

        except subprocess.TimeoutExpired:
            return TestResult(
                success=False,
                output="",
                c0_coverage=0.0,
                c1_coverage=0.0,
                error_message=f"pytest execution timed out after {self.timeout} seconds",
            )
        except Exception as e:
            return TestResult(
                success=False,
                output="",
                c0_coverage=0.0,
                c1_coverage=0.0,
                error_message=f"Error running pytest: {str(e)}",
            )
    
    def _parse_coverage(self, stdout: str) -> tuple[float, float]:
        """
        Parse coverage percentages from pytest output.
        
        With --cov-branch, format is: "TOTAL  Stmts  Miss  Branch  BrPart  Cover"
        Example: "TOTAL    75     7      16       0    84%"
        
        C0 (Statement Coverage) = (Stmts - Miss) / Stmts * 100
        C1 (Branch Coverage) = (Branch - BrPart) / Branch * 100
        
        Args:
            stdout: pytest stdout output
            
        Returns:
            Tuple of (c0_coverage, c1_coverage)
        """
        c0_coverage = 0.0
        c1_coverage = 0.0
        
        if not stdout:
            return c0_coverage, c1_coverage
        
        # Try branch coverage format first
        coverage_match = re.search(
            r"TOTAL\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\d+%", stdout
        )
        if coverage_match:
            total_stmts = int(coverage_match.group(1))
            missed_stmts = int(coverage_match.group(2))
            total_branches = int(coverage_match.group(3))
            partial_branches = int(coverage_match.group(4))
            
            # Calculate C0 (Statement Coverage)
            if total_stmts > 0:
                c0_coverage = ((total_stmts - missed_stmts) / total_stmts) * 100
            else:
                c0_coverage = 100.0
            
            # Calculate C1 (Branch Coverage)
            if total_branches > 0:
                c1_coverage = ((total_branches - partial_branches) / total_branches) * 100
            else:
                c1_coverage = 100.0  # No branches means 100% coverage
        else:
            # Fallback to old format without branch coverage
            coverage_match = re.search(
                r"TOTAL\s+(\d+)\s+(\d+)\s+(\d+)%", stdout
            )
            if coverage_match:
                total_stmts = int(coverage_match.group(1))
                missed_stmts = int(coverage_match.group(2))
                if total_stmts > 0:
                    c0_coverage = ((total_stmts - missed_stmts) / total_stmts) * 100
                else:
                    c0_coverage = 100.0
                c1_coverage = 100.0  # No branch coverage available
        
        return c0_coverage, c1_coverage
    
    def run_with_different_solution(
        self,
        test_file_path: str,
        original_solution: str,
        new_solution: str,
    ) -> TestResult:
        """
        Run pytest with a different solution substituted into the test file.
        
        Useful for bug detection where we want to test against a buggy solution.
        
        Args:
            test_file_path: Path to the test file
            original_solution: The original solution code to replace
            new_solution: The new solution code to substitute
            
        Returns:
            TestResult from running with the new solution
        """
        # Read the test file
        with open(test_file_path, "r", encoding="utf-8") as f:
            original_content = f.read()
        
        # Replace the solution
        modified_content = original_content.replace(original_solution, new_solution)
        
        # Write modified content
        with open(test_file_path, "w", encoding="utf-8") as f:
            f.write(modified_content)
        
        try:
            # Run pytest
            result = self.run_pytest(test_file_path)
            return result
        finally:
            # Restore original content
            with open(test_file_path, "w", encoding="utf-8") as f:
                f.write(original_content)

