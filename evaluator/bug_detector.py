#!/usr/bin/env python3
"""
Bug Detector Module

Handles bug detection evaluation for HumanEvalPack problems.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .test_runner import TestRunner
from .failure_analyzer import FailureAnalyzer, FailureType


@dataclass
class BugDetectionResult:
    """Result of bug detection evaluation."""
    
    canonical_passed: bool
    buggy_failed: bool  # True if buggy solution failed with ANY error
    true_bug_detected: bool  # True only if failed with AssertionError
    failure_type: FailureType
    c0_coverage: float
    c1_coverage: float
    buggy_file_path: Optional[str] = None
    
    @property
    def is_true_positive(self) -> bool:
        """True if this is a true bug detection (canonical passed, buggy failed with assertion)."""
        return self.canonical_passed and self.true_bug_detected
    
    @property
    def is_false_positive(self) -> bool:
        """True if buggy failed but not with assertion (runtime error)."""
        return (
            self.canonical_passed 
            and self.buggy_failed 
            and not self.true_bug_detected
            and self.failure_type != FailureType.NONE
        )
    
    @property
    def is_false_negative(self) -> bool:
        """True if buggy solution passed (bug not detected)."""
        return self.canonical_passed and not self.buggy_failed


class BugDetector:
    """Evaluates if generated tests can detect bugs in HumanEvalPack problems."""
    
    def __init__(self, test_runner: Optional[TestRunner] = None, verbose: bool = True):
        """
        Initialize the bug detector.
        
        Args:
            test_runner: TestRunner instance (creates new one if not provided)
            verbose: Whether to print detailed output
        """
        self.test_runner = test_runner or TestRunner()
        self.failure_analyzer = FailureAnalyzer()
        self.verbose = verbose
    
    def evaluate(
        self,
        test_file_path: str,
        problem: dict[str, Any],
    ) -> BugDetectionResult:
        """
        Evaluate if generated tests can detect bugs in the problem.
        
        This performs a two-phase evaluation:
        1. Run tests with canonical solution (should pass)
        2. Run tests with buggy solution (should fail with AssertionError)
        
        Args:
            test_file_path: Path to the test file
            problem: Dictionary containing problem data with:
                    - canonical_solution: The correct implementation
                    - buggy_solution: The buggy implementation
                    - bug_type: Type of bug in the buggy solution
            
        Returns:
            BugDetectionResult with detection metrics
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🔍 BUG DETECTION EVALUATION")
            print(f"{'='*80}")
            print(f"Bug type: {problem.get('bug_type', 'unknown')}")
        
        # Phase 1: Test with canonical solution (should pass)
        if self.verbose:
            print(f"\n📗 Phase 1: Testing with canonical solution...")
        
        canonical_result = self.test_runner.run_pytest(test_file_path)
        final_c0_coverage = canonical_result.c0_coverage
        final_c1_coverage = canonical_result.c1_coverage
        
        if self.verbose:
            if canonical_result.success:
                print(f"✅ Canonical solution: Tests PASSED "
                      f"(C0: {final_c0_coverage:.1f}%, C1: {final_c1_coverage:.1f}%)")
            else:
                print(f"❌ Canonical solution: Tests FAILED (unexpected)")
        
        # Check for buggy solution
        buggy_solution = problem.get("buggy_solution", "")
        if not buggy_solution:
            if self.verbose:
                print(f"⚠️  No buggy solution available for this problem")
            return BugDetectionResult(
                canonical_passed=canonical_result.success,
                buggy_failed=False,
                true_bug_detected=False,
                failure_type=FailureType.NONE,
                c0_coverage=final_c0_coverage,
                c1_coverage=final_c1_coverage,
            )
        
        # Phase 2: Test with buggy solution (should fail)
        if self.verbose:
            print(f"\n📕 Phase 2: Testing with buggy solution...")
        
        canonical_solution = problem.get("canonical_solution", "")
        
        # Run with buggy solution using temporary substitution
        buggy_result = self.test_runner.run_with_different_solution(
            test_file_path,
            canonical_solution,
            buggy_solution,
        )
        
        # Also save a separate buggy version file
        buggy_file_path = self._save_buggy_version(
            test_file_path, canonical_solution, buggy_solution
        )
        
        # Analyze the failure type
        failure_type = FailureType.NONE
        true_bug_detected = False
        buggy_failed = not buggy_result.success
        
        if buggy_failed:
            failure_type = self.failure_analyzer.analyze(buggy_result.output)
            true_bug_detected = failure_type == FailureType.ASSERTION
            
            if self.verbose:
                if true_bug_detected:
                    print(f"✅ Buggy solution: Tests FAILED with AssertionError")
                    print(f"🎯 TRUE BUG DETECTED: Test cases correctly identified incorrect behavior!")
                else:
                    print(f"⚠️  Buggy solution: Tests FAILED with {failure_type.value.upper()} error")
                    print(f"⚠️  This is NOT a true bug detection (runtime error, not logic error)")
        else:
            if self.verbose:
                print(f"❌ Buggy solution: Tests PASSED (bug NOT detected)")
        
        if self.verbose and buggy_file_path:
            print(f"💾 Buggy version saved to: {Path(buggy_file_path).name}")
        
        # Print summary
        result = BugDetectionResult(
            canonical_passed=canonical_result.success,
            buggy_failed=buggy_failed,
            true_bug_detected=true_bug_detected,
            failure_type=failure_type,
            c0_coverage=final_c0_coverage,
            c1_coverage=final_c1_coverage,
            buggy_file_path=buggy_file_path,
        )
        
        if self.verbose:
            self._print_summary(result)
        
        return result
    
    def _save_buggy_version(
        self,
        original_file_path: str,
        canonical_solution: str,
        buggy_solution: str,
    ) -> Optional[str]:
        """
        Save a separate file containing the buggy solution version.
        
        Args:
            original_file_path: Path to the original test file
            canonical_solution: The canonical solution to replace
            buggy_solution: The buggy solution to substitute
        
        Returns:
            Path to the saved buggy version file, or None if failed
        """
        try:
            original_path = Path(original_file_path)
            
            # Read original content
            with open(original_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Replace canonical with buggy
            buggy_content = content.replace(canonical_solution, buggy_solution)
            
            # Create buggy version filename
            stem = original_path.stem
            buggy_filename = f"{stem}_buggy{original_path.suffix}"
            buggy_file_path = original_path.parent / buggy_filename
            
            # Write buggy version
            with open(buggy_file_path, "w", encoding="utf-8") as f:
                f.write(buggy_content)
            
            return str(buggy_file_path)
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Warning: Could not save buggy version: {e}")
            return None
    
    def _print_summary(self, result: BugDetectionResult) -> None:
        """Print a summary of the bug detection result."""
        print(f"\n{'='*80}")
        
        if result.is_true_positive:
            print(f"🎉 BUG DETECTION: SUCCESS (True Positive)")
            print(f"   ✓ Canonical solution passed")
            print(f"   ✓ Buggy solution failed with AssertionError")
            print(f"   ✓ Test cases detected incorrect logic!")
        elif result.is_false_positive:
            print(f"⚠️  BUG DETECTION: FALSE POSITIVE")
            print(f"   ✓ Canonical solution passed")
            print(f"   ✗ Buggy solution failed with {result.failure_type.value.upper()} error")
            print(f"   ✗ This is a runtime error, not a logic error detection")
        elif not result.canonical_passed:
            print(f"⚠️  BUG DETECTION: INCONCLUSIVE")
            print(f"   ✗ Canonical solution failed")
            print(f"   → Cannot evaluate bug detection capability")
        else:
            print(f"❌ BUG DETECTION: FAILED (False Negative)")
            print(f"   ✓ Canonical solution passed")
            print(f"   ✗ Buggy solution passed (bug NOT detected)")
            print(f"   ✗ Test cases missed the bug")
        
        print(f"{'='*80}\n")

