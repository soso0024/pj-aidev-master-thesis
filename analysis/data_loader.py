#!/usr/bin/env python3
"""
Data Loading Module

Handles loading and parsing of test statistics files and combining them
with problem classification data.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any
from .problem_classifier import ProblemClassifier


class DataLoader:
    """Handles loading and parsing of test result statistics files."""

    def __init__(
        self,
        results_dir: str = "generated_tests",
        dataset_path: str = "dataset/HumanEval.jsonl",
    ):
        """Initialize the data loader with results and dataset paths."""
        self.results_dir = Path(results_dir)
        self.dataset_path = dataset_path
        self.data = []
        self.classifier = ProblemClassifier(dataset_path)
        # Define the desired order for configuration types
        self.config_order = ["basic", "ast", "docstring", "docstring_ast", "ast-fix"]

    def extract_model_from_path(self, file_path: Path) -> str:
        """Extract model information from the file path.
        Examples:
        - generated_tests_claude-3-5-haiku/file.json -> claude-3-5-haiku
        - generated_tests_claude-4-sonnet/file.json -> claude-4-sonnet
        - generated_tests/file.json -> unknown
        """
        parts = file_path.parts
        for part in parts:
            if part.startswith("generated_tests_"):
                model_name = part.replace("generated_tests_", "")
                return model_name
        return "unknown"

    def parse_filename(self, filename: str) -> Dict[str, Any]:
        """Parse configuration information from filename.
        Examples:
        - test_humaneval_0_success.stats.json -> {problem_id: 0, docstring: False, ast: False, success: True}
        - test_humaneval_5_docstring_ast_false.stats.json -> {problem_id: 5, docstring: True, ast: True, success: False}
        - test_humaneval_0_value_misuse_success.stats.json -> {problem_id: 0, bug_type: "value_misuse", success: True}
        """
        # Remove .stats.json extension
        base_name = filename.replace(".stats.json", "")

        # Extract problem ID
        match = re.search(r"test_humaneval_(\d+)", base_name)
        if not match:
            return None

        problem_id = int(match.group(1))

        # Check for bug type (HumanEvalPack)
        bug_type = None
        bug_type_patterns = [
            "value_misuse", "function_misuse", "excess_logic",
            "variable_misuse", "operator_misuse", "missing_condition"
        ]
        for bt in bug_type_patterns:
            if f"_{bt}_" in base_name or base_name.endswith(f"_{bt}"):
                bug_type = bt.replace("_", " ")
                # Remove bug type from base_name for further parsing
                base_name = base_name.replace(f"_{bt}", "")
                break

        # Check for configuration flags
        # Note: "ast-fix" is a separate option that should not be conflated with
        # the "ast" prompt-inclusion flag. Detect it first, then detect "_ast"
        # on a version of the name with "_ast-fix" removed.
        has_docstring = "_docstring" in base_name
        has_ast_fix = "_ast-fix" in base_name
        base_name_without_ast_fix = base_name.replace("_ast-fix", "")
        has_ast = "_ast" in base_name_without_ast_fix

        # Check success status
        is_success = base_name.endswith("_success")
        is_false = base_name.endswith("_false")

        # Determine configuration type
        if has_ast_fix:
            # Treat ast-fix as its own configuration bucket
            config_type = "ast-fix"
        elif has_docstring and has_ast:
            config_type = "docstring_ast"
        elif has_docstring:
            config_type = "docstring"
        elif has_ast:
            config_type = "ast"
        else:
            config_type = "basic"

        result = {
            "problem_id": problem_id,
            "has_docstring": has_docstring,
            "has_ast": has_ast,
            "has_ast_fix": has_ast_fix,
            "config_type": config_type,
            "success": is_success,
            "filename": filename,
        }
        
        # Add bug_type if present
        if bug_type:
            result["bug_type"] = bug_type
        
        return result

    def load_data(self) -> List[Dict[str, Any]]:
        """Load all stats.json files and parse them."""
        # First load problem classifications
        self.classifier.load_problem_classifications()

        # Determine search strategy based on the specified results directory
        search_paths = []

        # Check if user specified a specific model directory
        if self.results_dir.name.startswith("generated_tests_"):
            # User specified a specific model directory, only load from that
            search_paths = [self.results_dir]
            print(f"Loading data from specific model directory: {self.results_dir}")
        elif self.results_dir.name == "generated_tests":
            # User specified generic directory, look for all model-specific directories
            search_paths = [self.results_dir]
            base_dir = self.results_dir.parent
            # Include both regular and HumanEvalPack directories
            for path in base_dir.glob("generated_tests_*"):
                if path.is_dir():
                    search_paths.append(path)
            # Also look for HumanEvalPack-specific directories
            for path in base_dir.glob("generated_tests_humanevalpack_*"):
                if path.is_dir() and path not in search_paths:
                    search_paths.append(path)
            print(
                f"Loading data from {len(search_paths)} directories (including model-specific ones)..."
            )
        else:
            # Default: just use the specified directory
            search_paths = [self.results_dir]
            print(f"Loading data from specified directory: {self.results_dir}")

        all_stats_files = []

        for search_path in search_paths:
            stats_files = list(search_path.glob("*.stats.json"))
            all_stats_files.extend(stats_files)
            print(f"  Found {len(stats_files)} stats files in {search_path}")

        print(f"Total found {len(all_stats_files)} stats files")

        for stats_file in all_stats_files:
            # Parse filename for configuration info
            file_config = self.parse_filename(stats_file.name)
            if not file_config:
                print(f"Warning: Could not parse filename {stats_file.name}")
                continue

            # Extract model information from path
            model_name = self.extract_model_from_path(stats_file)

            # Load stats data
            try:
                with open(stats_file, "r") as f:
                    stats_data = json.load(f)

                # Combine filename info with stats data and model info
                combined_data = {**file_config, **stats_data, "model": model_name}

                # Handle missing code_coverage_percent field
                if "code_coverage_percent" not in combined_data:
                    combined_data["code_coverage_percent"] = 0.0

                # Handle missing dataset_type field (default to humaneval)
                if "dataset_type" not in combined_data:
                    combined_data["dataset_type"] = "humaneval"
                
                # Handle missing HumanEvalPack-specific fields
                if "bug_detection_success" not in combined_data:
                    combined_data["bug_detection_success"] = None
                if "canonical_solution_passed" not in combined_data:
                    combined_data["canonical_solution_passed"] = None
                if "buggy_solution_failed" not in combined_data:
                    combined_data["buggy_solution_failed"] = None
                if "buggy_failure_type" not in combined_data:
                    combined_data["buggy_failure_type"] = None
                if "is_true_positive" not in combined_data:
                    combined_data["is_true_positive"] = None
                if "is_false_positive" not in combined_data:
                    combined_data["is_false_positive"] = None

                # Add problem classification data
                problem_id = combined_data.get("problem_id")
                classification = self.classifier.get_classification(problem_id)
                if classification:
                    combined_data.update(classification)
                else:
                    # Add default classification if problem not found
                    combined_data.update(
                        {
                            "complexity_level": "unknown",
                            "algorithm_type": "unknown",
                            "complexity_score": 0,
                        }
                    )

                self.data.append(combined_data)

            except Exception as e:
                print(f"Error loading {stats_file}: {e}")

        print(f"Successfully loaded {len(self.data)} records")
        print(f"Models found: {sorted(set(d['model'] for d in self.data))}")
        
        # Print dataset type distribution
        dataset_types = {}
        for d in self.data:
            dtype = d.get("dataset_type", "humaneval")
            dataset_types[dtype] = dataset_types.get(dtype, 0) + 1
        
        if len(dataset_types) > 0:
            print(f"Dataset type distribution:")
            for dtype, count in sorted(dataset_types.items()):
                print(f"  - {dtype}: {count}")
        
        # Print HumanEvalPack-specific statistics if available
        humanevalpack_data = [d for d in self.data if d.get("dataset_type") == "humanevalpack"]
        if humanevalpack_data:
            print(f"\nHumanEvalPack statistics:")
            print(f"  Total problems: {len(humanevalpack_data)}")
            
            # Bug type distribution
            bug_types = {}
            for d in humanevalpack_data:
                bug_type = d.get("bug_type", "unknown")
                bug_types[bug_type] = bug_types.get(bug_type, 0) + 1
            
            print(f"  Bug type distribution:")
            for bug_type, count in sorted(bug_types.items()):
                print(f"    - {bug_type}: {count}")
            
            # True bug detection (assertion errors only)
            true_positive_count = sum(
                1 for d in humanevalpack_data 
                if d.get("is_true_positive") == True
            )
            
            # False positives (runtime errors, not logic errors)
            false_positive_count = sum(
                1 for d in humanevalpack_data 
                if d.get("is_false_positive") == True
            )
            
            # Total evaluated
            bug_detection_total = sum(
                1 for d in humanevalpack_data 
                if d.get("bug_detection_success") is not None
            )
            
            if bug_detection_total > 0:
                true_positive_rate = (true_positive_count / bug_detection_total) * 100
                false_positive_rate = (false_positive_count / bug_detection_total) * 100
                
                print(f"\n  Bug detection results:")
                print(f"    Total evaluated: {bug_detection_total}")
                print(f"    ✅ True Positives (AssertionError): {true_positive_count} ({true_positive_rate:.1f}%)")
                print(f"    ⚠️  False Positives (Runtime errors): {false_positive_count} ({false_positive_rate:.1f}%)")
                print(f"    ❌ False Negatives (Missed bugs): {bug_detection_total - true_positive_count - false_positive_count}")
                
                # Failure type distribution
                failure_types = {}
                for d in humanevalpack_data:
                    if d.get("buggy_failure_type"):
                        ft = d.get("buggy_failure_type")
                        failure_types[ft] = failure_types.get(ft, 0) + 1
                
                if failure_types:
                    print(f"\n  Failure type distribution:")
                    for ft, count in sorted(failure_types.items(), key=lambda x: x[1], reverse=True):
                        print(f"    - {ft}: {count}")
        
        return self.data

    def get_data(self) -> List[Dict[str, Any]]:
        """Get loaded data."""
        return self.data

    def get_classifier(self) -> ProblemClassifier:
        """Get the problem classifier instance."""
        return self.classifier
