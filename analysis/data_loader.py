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
        """
        # Remove .stats.json extension
        base_name = filename.replace(".stats.json", "")

        # Extract problem ID
        match = re.search(r"test_humaneval_(\d+)", base_name)
        if not match:
            return None

        problem_id = int(match.group(1))

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

        return {
            "problem_id": problem_id,
            "has_docstring": has_docstring,
            "has_ast": has_ast,
            "has_ast_fix": has_ast_fix,
            "config_type": config_type,
            "success": is_success,
            "filename": filename,
        }

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
            for path in base_dir.glob("generated_tests_*"):
                if path.is_dir():
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
        return self.data

    def get_data(self) -> List[Dict[str, Any]]:
        """Get loaded data."""
        return self.data

    def get_classifier(self) -> ProblemClassifier:
        """Get the problem classifier instance."""
        return self.classifier
