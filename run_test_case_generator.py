"""
HumanEval Test Case Generator using Claude API

This script reads the HumanEval dataset, randomly selects problems,
and generates pytest-compatible test cases using Claude API.
"""

import json
import random
import os
import argparse
import re
import ast
import subprocess
import requests
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv
from model_utils import get_available_models, get_default_model
from prompts import PromptBuilder
from llm_clients import create_clients_for_models, get_client_for_model, LLMResponse
from evaluator import TestRunner, BugDetector, FailureAnalyzer

# Import datasets library for HumanEvalPack
try:
    from datasets import load_dataset

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


# Constants for maintainability
PYTEST_TIMEOUT_SECONDS = 60  # Timeout for pytest execution
DEFAULT_MAX_TOKENS = (
    8000  # Max tokens for LLM response (high limit for research evaluation)
)
DEFAULT_TEMPERATURE = 0.0  # Temperature for deterministic responses
DISPLAY_LINE_LIMIT = 20  # Max lines to display without truncation
TRUNCATE_HEAD_LINES = 10  # Lines to show at start when truncating
TRUNCATE_TAIL_LINES = 10  # Lines to show at end when truncating
GENERATED_TEST_MARKER = (
    "# Generated test cases:\n"  # Marker for generated test code section
)


class TestCaseGenerator:
    def __init__(
        self,
        api_key: str = None,
        models: list[str] = None,
        include_docstring: bool = False,
        include_ast: bool = False,
        show_prompt: bool = False,
        enable_evaluation: bool = True,
        max_fix_attempts: int = 3,
        verbose_evaluation: bool = True,
        config_path: str = "models_config.json",
        dataset_type: str = "humaneval",
    ):
        """Initialize the test case generator with LLM clients."""
        self.api_key = api_key
        self.problems = []
        self.dataset_type = dataset_type

        # Load model configuration from external file
        self.config = self._load_model_config(config_path)
        self.model_mapping = {
            model: config["api_name"] for model, config in self.config["models"].items()
        }

        # Set default model if none provided
        if models is None:
            models = [self.config["default_model"]]

        # Validate models
        self.models = []
        for model in models:
            if model in self.model_mapping:
                self.models.append(model)
            else:
                raise ValueError(
                    f"Unsupported model: {model}. Supported models: {list(self.model_mapping.keys())}"
                )

        # Initialize clients for different providers (after models are set)
        self.clients = self._initialize_clients()

        self.include_docstring = include_docstring
        self.include_ast = include_ast
        self.show_prompt = show_prompt
        self.enable_evaluation = enable_evaluation
        self.max_fix_attempts = max_fix_attempts
        self.verbose_evaluation = verbose_evaluation
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

        # Initialize prompt builder for dynamic template loading
        self.prompt_builder = PromptBuilder()

        # Initialize evaluator components
        self.test_runner = TestRunner(timeout=PYTEST_TIMEOUT_SECONDS)
        self.bug_detector = BugDetector(
            test_runner=self.test_runner, verbose=self.verbose_evaluation
        )
        self.failure_analyzer = FailureAnalyzer()

    def _initialize_clients(self) -> dict[str, Any]:
        """Initialize clients for different LLM providers.

        Uses the llm_clients module for a cleaner, more maintainable implementation.
        """
        return create_clients_for_models(
            models_config=self.config,
            selected_models=self.models,
            anthropic_api_key=self.api_key,
        )

    def _load_model_config(self, config_path: str) -> dict[str, Any]:
        """Load model configuration from JSON file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model configuration file not found: {config_path}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in model configuration file: {e}")

    def get_model_folder_name(self, model: str) -> str:
        """Convert model name to a folder-friendly string."""
        return self.config["models"][model]["folder_name"]

    def load_dataset(self, file_path: str) -> None:
        """Load the dataset based on dataset_type."""
        if self.dataset_type == "humanevalpack":
            self.load_humanevalpack_dataset()
        else:
            self.load_humaneval_dataset(file_path)

    def load_humaneval_dataset(self, file_path: str = None) -> None:
        """Load the HumanEval dataset from Hugging Face.

        Args:
            file_path: Legacy parameter, kept for backward compatibility but not used.
                      Dataset is now loaded from Hugging Face instead.
        """
        if not DATASETS_AVAILABLE:
            raise ImportError(
                "datasets library is required for HumanEval. "
                "Install it with: pip install datasets"
            )

        print("Loading HumanEval dataset from Hugging Face...")

        try:
            # Load the HumanEval dataset from Hugging Face
            dataset = load_dataset("openai/openai_humaneval", split="test")

            # Convert dataset to problem format compatible with existing code
            for item in dataset:
                problem = {
                    "task_id": item["task_id"],
                    "prompt": item["prompt"],
                    "entry_point": item["entry_point"],
                    "canonical_solution": item["canonical_solution"],
                    "test": item["test"],
                }
                self.problems.append(problem)

            print(f"Loaded {len(self.problems)} problems from HumanEval dataset")

        except Exception as e:
            raise RuntimeError(f"Failed to load HumanEval dataset: {e}")

    def load_humanevalpack_dataset(self) -> None:
        """Load the HumanEvalPack dataset from Hugging Face."""
        if not DATASETS_AVAILABLE:
            raise ImportError(
                "datasets library is required for HumanEvalPack. "
                "Install it with: pip install datasets"
            )

        print("Loading HumanEvalPack dataset from Hugging Face...")

        try:
            # Load the Python subset of HumanEvalPack
            dataset = load_dataset("bigcode/humanevalpack", "python", split="test")

            # Convert dataset to problem format compatible with existing code
            for item in dataset:
                problem = {
                    "task_id": item["task_id"],
                    "prompt": item["prompt"],
                    "entry_point": item["entry_point"],
                    "canonical_solution": item["canonical_solution"],
                    "buggy_solution": item["buggy_solution"],
                    "bug_type": item["bug_type"],
                    "test": item["test"],
                }
                self.problems.append(problem)

            print(f"Loaded {len(self.problems)} problems from HumanEvalPack dataset")

            # Print bug type distribution
            bug_types = {}
            for problem in self.problems:
                bug_type = problem.get("bug_type", "unknown")
                bug_types[bug_type] = bug_types.get(bug_type, 0) + 1

            print(f"Bug type distribution:")
            for bug_type, count in sorted(bug_types.items()):
                print(f"  - {bug_type}: {count}")

        except Exception as e:
            raise RuntimeError(f"Failed to load HumanEvalPack dataset: {e}")

    def select_random_problem(self) -> dict[str, Any]:
        """Randomly select a problem from the dataset."""
        return random.choice(self.problems)

    def extract_function_signature(self, prompt: str, entry_point: str) -> str:
        """Extract just the function signature without docstring.

        Delegates to PromptBuilder for consistency.
        """
        return self.prompt_builder.extract_function_signature(prompt, entry_point)

    def generate_ast_string(
        self, canonical_solution: str, prompt: str, entry_point: str
    ) -> str:
        """Generate a readable AST representation of the canonical solution.

        Delegates to PromptBuilder for consistency.
        """
        return self.prompt_builder.generate_ast_string(
            canonical_solution, prompt, entry_point
        )

    def generate_prompt(self, problem: dict[str, Any]) -> str:
        """Create a prompt for Claude to generate test cases.

        Uses PromptBuilder for dynamic template loading to ensure
        research reproducibility.
        """
        return self.prompt_builder.build_prompt(
            problem=problem,
            include_docstring=self.include_docstring,
            include_ast=self.include_ast,
        )

    def clean_generated_code(self, raw_response: str) -> str:
        """Clean the generated response to extract only executable Python code."""
        # Remove markdown code blocks
        cleaned = re.sub(r"```python\s*\n?", "", raw_response)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)

        # Remove explanatory text before the first import or function definition
        lines = cleaned.split("\n")
        code_lines = []
        code_started = False
        seen_imports = set()  # Track imports to avoid duplicates

        for line in lines:
            stripped_line = line.strip()

            # Start collecting lines when we see imports, decorators, or function definitions
            if not code_started and (
                stripped_line.startswith("import ")
                or stripped_line.startswith("from ")
                or stripped_line.startswith("def ")
                or stripped_line.startswith("@pytest")
                or stripped_line.startswith("@")
            ):
                code_started = True

            if code_started:
                # Check for duplicate imports
                if stripped_line.startswith("import ") or stripped_line.startswith(
                    "from "
                ):
                    if stripped_line not in seen_imports:
                        seen_imports.add(stripped_line)
                        code_lines.append(line)
                    # Skip duplicate imports
                else:
                    code_lines.append(line)

        # Try to validate the code by checking for syntax errors
        try:
            result = "\n".join(code_lines).strip()
            # Basic validation - try to compile the code
            compile(result, "<string>", "exec")

            # If successful and contains test functions, return it
            if result and "def test_" in result:
                return result
        except SyntaxError:
            # If there's a syntax error, try to return the original response
            pass

        # Fallback: if validation fails, return the original response
        return raw_response

    def display_prompt_and_confirm(self, prompt: str) -> bool:
        """Display the prompt to user and ask for confirmation."""
        print("\n" + "=" * 80)
        print("PROMPT PREVIEW")
        print("=" * 80)
        print(prompt)
        print("=" * 80)

        # Estimate token count (rough approximation: 1 token ≈ 4 chars)
        estimated_tokens = len(prompt) // 4
        # Use the first model for cost estimation in prompt preview
        first_model = self.models[0] if self.models else self.config["default_model"]
        estimated_cost = (estimated_tokens / 1_000_000) * self.config["models"][first_model][
            "pricing"
        ]["input_per_1M"]

        print(f"Estimated input tokens: {estimated_tokens}")
        print(f"Estimated input cost: ${estimated_cost:.6f}")
        print("=" * 80)

        while True:
            response = input("\nProceed with this prompt? (y/n): ").lower().strip()
            if response in ["y", "yes"]:
                return True
            elif response in ["n", "no"]:
                print("Cancelled. Exiting...")
                return False
            else:
                print("Please enter 'y' (yes) or 'n' (no)")

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, model: str
    ) -> float:
        """Calculate the cost based on token usage and model."""
        pricing = self.config["models"][model]["pricing"]
        input_cost = (input_tokens / 1_000_000) * pricing["input_per_1M"]
        output_cost = (output_tokens / 1_000_000) * pricing["output_per_1M"]
        return input_cost + output_cost

    def get_total_fix_attempts(self) -> int:
        """Get the total number of fix attempts available."""
        return self.max_fix_attempts

    def get_usage_stats(self) -> dict[str, Any]:
        """Get current usage statistics."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 6),
        }

    def rename_file_with_result(
        self, original_filepath: str, evaluation_success: bool
    ) -> str:
        """Rename the test file to include success/failure status."""
        original_path = Path(original_filepath)

        # Get the current filename without extension
        stem = original_path.stem

        # Remove any existing success/failure suffix
        if stem.endswith("_success") or stem.endswith("_failed"):
            stem = "_".join(stem.split("_")[:-1])

        # Add the new suffix based on evaluation result
        suffix = "success" if evaluation_success else "failed"
        new_stem = f"{stem}_{suffix}"

        # Create new filepath
        new_filepath = original_path.parent / f"{new_stem}{original_path.suffix}"

        # Rename the file
        if original_path.exists():
            original_path.rename(new_filepath)
            print(f"📝 Renamed {original_path.name} → {new_filepath.name}")

        return str(new_filepath)

    def update_final_stats(
        self,
        filepath: str,
        problem: dict[str, Any],
        evaluation_success: bool,
        fix_attempts_used: int,
        c0_coverage: float = 0.0,
        c1_coverage: float = 0.0,
        canonical_passed: bool = None,
        buggy_failed: bool = None,
        failure_type: str = None,
    ) -> str:
        """Update the stats file with final statistics after evaluation process.

        Returns:
            str: The final filepath (potentially renamed)
        """
        # Rename file with evaluation result
        final_filepath = self.rename_file_with_result(filepath, evaluation_success)

        # Also rename the stats file to match
        original_stats_filepath = Path(filepath).with_suffix(".stats.json")
        final_stats_filepath = Path(final_filepath).with_suffix(".stats.json")

        final_stats = self.get_usage_stats()
        final_stats.update(
            {
                "task_id": problem["task_id"],
                "generated_file": str(final_filepath),
                "evaluation_enabled": self.enable_evaluation,
                "evaluation_success": evaluation_success,
                "fix_attempts_used": fix_attempts_used,
                "max_fix_attempts": self.max_fix_attempts,
                "code_coverage_c0_percent": c0_coverage,
                "code_coverage_c1_percent": c1_coverage,
                "dataset_type": self.dataset_type,
                # Add prompt template info for reproducibility
                "prompt_config": {
                    "include_docstring": self.include_docstring,
                    "include_ast": self.include_ast,
                    "template_hash": self.prompt_builder.get_template_hash(
                        self.include_docstring, self.include_ast
                    ),
                },
            }
        )

        # Add HumanEvalPack-specific stats
        if self.dataset_type == "humanevalpack":
            # True bug detection only if assertion error
            true_bug_detection = (
                canonical_passed and buggy_failed
                if canonical_passed is not None
                else None
            )

            final_stats.update(
                {
                    "bug_type": problem.get("bug_type", "unknown"),
                    "bug_detection_success": true_bug_detection,
                    "canonical_solution_passed": canonical_passed,
                    "buggy_solution_failed": buggy_failed,
                    "buggy_failure_type": failure_type,
                    "is_true_positive": true_bug_detection
                    and failure_type == "assertion",
                    "is_false_positive": (
                        canonical_passed
                        and buggy_failed is not False
                        and failure_type not in ["assertion", "none", None]
                    ),
                }
            )

        with open(final_stats_filepath, "w", encoding="utf-8") as f:
            json.dump(final_stats, f, indent=2)

        # Remove old stats file if it's different
        if (
            original_stats_filepath != final_stats_filepath
            and original_stats_filepath.exists()
        ):
            original_stats_filepath.unlink()

        print(f"📊 Final stats saved to {final_stats_filepath}")
        return str(final_filepath)

    def display_pytest_errors(self, error_output: str, attempt: int) -> None:
        """Display pytest errors in a readable format."""
        if not self.verbose_evaluation:
            return

        print(f"\n{'='*80}")
        print(f"📋 PYTEST ERROR DETAILS - Attempt {attempt}")
        print(f"{'='*80}")

        # Parse and display relevant parts of pytest output
        lines = error_output.split("\n")
        in_failures = False
        in_errors = False

        for line in lines:
            # Show test results summary
            if "FAILED" in line and "::" in line:
                print(f"❌ {line}")
            elif "PASSED" in line and "::" in line:
                print(f"✅ {line}")
            elif "=== FAILURES ===" in line:
                in_failures = True
                print(f"\n🔍 FAILURE DETAILS:")
                print("-" * 40)
            elif "=== ERRORS ===" in line:
                in_errors = True
                print(f"\n🚨 ERROR DETAILS:")
                print("-" * 40)
            elif in_failures and line.startswith("="):
                if "short test summary" in line:
                    in_failures = False
                    print("\n" + "=" * 40)
                else:
                    print(line)
            elif in_failures:
                print(line)
            elif (
                "SyntaxError" in line
                or "ImportError" in line
                or "ModuleNotFoundError" in line
            ):
                print(f"🐛 {line}")

        print(f"{'='*80}")
        print(f"📋 END OF ERROR DETAILS - Attempt {attempt}")
        print(f"{'='*80}")
        print(f"\n{'─'*80}")
        print(f"⏸️  PAUSING - Error details for Attempt {attempt} shown above")
        print(f"{'─'*80}\n")

    def display_fix_prompt(self, prompt: str, attempt: int) -> None:
        """Display the fix prompt being sent to LLM."""
        if not self.verbose_evaluation:
            return

        # Add clear visual separator before new fix prompt section
        print(f"\n{'#'*80}")
        print(f"{'#'*80}")
        print(f"{'#'*80}")
        total_fix_attempts = self.get_total_fix_attempts()
        print(f"🤖 LLM FIX PROMPT - Fix attempt {attempt} of {total_fix_attempts}")
        print(f"{'#'*80}")
        print(f"{'#'*80}")
        print(f"{'#'*80}\n")
        print(prompt)
        print(f"\n{'='*80}")
        print(f"🤖 END OF FIX PROMPT - Fix attempt {attempt} of {total_fix_attempts}")
        print(f"{'='*80}")

        # Ask user if they want to proceed (optional)
        if self.show_prompt:
            print(f"\n{'─'*80}")
            while True:
                response = (
                    input(f"Proceed with fix attempt {attempt}? (y/n): ")
                    .lower()
                    .strip()
                )
                if response in ["y", "yes"]:
                    print(f"{'─'*80}\n")
                    break
                elif response in ["n", "no"]:
                    print("Skipping fix attempt...")
                    print(f"{'─'*80}\n")
                    return False
                else:
                    print("Please enter 'y' (yes) or 'n' (no)")

        return True

    def display_fix_response(self, response: str, attempt: int) -> None:
        """Display the LLM's fix response."""
        if not self.verbose_evaluation:
            return

        print(f"\n{'='*80}")
        total_fix_attempts = self.get_total_fix_attempts()
        print(f"🔧 LLM FIX RESPONSE - Fix attempt {attempt} of {total_fix_attempts}")
        print(f"{'='*80}")

        # Show first few lines and last few lines of the response
        lines = response.split("\n")
        if len(lines) <= DISPLAY_LINE_LIMIT:
            print(response)
        else:
            print("\n".join(lines[:TRUNCATE_HEAD_LINES]))
            omitted_lines = len(lines) - (TRUNCATE_HEAD_LINES + TRUNCATE_TAIL_LINES)
            print(f"\n... ({omitted_lines} lines omitted) ...\n")
            print("\n".join(lines[-TRUNCATE_TAIL_LINES:]))

        print(f"{'='*80}")
        print(f"🔧 END OF FIX RESPONSE - Fix attempt {attempt} of {total_fix_attempts}")
        print(f"{'='*80}")
        print(f"\n{'─'*80}")
        print(f"✅ Fix response received for attempt {attempt}")
        print(f"{'─'*80}\n")

    def generate_test_cases(self, problem: dict[str, Any], model: str) -> str:
        """Generate test cases using LLM API.

        Uses the unified LLM client interface for cleaner code.
        """
        prompt = self.generate_prompt(problem)

        print(f"Generating test cases for {problem['task_id']} using {model}...")

        # Show prompt and get confirmation if requested
        if self.show_prompt:
            if not self.display_prompt_and_confirm(prompt):
                return ""

        try:
            # Get the appropriate client for this model
            client = get_client_for_model(self.clients, model, self.config)

            # Generate response using unified interface
            response: LLMResponse = client.generate(
                prompt=prompt,
                model=self.model_mapping[model],
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE,
            )

            # Track token usage and cost
            self.total_input_tokens += response.input_tokens
            self.total_output_tokens += response.output_tokens

            cost = self.calculate_cost(
                response.input_tokens, response.output_tokens, model
            )
            self.total_cost += cost

            return self.clean_generated_code(response.content)

        except RuntimeError as e:
            print(f"❌ {e}")
            return ""
        except Exception as e:
            print(f"❌ Error generating test cases with {model}: {e}")
            return ""

    def save_test_cases(
        self, problem: dict[str, Any], test_cases: str, output_dir: str, model: str
    ) -> str:
        """Save generated test cases to a file."""
        # Create model-specific output directory
        model_folder = self.get_model_folder_name(model)

        # Add dataset type to directory name for HumanEvalPack
        if self.dataset_type == "humanevalpack":
            model_output_dir = f"{output_dir}_humanevalpack_{model_folder}"
        else:
            model_output_dir = f"{output_dir}_{model_folder}"

        output_path = Path(model_output_dir)
        output_path.mkdir(exist_ok=True)

        # Create filename from task_id (no model name in filename since folder identifies model)
        base_name = f"test_{problem['task_id'].replace('/', '_').lower()}"
        filename_parts = [base_name]

        # Add bug type to filename for HumanEvalPack
        if self.dataset_type == "humanevalpack" and "bug_type" in problem:
            bug_type_str = problem["bug_type"].replace(" ", "_")
            filename_parts.append(bug_type_str)

        if self.include_docstring:
            filename_parts.append("docstring")
        if self.include_ast:
            filename_parts.append("ast")

        filename = f"{'_'.join(filename_parts)}.py"
        filepath = output_path / filename

        # Add the function implementation and test cases
        full_content = f"""# Test cases for {problem['task_id']}
# Generated using Claude API

{problem['prompt']}
{problem['canonical_solution']}

# Generated test cases:
{test_cases}
"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_content)

        print(f"Test cases saved to {filepath}")

        # Also save usage stats alongside the test file
        stats_filepath = filepath.with_suffix(".stats.json")
        stats = self.get_usage_stats()
        stats["task_id"] = problem["task_id"]
        stats["generated_file"] = str(filepath)

        with open(stats_filepath, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        return str(filepath)

    def run_pytest(self, test_file_path: str) -> tuple[bool, str, float, float]:
        """Run pytest on the test file and return (success, error_output, c0_coverage, c1_coverage).

        Delegates to TestRunner for cleaner code.
        """
        result = self.test_runner.run_pytest(test_file_path)

        # Handle error messages
        output = result.output
        if result.error_message:
            output = f"Error: {result.error_message}"

        return result.success, output, result.c0_coverage, result.c1_coverage

    def generate_fix_prompt(
        self,
        original_code: str,
        error_output: str,
        attempt: int,
        problem: dict[str, Any],
    ) -> str:
        """Generate a prompt to fix test case errors with white box testing approach.

        Uses PromptBuilder for consistent prompt generation.
        """
        return self.prompt_builder.build_fix_prompt(
            problem=problem,
            test_code=original_code,
            error_output=error_output,
            attempt=attempt,
            max_attempts=self.get_total_fix_attempts(),
            include_docstring=self.include_docstring,
            include_ast=self.include_ast,
        )

    def fix_test_cases(
        self,
        test_code: str,
        error_output: str,
        attempt: int,
        problem: dict[str, Any],
        model: str,
    ) -> str:
        """Use LLM to fix test case errors.

        Uses the unified LLM client interface for cleaner code.
        """
        fix_prompt = self.generate_fix_prompt(test_code, error_output, attempt, problem)

        # Display the fix prompt if verbose mode is enabled
        should_proceed = self.display_fix_prompt(fix_prompt, attempt)
        if should_proceed is False:  # User chose to skip
            return test_code

        try:
            print(f"🤖 Sending fix request to LLM (attempt {attempt}) using {model}...")

            # Get the appropriate client for this model
            client = get_client_for_model(self.clients, model, self.config)

            # Generate response using unified interface
            response: LLMResponse = client.generate(
                prompt=fix_prompt,
                model=self.model_mapping[model],
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE,
            )

            # Track token usage and cost
            self.total_input_tokens += response.input_tokens
            self.total_output_tokens += response.output_tokens

            cost = self.calculate_cost(
                response.input_tokens, response.output_tokens, model
            )
            self.total_cost += cost

            if self.verbose_evaluation:
                print(
                    f"💰 Fix attempt {attempt} cost: ${cost:.6f} "
                    f"(Input: {response.input_tokens}, Output: {response.output_tokens})"
                )

            cleaned_response = self.clean_generated_code(response.content)

            # Display the LLM's fix response
            self.display_fix_response(cleaned_response, attempt)

            return cleaned_response

        except RuntimeError as e:
            print(f"❌ {e}")
            return test_code  # Return original code if fix fails
        except Exception as e:
            print(f"❌ Error fixing test cases: {e}")
            return test_code  # Return original code if fixing fails

    def _analyze_failure_type(self, error_output: str) -> str:
        """Analyze pytest output to determine the type of failure.

        Delegates to FailureAnalyzer for cleaner code.

        Args:
            error_output: The complete pytest output

        Returns:
            str: Type of failure ('assertion', 'syntax', 'import', 'runtime', 'timeout', 'unknown')
        """
        return self.failure_analyzer.analyze(error_output).value

    def evaluate_bug_detection(
        self, test_file_path: str, problem: dict[str, Any]
    ) -> tuple[bool, bool, str, float, float]:
        """Evaluate if generated tests can detect bugs in HumanEvalPack.

        Delegates to BugDetector for cleaner code.

        Returns:
            tuple[bool, bool, str, float, float]: (canonical_passed, buggy_failed_with_assertion, failure_type, final_c0_coverage, final_c1_coverage)
        """
        result = self.bug_detector.evaluate(test_file_path, problem)

        return (
            result.canonical_passed,
            result.true_bug_detected,
            result.failure_type.value,
            result.c0_coverage,
            result.c1_coverage,
        )

    def evaluate_and_fix_tests(
        self, test_file_path: str, problem: dict[str, Any], model: str
    ) -> tuple[bool, int, float, float]:
        """Evaluate test file with pytest and fix errors iteratively.

        Returns:
            tuple[bool, int, float, float]: (success, attempts_used, c0_coverage, c1_coverage)
        """
        if not self.enable_evaluation:
            return True, 0, 0.0, 0.0

        print(f"🧪 Evaluating test file: {Path(test_file_path).name}")

        final_c0_coverage = 0.0
        final_c1_coverage = 0.0
        max_pytest_runs = self.max_fix_attempts + 1  # Initial run + fix attempts

        for attempt in range(1, max_pytest_runs + 1):
            # Add clear header for each attempt
            if attempt > 1:
                print()  # Extra blank line for better readability
                print(f"\n{'='*80}")
                print(f"🔄 STARTING ATTEMPT {attempt} of {max_pytest_runs}")
                print(f"{'='*80}\n")

            # Run pytest
            success, error_output, c0_coverage, c1_coverage = self.run_pytest(
                test_file_path
            )
            final_c0_coverage = c0_coverage
            final_c1_coverage = c1_coverage

            if success:
                print(
                    f"✅ Tests passed on attempt {attempt} (C0 Coverage: {c0_coverage:.1f}%, C1 Coverage: {c1_coverage:.1f}%)"
                )
                return True, attempt - 1, final_c0_coverage, final_c1_coverage

            print(f"❌ Tests failed on attempt {attempt}")

            # Display detailed error information
            self.display_pytest_errors(error_output, attempt)

            if attempt <= self.max_fix_attempts:
                print(f"\n{'─'*80}")
                print(f"🔧 ATTEMPTING TO FIX ERRORS...")
                print(f"{'─'*80}\n")

                # Read current test file content
                with open(test_file_path, "r", encoding="utf-8") as f:
                    current_content = f.read()

                # Extract just the test code part (after generated test marker)
                test_code_start = current_content.find(GENERATED_TEST_MARKER)
                if test_code_start != -1:
                    test_code = current_content[
                        test_code_start + len(GENERATED_TEST_MARKER) :
                    ]
                else:
                    test_code = current_content

                # Get fixed version from LLM
                fixed_code = self.fix_test_cases(
                    test_code, error_output, attempt, problem, model
                )

                # Update the test file with fixed code
                base_content = (
                    current_content[:test_code_start]
                    if test_code_start != -1
                    else f"""# Test cases for {problem['task_id']}
# Generated using Claude API

{problem['prompt']}
{problem['canonical_solution']}

"""
                )

                updated_content = base_content + GENERATED_TEST_MARKER + fixed_code

                with open(test_file_path, "w", encoding="utf-8") as f:
                    f.write(updated_content)

                print(f"📝 Updated test file with fixes")
                print(f"\n{'─'*80}")
                print(
                    f"✅ Fix attempt {attempt} completed - Will retry pytest on next attempt"
                )
                print(f"{'─'*80}")
                print("\n")  # Extra blank lines for better readability
            else:
                total_fix_attempts = max(0, self.get_total_fix_attempts())
                print(f"\n{'='*80}")
                print(f"🚫 Maximum fix attempts ({total_fix_attempts}) reached")
                print(f"{'='*80}")
                print("Final error output:")
                print(error_output)

        return False, self.max_fix_attempts, final_c0_coverage, final_c1_coverage

    def _generate_and_evaluate_test_cases(
        self, problem: dict[str, Any], output_dir: str = "generated_tests"
    ) -> list[str]:
        """Generate test cases for a problem using all selected models, evaluate them, and return final filepaths."""
        print(f"Selected problem: {problem['task_id']}")
        print(
            f"Generating test cases using {len(self.models)} model(s): {', '.join(self.models)}"
        )

        final_filepaths = []
        model_results = {}

        for model in self.models:
            print(f"\n{'='*60}")
            print(f"Processing with model: {model}")
            print(f"{'='*60}")

            test_cases = self.generate_test_cases(problem, model)
            if not test_cases:
                print(f"❌ Failed to generate test cases for model {model}")
                model_results[model] = {"status": "generation_failed", "filepath": None}
                continue

            filepath = self.save_test_cases(problem, test_cases, output_dir, model)

            # Run evaluation and fix cycle if enabled
            evaluation_success = True
            fix_attempts_used = 0
            c0_coverage = 0.0
            c1_coverage = 0.0
            canonical_passed = None
            buggy_failed = None
            failure_type = None

            if self.enable_evaluation:
                evaluation_success, fix_attempts_used, c0_coverage, c1_coverage = (
                    self.evaluate_and_fix_tests(filepath, problem, model)
                )

                # For HumanEvalPack, run additional bug detection evaluation
                if self.dataset_type == "humanevalpack" and evaluation_success:
                    print(
                        f"\n📊 Coverage before bug detection: C0={c0_coverage:.1f}%, C1={c1_coverage:.1f}%"
                    )

                    (
                        canonical_passed,
                        buggy_failed,
                        failure_type,
                        bug_c0_cov,
                        bug_c1_cov,
                    ) = self.evaluate_bug_detection(filepath, problem)
                    # Update coverage with the final values from bug detection evaluation
                    # (which re-runs tests with canonical solution to get accurate coverage)
                    c0_coverage = bug_c0_cov
                    c1_coverage = bug_c1_cov

                    print(
                        f"📊 Coverage after bug detection: C0={c0_coverage:.1f}%, C1={c1_coverage:.1f}%"
                    )
                    print(
                        f"📊 Final coverage to be saved: C0={c0_coverage:.1f}%, C1={c1_coverage:.1f}%"
                    )

                    bug_detection_success = canonical_passed and buggy_failed

                    if bug_detection_success:
                        print(
                            f"🎉 Test generation completed with TRUE bug detection for {model}!"
                        )
                    elif failure_type != "none":
                        print(
                            f"⚠️  Test generation completed but detected {failure_type.upper()} error (not true bug) for {model}"
                        )
                    else:
                        print(
                            f"⚠️  Test generation completed but bug detection failed for {model}"
                        )

                if evaluation_success:
                    print(
                        f"🎉 Test generation and evaluation completed successfully for {model}!"
                    )
                    model_results[model] = {
                        "status": "success",
                        "filepath": filepath,
                        "coverage": c0_coverage,
                        "c0_coverage": c0_coverage,
                        "c1_coverage": c1_coverage,
                    }
                else:
                    print(
                        f"⚠️  Test generation completed but evaluation failed for {model}"
                    )
                    model_results[model] = {
                        "status": "evaluation_failed",
                        "filepath": filepath,
                        "coverage": c0_coverage,
                        "c0_coverage": c0_coverage,
                        "c1_coverage": c1_coverage,
                    }
            else:
                model_results[model] = {
                    "status": "generated_no_eval",
                    "filepath": filepath,
                }

            # Update final stats after complete process and get final filepath
            final_filepath = self.update_final_stats(
                filepath,
                problem,
                evaluation_success,
                fix_attempts_used,
                c0_coverage,
                c1_coverage,
                canonical_passed,
                buggy_failed,
                failure_type,
            )

            final_filepaths.append(final_filepath)

        # Print summary of model results
        self._print_model_summary(model_results)

        return final_filepaths

    def _print_model_summary(self, model_results: dict[str, dict[str, Any]]) -> None:
        """Print a summary of results for all models."""
        print(f"\n{'='*60}")
        print(f"MODEL PROCESSING SUMMARY")
        print(f"{'='*60}")

        successful_models = []
        failed_generation = []
        failed_evaluation = []
        no_eval_models = []

        for model, result in model_results.items():
            status = result["status"]
            if status == "success":
                coverage = result.get("coverage", 0)
                successful_models.append(f"{model} (Coverage: {coverage:.1f}%)")
            elif status == "generation_failed":
                failed_generation.append(model)
            elif status == "evaluation_failed":
                coverage = result.get("coverage", 0)
                failed_evaluation.append(f"{model} (Coverage: {coverage:.1f}%)")
            elif status == "generated_no_eval":
                no_eval_models.append(model)

        if successful_models:
            print(f"✅ Successfully completed ({len(successful_models)}):")
            for model in successful_models:
                print(f"   • {model}")

        if no_eval_models:
            print(f"📝 Generated without evaluation ({len(no_eval_models)}):")
            for model in no_eval_models:
                print(f"   • {model}")

        if failed_evaluation:
            print(f"⚠️  Generated but evaluation failed ({len(failed_evaluation)}):")
            for model in failed_evaluation:
                print(f"   • {model}")

        if failed_generation:
            print(f"❌ Failed to generate ({len(failed_generation)}):")
            for model in failed_generation:
                print(f"   • {model}")

        total_attempted = len(model_results)
        total_with_output = len([r for r in model_results.values() if r["filepath"]])
        print(
            f"\n📊 Overall: {total_with_output}/{total_attempted} models produced test files"
        )

    def generate_for_random_problem(
        self, output_dir: str = "generated_tests"
    ) -> list[str]:
        """Generate test cases for a randomly selected problem."""
        if not self.problems:
            raise ValueError("No problems loaded. Call load_dataset() first.")

        problem = self.select_random_problem()
        return self._generate_and_evaluate_test_cases(problem, output_dir)

    def generate_for_specific_problem(
        self, task_id: str, output_dir: str = "generated_tests"
    ) -> list[str]:
        """Generate test cases for a specific problem by task_id.

        Args:
            task_id: Problem identifier. Can be:
                     - Number only: "0", "5", "10" (automatically adds prefix)
                     - Full ID: "HumanEval/0", "Python/0" (used as-is)
            output_dir: Output directory for generated tests

        Returns:
            List of generated test file paths
        """
        # Normalize task_id based on dataset type
        normalized_task_id = self._normalize_task_id(task_id)

        problem = next(
            (p for p in self.problems if p["task_id"] == normalized_task_id), None
        )
        if not problem:
            raise ValueError(
                f"Problem {normalized_task_id} not found in {self.dataset_type} dataset. "
                f"Available range: 0-{len(self.problems)-1}"
            )

        return self._generate_and_evaluate_test_cases(problem, output_dir)

    def _normalize_task_id(self, task_id: str) -> str:
        """Normalize task_id based on dataset type.

        Args:
            task_id: Raw task ID (e.g., "0", "HumanEval/0", "Python/0")

        Returns:
            Normalized task ID with proper prefix
        """
        # If task_id is just a number, add appropriate prefix
        if task_id.isdigit():
            if self.dataset_type == "humanevalpack":
                return f"Python/{task_id}"
            else:
                return f"HumanEval/{task_id}"

        # If task_id already has a prefix, use it as-is
        return task_id


def main():
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate test cases for HumanEval problems using Claude API"
    )
    parser.add_argument(
        "--dataset",
        default="dataset/HumanEval.jsonl",
        help="Path to HumanEval dataset file",
    )
    parser.add_argument(
        "--output-dir",
        default="data/generated_tests",
        help="Output directory for test files",
    )
    parser.add_argument(
        "--task-id",
        help="Specific task ID to generate tests for (e.g., '0', '5' or 'HumanEval/0')",
    )
    parser.add_argument("--api-key", help="Claude API key (or set in .env file)")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[get_default_model()],
        choices=get_available_models(),
        help="Claude model(s) to use for test generation (can specify multiple)",
    )
    parser.add_argument(
        "--include-docstring",
        action="store_true",
        help="Include function docstring in prompt (default: only function signature)",
    )
    parser.add_argument(
        "--include-ast",
        action="store_true",
        help="Include AST of canonical solution in prompt",
    )
    parser.add_argument(
        "--show-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Display the prompt before sending to LLM and ask for confirmation "
            "(default: True; use --no-show-prompt to disable)"
        ),
    )
    parser.add_argument(
        "--disable-evaluation",
        action="store_true",
        help="Disable automatic evaluation and fixing of generated tests",
    )
    parser.add_argument(
        "--max-fix-attempts",
        type=int,
        default=3,
        help="Maximum number of LLM fix attempts (default: 3)",
    )
    parser.add_argument(
        "--quiet-evaluation",
        action="store_true",
        help="Disable verbose output during error fixing process",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["humaneval", "humanevalpack"],
        default="humanevalpack",
        help="Dataset to use for test generation (default: humanevalpack)",
    )

    args = parser.parse_args()

    # Get API key from argument, environment, or .env file
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")

    # Check if we need an API key (only for Anthropic models)
    try:
        with open("models_config.json", "r") as f:
            models_config = json.load(f)
    except FileNotFoundError:
        print("Error: models_config.json not found.")
        return 1
    except json.JSONDecodeError:
        print("Error: models_config.json contains invalid JSON.")
        return 1

    # args.models already has default value, so it's never None
    selected_models = args.models

    # Check if any selected model requires Anthropic API
    needs_anthropic = any(
        models_config["models"][model].get("provider", "anthropic") == "anthropic"
        for model in selected_models
    )

    # Check if any selected model requires OpenAI API
    needs_openai = any(
        models_config["models"][model].get("provider") == "openai"
        for model in selected_models
    )

    # Check if any selected model requires Gemini API
    needs_gemini = any(
        models_config["models"][model].get("provider") == "gemini"
        for model in selected_models
    )

    if needs_anthropic and not api_key:
        print("Error: Anthropic models selected but no API key provided.")
        print("Please provide Claude API key via:")
        print("  1. --api-key argument")
        print("  2. ANTHROPIC_API_KEY environment variable")
        print("  3. Set ANTHROPIC_API_KEY in .env file")
        return 1

    if needs_openai and not os.getenv("OPENAI_API_KEY"):
        print("Error: OpenAI models selected but no API key provided.")
        print("Please provide OpenAI API key via:")
        print("  1. OPENAI_API_KEY environment variable")
        print("  2. Set OPENAI_API_KEY in .env file")
        return 1

    if needs_gemini and not os.getenv("GEMINI_API_KEY"):
        print("Error: Gemini models selected but no API key provided.")
        print("Please provide Gemini API key via:")
        print("  1. GEMINI_API_KEY environment variable")
        print("  2. Set GEMINI_API_KEY in .env file")
        return 1

    try:
        # Initialize generator
        generator = TestCaseGenerator(
            api_key,
            models=args.models,
            include_docstring=args.include_docstring,
            include_ast=args.include_ast,
            show_prompt=args.show_prompt,
            enable_evaluation=not args.disable_evaluation,
            max_fix_attempts=args.max_fix_attempts,
            verbose_evaluation=not args.quiet_evaluation,
            dataset_type=args.dataset_type,
        )

        # Load dataset
        generator.load_dataset(args.dataset)

        # Generate test cases
        if args.task_id:
            output_files = generator.generate_for_specific_problem(
                args.task_id, args.output_dir
            )
        else:
            output_files = generator.generate_for_random_problem(args.output_dir)

        # Check if any files were generated
        if not output_files:
            print(f"\n❌ No test cases were generated successfully for any model.")
            return 1

        # Display final usage statistics
        stats = generator.get_usage_stats()

        if output_files:
            print(f"\n✅ Generated test files!")
            print(f"📁 Output folders and files ({len(output_files)}):")

            # Group files by folder
            folders = {}
            for output_file in output_files:
                folder = Path(output_file).parent.name
                filename = Path(output_file).name
                if folder not in folders:
                    folders[folder] = []
                folders[folder].append(filename)

            for i, (folder, files) in enumerate(folders.items(), 1):
                print(f"  {i}. {folder}/")
                for file in files:
                    print(f"     └── {file}")

            print(f"\n📊 Token Usage & Cost:")
            print(f"  Input tokens: {stats['total_input_tokens']}")
            print(f"  Output tokens: {stats['total_output_tokens']}")
            print(f"  Total tokens: {stats['total_tokens']}")
            print(f"  Total cost: ${stats['total_cost_usd']}")
            print(f"\nTo run the tests:")
            if len(folders) == 1:
                folder_name = list(folders.keys())[0]
                print(f"  cd {folder_name}")
                print(f"  pytest . -v --cov")
            else:
                print(f"  # Run tests from specific model folder:")
                for folder in folders.keys():
                    print(f"  cd {folder} && pytest . -v --cov")
                print(f"  # Or run all tests from parent directory:")
                print(f"  pytest generated_tests_*/ -v --cov")
        else:
            print(f"\n❌ No test files were generated successfully.")
            print(f"📊 Token Usage & Cost:")
            print(f"  Input tokens: {stats['total_input_tokens']}")
            print(f"  Output tokens: {stats['total_output_tokens']}")
            print(f"  Total tokens: {stats['total_tokens']}")
            print(f"  Total cost: ${stats['total_cost_usd']}")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
