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
from pathlib import Path
from typing import Any, Optional
import anthropic
import google.generativeai as genai
from dotenv import load_dotenv
from model_utils import get_available_models, get_default_model
from ollama_client import Ollama


# Constants for maintainability
MAX_AST_NODES = 20  # Maximum number of AST nodes to include in snippet
MAX_AST_OUTPUT_NODES = 15  # Maximum nodes to output in final result
AST_SCORE_ERROR_MATCH = 10  # Score for nodes matching error patterns
AST_SCORE_LINE_OVERLAP = 5  # Score for nodes overlapping with error lines
AST_SCORE_COMMON_ERROR = 2  # Score for common error-prone operations
PYTEST_TIMEOUT_SECONDS = 60  # Timeout for pytest execution
DEFAULT_MAX_TOKENS = 8000  # Max tokens for LLM response (high limit for research evaluation)
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
        max_pytest_runs: int = 3,
        verbose_evaluation: bool = True,
        config_path: str = "models_config.json",
        ast_fix: bool = False,
    ):
        """Initialize the test case generator with LLM clients."""
        self.api_key = api_key
        self.problems = []

        # Load model configuration from external file
        self.config = self._load_model_config(config_path)
        self.model_mapping = {
            model: config["api_name"] for model, config in self.config["models"].items()
        }

        # Initialize clients for different providers
        self.clients = self._initialize_clients()

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

        self.include_docstring = include_docstring
        self.include_ast = include_ast
        self.show_prompt = show_prompt
        self.enable_evaluation = enable_evaluation
        self.max_pytest_runs = max_pytest_runs
        self.verbose_evaluation = verbose_evaluation
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        # Whether to include a focused AST snippet in error-fix prompts
        self.ast_fix = ast_fix

    def _initialize_clients(self) -> dict[str, Any]:
        """Initialize clients for different LLM providers."""
        clients = {}

        # Check which providers are needed
        providers_needed = set()
        for model_config in self.config["models"].values():
            provider = model_config.get("provider", "anthropic")
            providers_needed.add(provider)

        # Initialize Anthropic client if needed
        if "anthropic" in providers_needed:
            if self.api_key:
                clients["anthropic"] = anthropic.Anthropic(api_key=self.api_key)
            else:
                # Will fail later if trying to use Anthropic models without API key
                clients["anthropic"] = None

        # Initialize Gemini client if needed
        if "gemini" in providers_needed:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
                clients["gemini"] = genai  # Store the module itself as the client
            else:
                # Will fail later if trying to use Gemini models without API key
                clients["gemini"] = None

        # Initialize Ollama client if needed
        if "ollama" in providers_needed:
            # Find Ollama configuration
            ollama_base_url = None
            for model_config in self.config["models"].values():
                if model_config.get("provider") == "ollama":
                    ollama_base_url = model_config.get(
                        "base_url", "http://localhost:11434"
                    )
                    break

            # Allow environment variable override for Ollama base URL
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", ollama_base_url)

            if ollama_base_url:
                clients["ollama"] = Ollama(base_url=ollama_base_url)

        return clients

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
        """Load the HumanEval dataset from JSONL file."""
        print(f"Loading dataset from {file_path}...")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    problem = json.loads(line.strip())
                    self.problems.append(problem)

        print(f"Loaded {len(self.problems)} problems from dataset")

    def select_random_problem(self) -> dict[str, Any]:
        """Randomly select a problem from the dataset."""
        return random.choice(self.problems)

    def extract_function_signature(self, prompt: str, entry_point: str) -> str:
        """Extract just the function signature without docstring."""
        lines = prompt.split("\n")
        signature_lines = []
        in_signature = False

        for line in lines:
            if line.strip().startswith(f"def {entry_point}("):
                in_signature = True
                signature_lines.append(line)
            elif in_signature:
                if line.strip().startswith('"""') or line.strip().startswith("'''"):
                    # Stop at docstring
                    break
                elif (
                    line.strip()
                    and not line.startswith(" ")
                    and not line.startswith("\t")
                ):
                    # Stop at next top-level statement
                    break
                else:
                    signature_lines.append(line)
                    if line.strip().endswith(":"):
                        # End of function signature
                        break

        return "\n".join(signature_lines)

    def generate_ast_string(
        self, canonical_solution: str, prompt: str, entry_point: str
    ) -> str:
        """Generate a readable AST representation of the canonical solution."""
        try:
            # Extract function signature from the prompt more robustly
            # Find the def line and continue until we hit the docstring or end of signature
            lines = prompt.split("\n")
            signature_lines = []
            signature_started = False

            for line in lines:
                if line.strip().startswith(f"def {entry_point}("):
                    signature_started = True
                    signature_lines.append(line.strip())
                elif signature_started:
                    if line.strip().startswith('"""') or line.strip().startswith("'''"):
                        # Hit docstring, stop
                        break
                    elif line.strip().endswith(":") or line.strip() == "":
                        # End of signature or empty line
                        if line.strip().endswith(":"):
                            signature_lines.append(line.strip())
                        break
                    else:
                        signature_lines.append(line.strip())

            if not signature_lines:
                return "Error: Could not extract function signature"

            signature = " ".join(signature_lines)
            if not signature.endswith(":"):
                signature += ":"

            # Create complete function code
            full_function = f"{signature}\n{canonical_solution}"

            # Parse the AST
            tree = ast.parse(full_function)

            # Format the AST as a readable string
            return ast.dump(tree, indent=2)
        except Exception as e:
            return f"Error generating AST: {e}"

    # --- Helper methods to keep generate_relevant_ast_snippet maintainable ---
    def _normalize_line(self, s: str) -> str:
        return re.sub(r"\s+", " ", s.strip())

    def _build_normalized_index(self, full_lines: list[str]) -> dict[str, list[int]]:
        index: dict[str, list[int]] = {}
        for idx, fl in enumerate(full_lines, start=1):
            nf = self._normalize_line(fl)
            if not nf:
                continue
            index.setdefault(nf, []).append(idx)
        return index

    def _parse_error_output_for_lines(
        self,
        error_output: str,
        full_lines: list[str],
        normalized_index: dict[str, list[int]],
    ) -> set[int]:
        candidate_set: set[int] = set()
        for raw in error_output.split("\n"):
            ln = raw.strip()
            if not ln:
                continue
            if ln.startswith("STDOUT:") or ln.startswith("STDERR:"):
                continue
            if (
                ln.startswith("=== ")
                or ln.startswith("FAILED ")
                or ln.startswith("PASSED ")
                or ln.startswith("Traceback")
            ):
                continue

            # explicit line numbers, e.g., "line 12"
            for m in re.finditer(r"\bline\s+(\d+)\b", ln):
                try:
                    num = int(m.group(1))
                    if 1 <= num <= len(full_lines):
                        candidate_set.add(num)
                except ValueError:
                    pass

            # pytest excerpt line starting with '>'
            if raw.lstrip().startswith(">"):
                excerpt = raw.lstrip().lstrip(">")
                nf = self._normalize_line(excerpt)
                if nf in normalized_index:
                    candidate_set.update(normalized_index[nf])
                continue

            # general normalized content match
            nf = self._normalize_line(ln)
            if nf and len(nf.split()) >= 2 and nf in normalized_index:
                candidate_set.update(normalized_index[nf])

            # lines prefixed with 'E '
            if ln.startswith("E "):
                ln_after = ln[1:].strip()
                if ln_after:
                    nf2 = self._normalize_line(ln_after)
                    if nf2 and len(nf2.split()) >= 2 and nf2 in normalized_index:
                        candidate_set.update(normalized_index[nf2])

        return candidate_set

    def _expand_candidate_lines(
        self,
        tree: ast.AST,
        full_lines: list[str],
        candidate_set: set[int],
        node_span_cap: int = 20,
    ) -> list[int]:
        expanded: set[int] = set(candidate_set)
        for cl in list(candidate_set):
            if cl - 1 >= 1:
                expanded.add(cl - 1)
            if cl + 1 <= len(full_lines):
                expanded.add(cl + 1)

        for node in ast.walk(tree):
            lineno = getattr(node, "lineno", None)
            end_lineno = getattr(node, "end_lineno", lineno)
            if lineno is None or end_lineno is None:
                continue
            overlaps_any = any(lineno <= cl <= end_lineno for cl in candidate_set)
            if overlaps_any and (end_lineno - lineno + 1) <= node_span_cap:
                for ln_i in range(lineno, end_lineno + 1):
                    if 1 <= ln_i <= len(full_lines):
                        expanded.add(ln_i)
        return sorted(expanded)

    def generate_relevant_ast_snippet(
        self, problem: dict[str, Any], error_output: str
    ) -> str:
        """Generate a compact AST snippet focusing on nodes related to the error.

        Heuristics:
        - Match error output lines to lines in the canonical implementation and include
          nodes whose line ranges overlap.
        - Include nodes associated with common Python error types found in the output.
        """
        try:
            entry_point = problem.get("entry_point", "")
            # Build function text (signature + body)
            signature = self.extract_function_signature(
                problem.get("prompt", ""), entry_point
            )
            if not signature:
                # Fallback minimal signature
                if entry_point and entry_point.strip():
                    signature = f"def {entry_point}(*args, **kwargs):"
                else:
                    signature = "def _func(*args, **kwargs):"
            if not signature.endswith(":"):
                signature += ":"
            full_function = f"{signature}\n{problem.get('canonical_solution', '')}"

            tree = ast.parse(full_function)
            full_lines = full_function.splitlines()

            # Candidate lines from error output
            normalized_to_indices = self._build_normalized_index(full_lines)
            candidate_set = self._parse_error_output_for_lines(
                error_output, full_lines, normalized_to_indices
            )
            candidate_lines: list[int] = self._expand_candidate_lines(
                tree, full_lines, candidate_set
            )

            # Error keyword → node predicate mapping
            # Use more specific error patterns for better accuracy
            predicates = []
            low = error_output.lower()

            # Division errors
            if (
                "zerodivisionerror" in low
                or "division by zero" in low
                or "divide by zero" in low
            ):
                predicates.append(
                    lambda n: isinstance(n, ast.BinOp)
                    and isinstance(n.op, (ast.Div, ast.Mod, ast.FloorDiv))
                )

            # Index errors
            if (
                "indexerror" in low
                or "list index out of range" in low
                or "string index out of range" in low
                or "tuple index out of range" in low
            ):
                predicates.append(lambda n: isinstance(n, ast.Subscript))

            # Key errors (dict/mapping access)
            if "keyerror" in low:
                predicates.append(lambda n: isinstance(n, ast.Subscript))

            # Attribute errors
            if "attributeerror" in low or "has no attribute" in low:
                predicates.append(lambda n: isinstance(n, ast.Attribute))

            # Type errors with operators
            if "typeerror" in low and (
                "operand" in low or "not supported between" in low
            ):
                predicates.append(lambda n: isinstance(n, ast.BinOp))

            # Type errors with calls
            if "typeerror" in low and (
                "takes" in low or "required" in low or "argument" in low
            ):
                predicates.append(lambda n: isinstance(n, ast.Call))

            # Value errors
            if "valueerror" in low:
                predicates.append(
                    lambda n: isinstance(n, ast.Raise) or isinstance(n, ast.Call)
                )

            # Recursion errors
            if "recursionerror" in low and entry_point:
                predicates.append(
                    lambda n: isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id == entry_point
                )

            # Name errors
            if "nameerror" in low or "is not defined" in low:
                predicates.append(lambda n: isinstance(n, ast.Name))

            # Import errors
            if "importerror" in low or "modulenotfounderror" in low:
                predicates.append(lambda n: isinstance(n, (ast.Import, ast.ImportFrom)))

            # Collect nodes with priority scoring
            node_scores: list[tuple[ast.AST, int]] = []

            for node in ast.walk(tree):
                lineno = getattr(node, "lineno", None)
                end_lineno = getattr(node, "end_lineno", lineno)

                # Check if node overlaps with error lines
                overlaps = False
                if lineno is not None:
                    for cl in candidate_lines:
                        if lineno <= cl <= (end_lineno or lineno):
                            overlaps = True
                            break

                # Check if node matches error-specific predicates
                matches = (
                    any(pred(node) for pred in predicates) if predicates else False
                )

                # Define interesting node types
                interesting = isinstance(
                    node,
                    (
                        ast.If,
                        ast.For,
                        ast.While,
                        ast.With,
                        ast.Try,
                        ast.Assign,
                        ast.AugAssign,
                        ast.Return,
                        ast.Raise,
                        ast.Call,
                        ast.BinOp,
                        ast.BoolOp,
                        ast.Compare,
                        ast.Subscript,
                        ast.Attribute,
                        ast.ListComp,
                        ast.DictComp,
                        ast.SetComp,
                        ast.GeneratorExp,
                        ast.Lambda,
                        ast.Name,
                    ),
                )

                if interesting:
                    # Calculate priority score
                    score = 0
                    if matches:
                        score += AST_SCORE_ERROR_MATCH  # High priority for error-specific matches
                    if overlaps:
                        score += (
                            AST_SCORE_LINE_OVERLAP  # Medium priority for line overlaps
                        )

                    # Additional scoring based on node type relevance
                    if isinstance(
                        node, (ast.Call, ast.BinOp, ast.Subscript, ast.Attribute)
                    ):
                        score += AST_SCORE_COMMON_ERROR  # Common error sources

                    if score > 0:
                        node_scores.append((node, score))

            # Sort by score (highest first) and select top nodes
            node_scores.sort(key=lambda x: x[1], reverse=True)
            selected = [
                node for node, _ in node_scores[:MAX_AST_NODES]
            ]  # Limit to most relevant nodes

            if not selected:
                # Fallback: first few body statements for some structure
                func_def = next(
                    (n for n in tree.body if isinstance(n, ast.FunctionDef)), None
                )
                if func_def and func_def.body:
                    selected.extend(func_def.body[:3])

            # Generate concise AST representations
            parts: list[str] = []
            seen_nodes = set()  # Avoid duplicate nodes

            for n in selected:
                # Create a unique identifier for the node to avoid duplicates
                node_id = (
                    type(n).__name__,
                    getattr(n, "lineno", None),
                    getattr(n, "col_offset", None),
                )
                if node_id in seen_nodes:
                    continue
                seen_nodes.add(node_id)

                try:
                    # Create a more concise representation
                    node_info = f"Line {getattr(n, 'lineno', '?')}: {type(n).__name__}"

                    # Add relevant details based on node type
                    if isinstance(n, ast.BinOp):
                        node_info += f" ({type(n.op).__name__})"
                    elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                        node_info += f" ({n.func.id})"
                    elif isinstance(n, ast.Attribute):
                        node_info += f" (.{n.attr})"
                    elif isinstance(n, ast.Name):
                        node_info += f" ({n.id})"

                    # Add the full AST dump
                    parts.append(f"{node_info}\n{ast.dump(n, indent=2)}")
                except Exception:
                    parts.append(
                        f"Line {getattr(n, 'lineno', '?')}: {type(n).__name__}"
                    )

            return (
                "\n\n".join(parts[:MAX_AST_OUTPUT_NODES])
                if parts
                else "(no relevant AST nodes found)"
            )
        except Exception as e:
            return f"Error generating relevant AST snippet: {e}"

    def generate_prompt(self, problem: dict[str, Any]) -> str:
        """Create a prompt for Claude to generate test cases."""

        # Optionally include function signature/docstring section.
        # If --include-docstring is NOT set, omit this section entirely to avoid redundancy
        # since the canonical implementation below already includes the signature.
        signature_section = ""
        if self.include_docstring:
            # Include full function definition (signature + docstring)
            function_info = problem["prompt"]
            signature_section = f"\nSignature and Docstring:\n{function_info}\n\n"

        ast_section = ""
        if self.include_ast:
            ast_repr = self.generate_ast_string(
                problem["canonical_solution"], problem["prompt"], problem["entry_point"]
            )
            ast_section = f"""

AST representation of canonical solution:
```
{ast_repr}
```"""

        prompt = f"""Generate pytest test cases for this function. Return ONLY executable Python code, no explanations or markdown.

{signature_section}Canonical implementation:
```python
{self.extract_function_signature(problem['prompt'], problem['entry_point'])}
{problem['canonical_solution']}
```{ast_section}

Requirements:
- Return ONLY Python code that can be executed directly
- Include comprehensive test cases covering edge cases, normal cases, and error conditions
- Use pytest format
- Include necessary imports
- DO NOT include explanations, markdown, or the original function implementation
- DO NOT wrap code in ```python``` blocks
- IMPORTANT: When using @pytest.mark.parametrize, properly escape quotes in parameter values
- Use single quotes inside double quotes or vice versa, or use triple quotes for complex strings
- Example: @pytest.mark.parametrize("input,expected", [("test", True), ('another', False)])

Start your response with "import pytest" and include only executable Python test code:"""

        return prompt

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
        estimated_cost = (estimated_tokens / 1000) * self.config["models"][first_model][
            "pricing"
        ]["input_per_1k"]

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
        input_cost = (input_tokens / 1000) * pricing["input_per_1k"]
        output_cost = (output_tokens / 1000) * pricing["output_per_1k"]
        return input_cost + output_cost

    def get_total_fix_attempts(self) -> int:
        """Get the total number of fix attempts available."""
        return max(1, self.max_pytest_runs - 1)

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
        if stem.endswith("_success") or stem.endswith("_false"):
            stem = "_".join(stem.split("_")[:-1])

        # Add the new suffix based on evaluation result
        suffix = "success" if evaluation_success else "false"
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
                "max_pytest_runs": self.max_pytest_runs,
                "code_coverage_percent": c0_coverage,
                "code_coverage_c0_percent": c0_coverage,
                "code_coverage_c1_percent": c1_coverage,
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
            print(f"\n... ({len(lines) - DISPLAY_LINE_LIMIT} lines omitted) ...\n")
            print("\n".join(lines[-TRUNCATE_TAIL_LINES:]))

        print(f"{'='*80}")
        print(f"🔧 END OF FIX RESPONSE - Fix attempt {attempt} of {total_fix_attempts}")
        print(f"{'='*80}")
        print(f"\n{'─'*80}")
        print(f"✅ Fix response received for attempt {attempt}")
        print(f"{'─'*80}\n")

    def generate_test_cases(self, problem: dict[str, Any], model: str) -> str:
        """Generate test cases using LLM API."""
        prompt = self.generate_prompt(problem)

        print(f"Generating test cases for {problem['task_id']} using {model}...")

        # Show prompt and get confirmation if requested
        if self.show_prompt:
            if not self.display_prompt_and_confirm(prompt):
                return ""

        try:
            # Get the provider for this model
            provider = self.config["models"][model].get("provider", "anthropic")
            client = self.clients.get(provider)

            if not client:
                if provider == "anthropic":
                    raise ValueError(f"Anthropic API key required for model {model}")
                elif provider == "gemini":
                    raise ValueError(f"Gemini API key required for model {model}")
                else:
                    raise ValueError(f"Client not initialized for provider {provider}")

            # Handle different provider APIs
            if provider == "gemini":
                # Gemini API call
                gemini_model = client.GenerativeModel(self.model_mapping[model])
                response = gemini_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=DEFAULT_MAX_TOKENS,
                        temperature=DEFAULT_TEMPERATURE,
                    ),
                )

                # Check if response was blocked or has no candidates
                if not response.candidates:
                    if response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason'):
                        print(f"❌ Gemini API blocked content: {response.prompt_feedback.block_reason}")
                    else:
                        print("❌ Gemini API returned no candidates.")
                    return ""

                # Track token usage and cost
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count

                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens

                cost = self.calculate_cost(input_tokens, output_tokens, model)
                self.total_cost += cost

                raw_response = response.text
                return self.clean_generated_code(raw_response)

            else:
                # Anthropic/Ollama API call (existing code)
                response = client.messages.create(
                    model=self.model_mapping[model],
                    max_tokens=DEFAULT_MAX_TOKENS,
                    temperature=DEFAULT_TEMPERATURE,  # A temperature of 0.0 results in the most deterministic and consistent responses, as the model will consistently choose the most probable words and sequences.
                    messages=[{"role": "user", "content": prompt}],
                )

                # Track token usage and cost
                usage = response.usage
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens

                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens

                cost = self.calculate_cost(input_tokens, output_tokens, model)
                self.total_cost += cost

                raw_response = response.content[0].text
                return self.clean_generated_code(raw_response)

        except Exception as e:
            print(f"❌ Error generating test cases with {model}: {e}")
            return ""

    def save_test_cases(
        self, problem: dict[str, Any], test_cases: str, output_dir: str, model: str
    ) -> str:
        """Save generated test cases to a file."""
        # Create model-specific output directory
        model_folder = self.get_model_folder_name(model)
        model_output_dir = f"{output_dir}_{model_folder}"
        output_path = Path(model_output_dir)
        output_path.mkdir(exist_ok=True)

        # Create filename from task_id (no model name in filename since folder identifies model)
        base_name = f"test_{problem['task_id'].replace('/', '_').lower()}"
        filename_parts = [base_name]

        if self.include_docstring:
            filename_parts.append("docstring")
        if self.include_ast:
            filename_parts.append("ast")
        if getattr(self, "ast_fix", False):
            filename_parts.append("ast-fix")

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
        """Run pytest on the test file and return (success, error_output, c0_coverage, c1_coverage)."""

        # Use absolute path and run from project root
        abs_path = Path(test_file_path).resolve()
        cmd = ["pytest", str(abs_path), "--cov", "--cov-branch", "-v"]

        try:
            # Run pytest and capture output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=PYTEST_TIMEOUT_SECONDS,
                cwd=Path.cwd(),  # Run from current working directory
            )

            # Check if tests passed (return code 0)
            success = result.returncode == 0

            # Combine stdout and stderr for complete error information
            output = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

            # Extract coverage percentages from pytest output
            c0_coverage = 0.0
            c1_coverage = 0.0
            if result.stdout:
                # Look for statement coverage (C0) in format like "TOTAL    100   50   10   5   95%"
                # Format: Name Stmts Miss Branch BrPart Cover
                coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)%", result.stdout)
                if coverage_match:
                    c0_coverage = float(coverage_match.group(1))
                else:
                    # Fallback to old format without branch coverage "TOTAL    100   50   95%"
                    coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", result.stdout)
                    if coverage_match:
                        c0_coverage = float(coverage_match.group(1))
                
                # Extract branch coverage (C1) from the same line
                # With --cov-branch, the output includes branch coverage separately
                # We need to calculate it from Branch and BrPart columns
                branch_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)\s+(\d+)\s+\d+%", result.stdout)
                if branch_match:
                    total_branches = int(branch_match.group(1))
                    partial_branches = int(branch_match.group(2))
                    if total_branches > 0:
                        # Branch coverage = (covered branches / total branches) * 100
                        # Covered branches = total branches - partial branches
                        c1_coverage = ((total_branches - partial_branches) / total_branches) * 100
                    else:
                        c1_coverage = 100.0  # No branches means 100% coverage
                else:
                    # If no branches exist, set C1 to 100%
                    c1_coverage = 100.0

            return success, output, c0_coverage, c1_coverage

        except subprocess.TimeoutExpired:
            return (
                False,
                f"Error: pytest execution timed out after {PYTEST_TIMEOUT_SECONDS} seconds",
                0.0,
                0.0,
            )
        except Exception as e:
            return False, f"Error running pytest: {str(e)}", 0.0, 0.0

    def generate_fix_prompt(
        self,
        original_code: str,
        error_output: str,
        attempt: int,
        problem: dict[str, Any],
        ast_snippet: Optional[str] = None,
    ) -> str:
        """Generate a prompt to fix test case errors with white box testing approach."""
        ast_section = ""
        if self.include_ast:
            # Include full AST representation like in initial prompt
            ast_repr = self.generate_ast_string(
                problem["canonical_solution"], problem["prompt"], problem["entry_point"]
            )
            ast_section = (
                f"\n\nAST representation of canonical solution:\n```\n{ast_repr}\n```\n"
            )
        elif self.ast_fix and ast_snippet:
            # Include relevant AST snippet if ast_fix is enabled but not full AST
            ast_section = f"\n\nRELEVANT AST SNIPPET OF FUNCTION (focus on error):\n```\n{ast_snippet}\n```\n"

        # Distinguish between pytest attempts and fix attempts for clarity in the prompt
        # A fix prompt is only shown when attempt < self.max_pytest_runs, so fix attempt index == attempt
        total_fix_attempts = self.get_total_fix_attempts()
        fix_attempt_line = f"This is fix attempt {attempt} of {total_fix_attempts}."

        # Get function info based on include_docstring flag
        if self.include_docstring:
            function_info = problem.get("prompt", "")
        else:
            entry_point = problem.get("entry_point")
            if entry_point and entry_point.strip():
                function_info = self.extract_function_signature(
                    problem.get("prompt", ""), entry_point
                )
            else:
                # Fallback gracefully when entry_point is missing or empty in problem dict
                function_info = problem.get("prompt", "")

        return f"""The following test code has errors when running pytest. Please fix the issues and return ONLY the corrected Python code, no explanations or markdown.

FUNCTION BEING TESTED:
```python
{function_info}
{problem['canonical_solution']}
```
{ast_section}

CURRENT TEST CODE WITH ERRORS:
```python
{original_code}
```

PYTEST ERROR OUTPUT:
```
{error_output}
```

 {fix_attempt_line}

Requirements:
- Return ONLY executable Python code that can be run directly
- Fix all syntax errors, import errors, and test failures
- Use the provided function implementation to understand expected behavior
- Ensure tests properly validate the function's actual behavior
- Maintain comprehensive test coverage
- DO NOT include explanations, markdown, or code blocks
- DO NOT wrap code in ```python``` blocks
- DO NOT include the function implementation in your response (it's already in the file)
- Start your response with the corrected imports

Corrected code:"""

    def fix_test_cases(
        self,
        test_code: str,
        error_output: str,
        attempt: int,
        problem: dict[str, Any],
        model: str,
    ) -> str:
        """Use LLM to fix test case errors."""
        ast_snippet: Optional[str] = None
        if self.ast_fix:
            ast_snippet = self.generate_relevant_ast_snippet(problem, error_output)

        fix_prompt = self.generate_fix_prompt(
            test_code, error_output, attempt, problem, ast_snippet
        )

        # Display the fix prompt if verbose mode is enabled
        should_proceed = self.display_fix_prompt(fix_prompt, attempt)
        if should_proceed is False:  # User chose to skip
            return test_code

        try:
            print(f"🤖 Sending fix request to LLM (attempt {attempt}) using {model}...")

            # Get the provider for this model
            provider = self.config["models"][model].get("provider", "anthropic")
            client = self.clients.get(provider)

            if not client:
                if provider == "anthropic":
                    raise ValueError(f"Anthropic API key required for model {model}")
                elif provider == "gemini":
                    raise ValueError(f"Gemini API key required for model {model}")
                else:
                    raise ValueError(f"Client not initialized for provider {provider}")

            # Handle different provider APIs
            if provider == "gemini":
                # Gemini API call
                gemini_model = client.GenerativeModel(self.model_mapping[model])
                response = gemini_model.generate_content(
                    fix_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=DEFAULT_MAX_TOKENS,
                        temperature=DEFAULT_TEMPERATURE,
                    ),
                )

                # Check if response was blocked or has no candidates
                if not response.candidates:
                    if response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason'):
                        print(f"❌ Gemini API blocked content: {response.prompt_feedback.block_reason}")
                    else:
                        print("❌ Gemini API returned no candidates.")
                    return test_code  # Return original code if fix fails

                # Track token usage
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens

                cost = self.calculate_cost(input_tokens, output_tokens, model)
                self.total_cost += cost

                if self.verbose_evaluation:
                    print(
                        f"💰 Fix attempt {attempt} cost: ${cost:.6f} (Input: {input_tokens}, Output: {output_tokens})"
                    )

                raw_response = response.text
                cleaned_response = self.clean_generated_code(raw_response)

            else:
                # Anthropic/Ollama API call (existing code)
                response = client.messages.create(
                    model=self.model_mapping[model],
                    max_tokens=DEFAULT_MAX_TOKENS,
                    temperature=DEFAULT_TEMPERATURE,
                    messages=[{"role": "user", "content": fix_prompt}],
                )

                # Track token usage
                usage = response.usage
                self.total_input_tokens += usage.input_tokens
                self.total_output_tokens += usage.output_tokens

                cost = self.calculate_cost(usage.input_tokens, usage.output_tokens, model)
                self.total_cost += cost

                if self.verbose_evaluation:
                    print(
                        f"💰 Fix attempt {attempt} cost: ${cost:.6f} (Input: {usage.input_tokens}, Output: {usage.output_tokens})"
                    )

                raw_response = response.content[0].text
                cleaned_response = self.clean_generated_code(raw_response)

            # Display the LLM's fix response
            self.display_fix_response(cleaned_response, attempt)

            return cleaned_response

        except Exception as e:
            print(f"❌ Error fixing test cases: {e}")
            return test_code  # Return original code if fixing fails

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

        for attempt in range(1, self.max_pytest_runs + 1):
            # Add clear header for each attempt
            if attempt > 1:
                print()  # Extra blank line for better readability
                print(f"\n{'='*80}")
                print(f"🔄 STARTING ATTEMPT {attempt} of {self.max_pytest_runs}")
                print(f"{'='*80}\n")
            
            # Run pytest
            success, error_output, c0_coverage, c1_coverage = self.run_pytest(test_file_path)
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

            if attempt < self.max_pytest_runs:
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
                print(f"✅ Fix attempt {attempt} completed - Will retry pytest on next attempt")
                print(f"{'─'*80}")
                print()  # Extra blank line for better readability
                print()  # Extra blank line for better readability
            else:
                total_fix_attempts = max(0, self.get_total_fix_attempts())
                print(f"\n{'='*80}")
                print(f"🚫 Maximum fix attempts ({total_fix_attempts}) reached")
                print(f"{'='*80}")
                print("Final error output:")
                print(error_output)

        return False, self.max_pytest_runs - 1, final_c0_coverage, final_c1_coverage

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

            if self.enable_evaluation:
                evaluation_success, fix_attempts_used, c0_coverage, c1_coverage = (
                    self.evaluate_and_fix_tests(filepath, problem, model)
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
                filepath, problem, evaluation_success, fix_attempts_used, c0_coverage, c1_coverage
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
        """Generate test cases for a specific problem by task_id."""
        problem = next((p for p in self.problems if p["task_id"] == task_id), None)
        if not problem:
            raise ValueError(f"Problem {task_id} not found in dataset")

        return self._generate_and_evaluate_test_cases(problem, output_dir)


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
        "--task-id", help="Specific task ID to generate tests for (optional)"
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
        "--max-pytest-runs",
        type=int,
        default=3,
        help="Maximum number of pytest runs (initial + fixes) (default: 3)",
    )
    parser.add_argument(
        "--quiet-evaluation",
        action="store_true",
        help="Disable verbose output during error fixing process",
    )
    parser.add_argument(
        "--ast-fix",
        action="store_true",
        help="Enable AST-focused error fixing (adds relevant AST snippet to LLM fix prompts)",
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

    selected_models = args.models if args.models else [models_config["default_model"]]

    # Check if any selected model requires Anthropic API
    needs_anthropic = any(
        models_config["models"][model].get("provider", "anthropic") == "anthropic"
        for model in selected_models
    )

    if needs_anthropic and not api_key:
        print("Error: Anthropic models selected but no API key provided.")
        print("Please provide Claude API key via:")
        print("  1. --api-key argument")
        print("  2. ANTHROPIC_API_KEY environment variable")
        print("  3. Set ANTHROPIC_API_KEY in .env file")
        print("\nOr use Ollama models which don't require an API key.")
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
            max_pytest_runs=args.max_pytest_runs,
            verbose_evaluation=not args.quiet_evaluation,
            ast_fix=args.ast_fix,
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
