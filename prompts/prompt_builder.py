#!/usr/bin/env python3
"""
Prompt Builder Module

Handles dynamic loading and rendering of prompt templates for test case generation.
This module ensures research reproducibility by using external template files
instead of hardcoded prompts.
"""

import ast
from pathlib import Path
from typing import Any, Optional


class PromptBuilder:
    """Builds prompts from template files for test case generation."""

    # Template file mapping based on configuration flags
    TEMPLATE_MAPPING = {
        (False, False): "basic.txt",  # No docstring, no AST
        (True, False): "docstring.txt",  # With docstring, no AST
        (False, True): "ast.txt",  # No docstring, with AST
        (True, True): "docstring_ast.txt",  # With docstring, with AST
    }

    # Fix template mapping for error correction prompts
    FIX_TEMPLATE_MAPPING = {
        (False, False): "fix_basic.txt",  # No docstring, no AST
        (True, False): "fix_docstring.txt",  # With docstring, no AST
        (False, True): "fix_ast.txt",  # No docstring, with AST
        (True, True): "fix_docstring_ast.txt",  # With docstring, with AST
    }

    def __init__(self, prompts_dir: str = None):
        """
        Initialize the prompt builder.

        Args:
            prompts_dir: Path to the directory containing prompt templates.
                        Defaults to 'prompts/' relative to the project root.
        """
        if prompts_dir is None:
            # Default to prompts/ directory relative to this file's parent
            self.prompts_dir = Path(__file__).parent
        else:
            self.prompts_dir = Path(prompts_dir)

        self._validate_templates()

    def _validate_templates(self) -> None:
        """Validate that all required template files exist."""
        missing_templates = []
        
        # Check generation templates
        for template_file in self.TEMPLATE_MAPPING.values():
            template_path = self.prompts_dir / template_file
            if not template_path.exists():
                missing_templates.append(template_file)
        
        # Check fix templates
        for template_file in self.FIX_TEMPLATE_MAPPING.values():
            template_path = self.prompts_dir / template_file
            if not template_path.exists():
                missing_templates.append(template_file)

        if missing_templates:
            raise FileNotFoundError(
                f"Missing prompt template files in {self.prompts_dir}: {missing_templates}"
            )

    def get_template_name(
        self, include_docstring: bool = False, include_ast: bool = False
    ) -> str:
        """
        Get the appropriate template filename based on configuration.

        Args:
            include_docstring: Whether to include function docstring in prompt
            include_ast: Whether to include AST representation in prompt

        Returns:
            Template filename
        """
        return self.TEMPLATE_MAPPING[(include_docstring, include_ast)]

    def load_template(self, template_name: str) -> str:
        """
        Load a template file content.

        Args:
            template_name: Name of the template file

        Returns:
            Template content as string
        """
        template_path = self.prompts_dir / template_name
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()

    def extract_function_signature(self, prompt: str, entry_point: str) -> str:
        """
        Extract just the function signature without docstring.

        Args:
            prompt: The full prompt containing the function definition
            entry_point: The function name to extract

        Returns:
            Function signature as string
        """
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
        """
        Generate a readable AST representation of the canonical solution.

        Args:
            canonical_solution: The reference implementation body
            prompt: The full prompt containing the function definition
            entry_point: The function name

        Returns:
            AST dump as string, or error message if parsing fails
        """
        try:
            # Extract function signature from the prompt more robustly
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

    def build_prompt(
        self,
        problem: dict[str, Any],
        include_docstring: bool = False,
        include_ast: bool = False,
    ) -> str:
        """
        Build a complete prompt from template and problem data.

        Args:
            problem: Dictionary containing problem data with keys:
                    - prompt: The function signature with docstring
                    - canonical_solution: The reference implementation
                    - entry_point: The function name
            include_docstring: Whether to include function docstring
            include_ast: Whether to include AST representation

        Returns:
            Rendered prompt string ready for LLM
        """
        # Get the appropriate template
        template_name = self.get_template_name(include_docstring, include_ast)
        template = self.load_template(template_name)

        # Extract function signature
        function_signature = self.extract_function_signature(
            problem["prompt"], problem["entry_point"]
        )

        # Prepare template variables
        template_vars = {
            "function_signature": function_signature,
            "canonical_solution": problem["canonical_solution"],
        }

        # Add docstring if needed
        if include_docstring:
            template_vars["function_with_docstring"] = problem["prompt"]

        # Add AST if needed
        if include_ast:
            ast_repr = self.generate_ast_string(
                problem["canonical_solution"],
                problem["prompt"],
                problem["entry_point"],
            )
            template_vars["ast_representation"] = ast_repr

        # Render template
        try:
            return template.format(**template_vars)
        except KeyError as e:
            raise ValueError(
                f"Template '{template_name}' requires variable {e} "
                f"but it was not provided"
            )

    def build_fix_prompt(
        self,
        problem: dict[str, Any],
        test_code: str,
        error_output: str,
        attempt: int,
        max_attempts: int,
        include_docstring: bool = False,
        include_ast: bool = False,
    ) -> str:
        """
        Build a prompt for fixing test case errors from template.

        Args:
            problem: Dictionary containing problem data
            test_code: Current test code with errors
            error_output: Pytest error output
            attempt: Current fix attempt number
            max_attempts: Maximum number of fix attempts
            include_docstring: Whether to include function docstring
            include_ast: Whether to include full AST

        Returns:
            Fix prompt string ready for LLM
        """
        # Get function info based on include_docstring flag
        if include_docstring:
            function_info = problem.get("prompt", "")
        else:
            entry_point = problem.get("entry_point")
            if entry_point and entry_point.strip():
                function_info = self.extract_function_signature(
                    problem.get("prompt", ""), entry_point
                )
            else:
                function_info = problem.get("prompt", "")

        # Build AST section if needed
        ast_section = ""
        if include_ast:
            ast_repr = self.generate_ast_string(
                problem["canonical_solution"],
                problem["prompt"],
                problem["entry_point"],
            )
            ast_section = (
                f"\n\nAST representation of canonical solution:\n```\n{ast_repr}\n```\n"
            )

        fix_attempt_line = f"This is fix attempt {attempt} of {max_attempts}."

        # Load template and render
        template_name = self.FIX_TEMPLATE_MAPPING[(include_docstring, include_ast)]
        template = self.load_template(template_name)

        template_vars = {
            "function_info": function_info,
            "canonical_solution": problem["canonical_solution"],
            "ast_section": ast_section,
            "test_code": test_code,
            "error_output": error_output,
            "fix_attempt_line": fix_attempt_line,
        }

        try:
            return template.format(**template_vars)
        except KeyError as e:
            raise ValueError(
                f"Fix template '{template_name}' requires variable {e} "
                f"but it was not provided"
            )

    def get_template_hash(
        self, include_docstring: bool = False, include_ast: bool = False
    ) -> str:
        """
        Get a hash of the template content for reproducibility tracking.

        Args:
            include_docstring: Whether to include function docstring
            include_ast: Whether to include AST representation

        Returns:
            SHA256 hash of the template content
        """
        import hashlib

        template_name = self.get_template_name(include_docstring, include_ast)
        template_content = self.load_template(template_name)
        return hashlib.sha256(template_content.encode()).hexdigest()[:16]
