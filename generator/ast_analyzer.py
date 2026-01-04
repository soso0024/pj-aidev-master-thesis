"""
AST Analysis Module

Provides utilities for analyzing Python AST to generate focused snippets
for error fixing and debugging.
"""

import ast
import re
from typing import Any

from config import (
    MAX_AST_NODES,
    MAX_AST_OUTPUT_NODES,
    AST_SCORE_ERROR_MATCH,
    AST_SCORE_LINE_OVERLAP,
    AST_SCORE_COMMON_ERROR,
)


class ASTAnalyzer:
    """Analyzes Python AST to generate focused snippets for error context."""

    @staticmethod
    def _normalize_line(s: str) -> str:
        """Normalize a line of code by removing extra whitespace."""
        return re.sub(r"\s+", " ", s.strip())

    @staticmethod
    def _build_normalized_index(full_lines: list[str]) -> dict[str, list[int]]:
        """
        Build an index mapping normalized lines to their line numbers.
        
        Args:
            full_lines: List of code lines
            
        Returns:
            Dictionary mapping normalized line content to list of line numbers
        """
        index: dict[str, list[int]] = {}
        for idx, line in enumerate(full_lines, start=1):
            normalized = ASTAnalyzer._normalize_line(line)
            if not normalized:
                continue
            index.setdefault(normalized, []).append(idx)
        return index

    @staticmethod
    def _parse_error_output_for_lines(
        error_output: str,
        full_lines: list[str],
        normalized_index: dict[str, list[int]],
    ) -> set[int]:
        """
        Parse error output to identify relevant line numbers.
        
        Args:
            error_output: Pytest error output
            full_lines: List of code lines
            normalized_index: Index mapping normalized lines to line numbers
            
        Returns:
            Set of line numbers mentioned in the error output
        """
        candidate_set: set[int] = set()
        
        for raw_line in error_output.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            
            # Skip common header/footer lines
            if line.startswith("STDOUT:") or line.startswith("STDERR:"):
                continue
            if (
                line.startswith("=== ")
                or line.startswith("FAILED ")
                or line.startswith("PASSED ")
                or line.startswith("Traceback")
            ):
                continue

            # Extract explicit line numbers (e.g., "line 12")
            for match in re.finditer(r"\bline\s+(\d+)\b", line):
                try:
                    num = int(match.group(1))
                    if 1 <= num <= len(full_lines):
                        candidate_set.add(num)
                except ValueError:
                    pass

            # Check for pytest excerpt lines (starting with '>')
            if raw_line.lstrip().startswith(">"):
                excerpt = raw_line.lstrip().lstrip(">")
                normalized = ASTAnalyzer._normalize_line(excerpt)
                if normalized in normalized_index:
                    candidate_set.update(normalized_index[normalized])
                continue

            # General normalized content match
            normalized = ASTAnalyzer._normalize_line(line)
            if normalized and len(normalized.split()) >= 2 and normalized in normalized_index:
                candidate_set.update(normalized_index[normalized])

            # Lines prefixed with 'E ' (pytest assertion errors)
            if line.startswith("E "):
                line_after = line[1:].strip()
                if line_after:
                    normalized = ASTAnalyzer._normalize_line(line_after)
                    if normalized and len(normalized.split()) >= 2 and normalized in normalized_index:
                        candidate_set.update(normalized_index[normalized])

        return candidate_set

    @staticmethod
    def _expand_candidate_lines(
        tree: ast.AST,
        full_lines: list[str],
        candidate_set: set[int],
        node_span_cap: int = 20,
    ) -> list[int]:
        """
        Expand candidate lines to include surrounding context and AST nodes.
        
        Args:
            tree: Parsed AST
            full_lines: List of code lines
            candidate_set: Initial set of candidate line numbers
            node_span_cap: Maximum span of a node to include
            
        Returns:
            Sorted list of expanded line numbers
        """
        expanded: set[int] = set(candidate_set)
        
        # Add adjacent lines
        for line_num in list(candidate_set):
            if line_num - 1 >= 1:
                expanded.add(line_num - 1)
            if line_num + 1 <= len(full_lines):
                expanded.add(line_num + 1)

        # Add lines from AST nodes that overlap with candidate lines
        for node in ast.walk(tree):
            lineno = getattr(node, "lineno", None)
            end_lineno = getattr(node, "end_lineno", lineno)
            if lineno is None or end_lineno is None:
                continue
                
            # Check if node overlaps with any candidate line
            overlaps_any = any(lineno <= cl <= end_lineno for cl in candidate_set)
            if overlaps_any and (end_lineno - lineno + 1) <= node_span_cap:
                for line_idx in range(lineno, end_lineno + 1):
                    if 1 <= line_idx <= len(full_lines):
                        expanded.add(line_idx)
                        
        return sorted(expanded)

    @staticmethod
    def _create_error_predicates(error_output: str, entry_point: str) -> list:
        """
        Create AST node predicates based on error patterns in the output.
        
        Args:
            error_output: Pytest error output
            entry_point: Function entry point name
            
        Returns:
            List of predicate functions for matching AST nodes
        """
        predicates = []
        error_lower = error_output.lower()

        # Division errors
        if (
            "zerodivisionerror" in error_lower
            or "division by zero" in error_lower
            or "divide by zero" in error_lower
        ):
            predicates.append(
                lambda n: isinstance(n, ast.BinOp)
                and isinstance(n.op, (ast.Div, ast.Mod, ast.FloorDiv))
            )

        # Index errors
        if (
            "indexerror" in error_lower
            or "list index out of range" in error_lower
            or "string index out of range" in error_lower
            or "tuple index out of range" in error_lower
        ):
            predicates.append(lambda n: isinstance(n, ast.Subscript))

        # Key errors
        if "keyerror" in error_lower:
            predicates.append(lambda n: isinstance(n, ast.Subscript))

        # Attribute errors
        if "attributeerror" in error_lower or "has no attribute" in error_lower:
            predicates.append(lambda n: isinstance(n, ast.Attribute))

        # Type errors with operators
        if "typeerror" in error_lower and (
            "operand" in error_lower or "not supported between" in error_lower
        ):
            predicates.append(lambda n: isinstance(n, ast.BinOp))

        # Type errors with calls
        if "typeerror" in error_lower and (
            "takes" in error_lower or "required" in error_lower or "argument" in error_lower
        ):
            predicates.append(lambda n: isinstance(n, ast.Call))

        # Value errors
        if "valueerror" in error_lower:
            predicates.append(
                lambda n: isinstance(n, ast.Raise) or isinstance(n, ast.Call)
            )

        # Recursion errors
        if "recursionerror" in error_lower and entry_point:
            predicates.append(
                lambda n: isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == entry_point
            )

        # Name errors
        if "nameerror" in error_lower or "is not defined" in error_lower:
            predicates.append(lambda n: isinstance(n, ast.Name))

        # Import errors
        if "importerror" in error_lower or "modulenotfounderror" in error_lower:
            predicates.append(lambda n: isinstance(n, (ast.Import, ast.ImportFrom)))

        return predicates

    @staticmethod
    def generate_relevant_ast_snippet(
        function_code: str,
        error_output: str,
        entry_point: str = "",
    ) -> str:
        """
        Generate a compact AST snippet focusing on nodes related to the error.
        
        Args:
            function_code: Complete function code (signature + body)
            error_output: Pytest error output
            entry_point: Function entry point name
            
        Returns:
            AST snippet as string, or error message if parsing fails
        """
        try:
            tree = ast.parse(function_code)
            full_lines = function_code.splitlines()

            # Build index and find candidate lines
            normalized_index = ASTAnalyzer._build_normalized_index(full_lines)
            candidate_set = ASTAnalyzer._parse_error_output_for_lines(
                error_output, full_lines, normalized_index
            )
            candidate_lines = ASTAnalyzer._expand_candidate_lines(
                tree, full_lines, candidate_set
            )

            # Create error-specific predicates
            predicates = ASTAnalyzer._create_error_predicates(error_output, entry_point)

            # Collect nodes with priority scoring
            node_scores: list[tuple[ast.AST, int]] = []

            for node in ast.walk(tree):
                lineno = getattr(node, "lineno", None)
                end_lineno = getattr(node, "end_lineno", lineno)

                # Check if node overlaps with error lines
                overlaps = False
                if lineno is not None:
                    for line_num in candidate_lines:
                        if lineno <= line_num <= (end_lineno or lineno):
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
                        score += AST_SCORE_ERROR_MATCH
                    if overlaps:
                        score += AST_SCORE_LINE_OVERLAP
                    # Additional scoring for common error sources
                    if isinstance(
                        node, (ast.Call, ast.BinOp, ast.Subscript, ast.Attribute)
                    ):
                        score += AST_SCORE_COMMON_ERROR

                    if score > 0:
                        node_scores.append((node, score))

            # Sort by score and select top nodes
            node_scores.sort(key=lambda x: x[1], reverse=True)
            selected = [node for node, _ in node_scores[:MAX_AST_NODES]]

            if not selected:
                # Fallback: first few body statements
                func_def = next(
                    (n for n in tree.body if isinstance(n, ast.FunctionDef)), None
                )
                if func_def and func_def.body:
                    selected.extend(func_def.body[:3])

            # Generate concise AST representations
            parts: list[str] = []
            seen_nodes = set()

            for node in selected:
                # Create unique identifier to avoid duplicates
                node_id = (
                    type(node).__name__,
                    getattr(node, "lineno", None),
                    getattr(node, "col_offset", None),
                )
                if node_id in seen_nodes:
                    continue
                seen_nodes.add(node_id)

                try:
                    # Create concise representation
                    node_info = f"Line {getattr(node, 'lineno', '?')}: {type(node).__name__}"

                    # Add relevant details based on node type
                    if isinstance(node, ast.BinOp):
                        node_info += f" ({type(node.op).__name__})"
                    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        node_info += f" ({node.func.id})"
                    elif isinstance(node, ast.Attribute):
                        node_info += f" (.{node.attr})"
                    elif isinstance(node, ast.Name):
                        node_info += f" ({node.id})"

                    # Add full AST dump
                    parts.append(f"{node_info}\n{ast.dump(node, indent=2)}")
                except Exception:
                    parts.append(
                        f"Line {getattr(node, 'lineno', '?')}: {type(node).__name__}"
                    )

            return (
                "\n\n".join(parts[:MAX_AST_OUTPUT_NODES])
                if parts
                else "(no relevant AST nodes found)"
            )
            
        except Exception as e:
            return f"Error generating relevant AST snippet: {e}"



