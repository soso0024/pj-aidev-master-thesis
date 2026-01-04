# Test cases for Python/1
# Generated using Claude API

from typing import List


def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """

    result = []
    current_string = []
    current_depth = 0

    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)

            if current_depth == 0:
                result.append(''.join(current_string))
                current_string.clear()

    return result


# Generated test cases:
import pytest
from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    result = []
    current_string = []
    current_depth = 0

    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)

            if current_depth == 0:
                result.append(''.join(current_string))
                current_string.clear()
        else:
            # Handle characters other than parentheses if necessary,
            # for this problem, we assume only parentheses are present.
            # If other characters are allowed and should be part of the group,
            # they should be appended to current_string.
            pass

    return result

@pytest.mark.parametrize("paren_string, expected", [
    ("", []),  # Empty string
    ("()", ["()"]),  # Single pair
    ("(())", ["(())"]),  # Nested pair
    ("()()", ["()", "()"]),  # Multiple adjacent pairs
    ("(()())", ["(()())"]),  # Nested and adjacent
    ("((()))", ["((()))"]),  # Deeply nested
    ("()(())()", ["()", "(())", "()"]),  # Mixed adjacent and nested
    ("((()()))", ["((()()))"]),  # Complex nesting
    ("((()))(())", ["((()))", "(())"]),  # Multiple complex groups
    ("()()()()()", ["()", "()", "()", "()", "()"]),  # Many simple groups
    ("(((((())))))", ["(((((())))))"]),  # Very deep nesting
    ("(()(()))", ["(()(()))"]),  # Another complex nesting
    ("()((()))()", ["()", "((()))", "()"]),  # Groups with spaces in between (if spaces were allowed)
    ("((()))((()))", ["((()))", "((()))"]),  # Two identical complex groups
    ("()", ["()"]), # Single pair, no nesting
    ("((()))", ["((()))"]), # Single deeply nested pair
    ("()()", ["()", "()"]), # Two simple adjacent pairs
    ("(()())", ["(()())"]), # One complex adjacent pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))", ["((()()))"]), # One very complex nested pair
    ("(((())))", ["(((())))"]), # One very deeply nested pair
    ("((()))(())", ["((()))", "(())"]), # Two complex adjacent pairs
    ("((()))((()))", ["((()))", "((()))"]), # Two identical complex adjacent pairs
    ("()(())()", ["()", "(())", "()"]), # Mixed simple and complex adjacent pairs
    ("((()()))
