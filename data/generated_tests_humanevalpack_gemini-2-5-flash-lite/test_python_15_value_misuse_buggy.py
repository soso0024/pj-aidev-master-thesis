# Test cases for Python/15
# Generated using Claude API



def string_sequence(n: int) -> str:
    """ Return a string containing space-delimited numbers starting from 0 upto n inclusive.
    >>> string_sequence(0)
    '0'
    >>> string_sequence(5)
    '0 1 2 3 4 5'
    """

    return ' '.join([str(x) for x in range(n)])


# Generated test cases:
import pytest

@pytest.mark.parametrize("n,expected", [
    (0, "0"),
    (1, "0 1"),
    (5, "0 1 2 3 4 5"),
    (10, "0 1 2 3 4 5 6 7 8 9 10"),
])
def test_string_sequence_normal_cases(n, expected):
    assert string_sequence(n) == expected

@pytest.mark.parametrize("n,expected", [
    (0, "0"),
])
def test_string_sequence_edge_cases(n, expected):
    assert string_sequence(n) == expected

@pytest.mark.parametrize("n,expected", [
    (-1, ""),
    (-5, ""),
])
def test_string_sequence_negative_input(n, expected):
    # For negative input, range(n+1) will be empty, so join will return an empty string.
    # The original implementation had a bug where it returned "0" for negative inputs.
    # This test now asserts the correct behavior for negative inputs, which is an empty string.
    assert string_sequence(n) == expected

@pytest.mark.parametrize("n", [
    "abc",
    None,
    [1, 2],
    1.5,
])
def test_string_sequence_invalid_input_type(n):
    with pytest.raises(TypeError):
        string_sequence(n)