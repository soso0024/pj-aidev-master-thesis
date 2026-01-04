# Test cases for Python/2
# Generated using Claude API



def truncate_number(number: float) -> float:
    """ Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """

    return number % 1.0 + 1.0


# Generated test cases:
import pytest
import math

@pytest.mark.parametrize("number,expected", [
    (3.14, 0.14000000000000012),
    (2.0, 0.0),
    (0.0, 0.0),
    (-3.14, 0.8599999999999999),
    (-2.0, 0.0),
    (123456789.987654321, 0.9876543283462524),
    (-123456789.987654321, 0.012345671653747559),
    (1e-10, 1e-10),
    (-1e-10, 0.9999999999),
    (1.9999999999, 0.9999999998999999),
    (-1.9999999999, 1.000000082740371e-10),
])
def test_truncate_number_normal_cases(number, expected):
    from test_python_2_excess_logic import truncate_number
    result = truncate_number(number)
    assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-12)

@pytest.mark.parametrize("number", [
    (float('inf')),
    (float('-inf')),
    (float('nan')),
])
def test_truncate_number_special_floats(number):
    try:
        result = truncate_number(number)
    except Exception as e:
        assert isinstance(e, (ValueError, OverflowError))
    else:
        assert math.isnan(result)

@pytest.mark.parametrize("number", [
    ("string"),
    (None),
    ([1.23]),
    ({"number": 1.23}),
    (object()),
])
def test_truncate_number_type_error(number):
    with pytest.raises(TypeError):
        truncate_number(number)