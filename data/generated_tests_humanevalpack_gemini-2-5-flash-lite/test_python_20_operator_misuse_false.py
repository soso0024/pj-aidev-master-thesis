# Test cases for Python/20
# Generated using Claude API

from typing import List, Tuple


def find_closest_elements(numbers: List[float]) -> Tuple[float, float]:
    """ From a supplied list of numbers (of length at least two) select and return two that are the closest to each
    other and return them in order (smaller number, larger number).
    >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2])
    (2.0, 2.2)
    >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0])
    (2.0, 2.0)
    """

    closest_pair = None
    distance = None

    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                if distance is None:
                    distance = abs(elem - elem2)
                    closest_pair = tuple(sorted([elem, elem2]))
                else:
                    new_distance = abs(elem - elem2)
                    if new_distance < distance:
                        distance = new_distance
                        closest_pair = tuple(sorted([elem, elem2]))

    return closest_pair


# Generated test cases:
import pytest
from typing import List, Tuple

@pytest.mark.parametrize("numbers,expected", [
    ([1.0, 2.0, 3.0, 4.0], (1.0, 2.0)),
    ([5.5, 2.1, 8.9, 3.2, 7.0], (2.1, 3.2)),
    ([10.0, 10.1, 10.2, 10.3], (10.0, 10.1)),
    ([-1.0, -2.0, -3.0, -4.0], (-2.0, -1.0)),
    ([1.0, -1.0, 2.0, -2.0], (-1.0, 1.0)),
    ([0.0, 0.0, 0.0], (0.0, 0.0)),
    ([1.0, 1.0, 2.0, 2.0], (1.0, 1.0)),
    ([1.0, 5.0, 10.0, 15.0, 20.0], (1.0, 5.0)),
    ([20.0, 15.0, 10.0, 5.0, 1.0], (1.0, 5.0)),
    ([1.23, 4.56, 7.89, 1.24], (1.23, 1.24)),
    ([100.0, 0.1, 50.0, 0.2], (0.1, 0.2)),
    ([1.0, 1.0000001, 2.0], (1.0, 1.0000001)),
    ([1.0, 1.0, 1.0, 1.0], (1.0, 1.0)),
    ([1.0, 2.0, 1.0, 2.0], (1.0, 1.0)),
    ([1.0, 3.0, 5.0, 7.0, 9.0], (1.0, 3.0)),
    ([9.0, 7.0, 5.0, 3.0, 1.0], (1.0, 3.0)),
    ([1.0, 10.0, 2.0, 9.0, 3.0, 8.0], (1.0, 2.0)),
    ([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], (1.0, 1.0)),
    ([1.0, 2.0, 3.0, 1.5, 2.5], (1.0, 1.5)),
    ([1.0, 2.0, 3.0, 2.5, 1.5], (1.0, 1.5)),
])
def test_find_closest_elements_normal_cases(numbers, expected):
    assert find_closest_elements(numbers) == expected

@pytest.mark.parametrize("numbers,expected", [
    ([5.0, 5.0], (5.0, 5.0)),
    ([1.0, 1.0, 1.0], (1.0, 1.0)),
    ([1.0, 2.0, 1.0], (1.0, 1.0)),
    ([2.0, 1.0, 2.0], (1.0, 2.0)),
])
def test_find_closest_elements_duplicate_numbers(numbers, expected):
    assert find_closest_elements(numbers) == expected

@pytest.mark.parametrize("numbers,expected", [
    ([1.0, 1000000.0], (1.0, 1000000.0)),
    ([-1000000.0, 1.0], (-1000000.0, 1.0)),
    ([0.0, 1000000.0], (0.0, 1000000.0)),
])
def test_find_closest_elements_large_range(numbers, expected):
    assert find_closest_elements(numbers) == expected

@pytest.mark.parametrize("numbers,expected", [
    ([1e-9, 2e-9, 3e-9], (1e-9, 2e-9)),
    ([1.0, 1.0 + 1e-9, 2.0], (1.0, 1.0 + 1e-9)),
    ([1.0, 2.0, 1.0 + 1e-9], (1.0, 1.0 + 1e-9)),
])
def test_find_closest_elements_small_differences(numbers, expected):
    assert find_closest_elements(numbers) == expected

@pytest.mark.parametrize("numbers,expected", [
    ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], (1.0, 2.0)),
    ([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], (1.0, 2.0)),
])
def test_find_closest_elements_larger_list(numbers, expected):
    assert find_closest_elements(numbers) == expected

def test_find_closest_elements_empty_list():
    with pytest.raises(ValueError, match="Input list must contain at least two elements."):
        find_closest_elements([])

def test_find_closest_elements_single_element():
    with pytest.raises(ValueError, match="Input list must contain at least two elements."):
        find_closest_elements([5.0])