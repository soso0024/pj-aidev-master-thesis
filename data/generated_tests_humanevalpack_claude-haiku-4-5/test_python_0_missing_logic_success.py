# Test cases for Python/0
# Generated using Claude API

from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """

    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False


# Generated test cases:
import pytest
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False


class TestHasCloseElements:
    
    def test_empty_list(self):
        assert has_close_elements([], 0.1) == False
    
    def test_single_element(self):
        assert has_close_elements([1.0], 0.1) == False
    
    def test_two_elements_close(self):
        assert has_close_elements([1.0, 1.05], 0.1) == True
    
    def test_two_elements_not_close(self):
        assert has_close_elements([1.0, 2.0], 0.5) == False
    
    def test_two_elements_exactly_threshold(self):
        assert has_close_elements([1.0, 1.1], 0.1) == False
    
    def test_two_elements_just_below_threshold(self):
        assert has_close_elements([1.0, 1.099], 0.1) == True
    
    def test_multiple_elements_with_close_pair(self):
        assert has_close_elements([1.0, 5.0, 1.05], 0.1) == True
    
    def test_multiple_elements_no_close_pairs(self):
        assert has_close_elements([1.0, 5.0, 10.0], 0.5) == False
    
    def test_negative_numbers_close(self):
        assert has_close_elements([-1.0, -1.05], 0.1) == True
    
    def test_negative_numbers_not_close(self):
        assert has_close_elements([-1.0, -2.0], 0.5) == False
    
    def test_mixed_positive_negative_close(self):
        assert has_close_elements([-0.05, 0.05], 0.11) == True
    
    def test_mixed_positive_negative_not_close(self):
        assert has_close_elements([-1.0, 1.0], 0.5) == False
    
    def test_zero_threshold(self):
        assert has_close_elements([1.0, 1.0], 0.0) == False
    
    def test_zero_threshold_different_elements(self):
        assert has_close_elements([1.0, 1.1], 0.0) == False
    
    def test_large_threshold(self):
        assert has_close_elements([1.0, 100.0], 1000.0) == True
    
    def test_very_small_numbers(self):
        assert has_close_elements([0.0001, 0.00011], 0.00001) == True
    
    def test_very_large_numbers(self):
        assert has_close_elements([1e10, 1e10 + 1], 10.0) == True
    
    def test_duplicate_elements(self):
        assert has_close_elements([1.0, 1.0], 0.1) == True
    
    def test_three_elements_first_and_second_close(self):
        assert has_close_elements([1.0, 1.05, 10.0], 0.1) == True
    
    def test_three_elements_second_and_third_close(self):
        assert has_close_elements([1.0, 10.0, 10.05], 0.1) == True
    
    def test_three_elements_first_and_third_close(self):
        assert has_close_elements([1.0, 10.0, 1.05], 0.1) == True
    
    def test_three_elements_none_close(self):
        assert has_close_elements([1.0, 5.0, 10.0], 0.5) == False
    
    def test_many_elements_with_close_pair_at_end(self):
        assert has_close_elements([1.0, 2.0, 3.0, 4.0, 4.05], 0.1) == True
    
    def test_many_elements_no_close_pairs(self):
        assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == False
    
    def test_negative_threshold(self):
        assert has_close_elements([1.0, 1.05], -0.1) == False
    
    def test_float_precision(self):
        assert has_close_elements([0.1 + 0.2, 0.3], 0.0001) == True
    
    @pytest.mark.parametrize("numbers,threshold,expected", [
        ([1.0, 1.05], 0.1, True),
        ([1.0, 2.0], 0.5, False),
        ([], 0.1, False),
        ([1.0], 0.1, False),
        ([1.0, 1.0], 0.1, True),
        ([-1.0, -1.05], 0.1, True),
        ([0.0, 0.05], 0.1, True),
    ])
    def test_parametrized_cases(self, numbers, threshold, expected):
        assert has_close_elements(numbers, threshold) == expected