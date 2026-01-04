# Test cases for Python/19
# Generated using Claude API

from typing import List


def sort_numbers(numbers: str) -> str:
    """ Input is a space-delimited string of numberals from 'zero' to 'nine'.
    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
    Return the string with numbers sorted from smallest to largest
    >>> sort_numbers('three one five')
    'one three five'
    """

    value_map = {
        'zero': 0,
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9
    }
    return ' '.join([x for x in numbers.split(' ') if x])


# Generated test cases:
import pytest

def sort_numbers(numbers: str) -> str:
    value_map = {
        'zero': 0,
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9
    }
    return ' '.join([x for x in numbers.split(' ') if x])

@pytest.mark.parametrize("input_str,expected_output", [
    ("one two three", "one two three"),
    ("three two one", "one two three"),
    ("nine zero one", "zero one nine"),
    ("five four three two one zero", "zero one two three four five"),
    ("one", "one"),
    ("", ""),
    ("  ", ""),
    ("one  two   three", "one two three"),
    ("zero zero one one", "zero zero one one"),
    ("nine eight seven six five four three two one zero", "zero one two three four five six seven eight nine"),
    ("two one two one", "one one two two"),
    ("four zero four", "zero four four"),
])
def test_sort_numbers_normal_cases(input_str, expected_output):
    assert sort_numbers(input_str) == expected_output

@pytest.mark.parametrize("input_str,expected_output", [
    ("", ""),
    (" ", ""),
    ("   ", ""),
    ("one", "one"),
    ("zero", "zero"),
    ("nine", "nine"),
])
def test_sort_numbers_single_word_and_empty(input_str, expected_output):
    assert sort_numbers(input_str) == expected_output

@pytest.mark.parametrize("input_str,expected_output", [
    ("one two three four five six seven eight nine zero", "zero one two three four five six seven eight nine"),
    ("nine eight seven six five four three two one zero", "zero one two three four five six seven eight nine"),
    ("zero one two three four five six seven eight nine", "zero one two three four five six seven eight nine"),
])
def test_sort_numbers_all_digits(input_str, expected_output):
    assert sort_numbers(input_str) == expected_output

@pytest.mark.parametrize("input_str,expected_output", [
    ("one  two", "one two"),
    ("  three  four ", "three four"),
    ("five   six   seven", "five six seven"),
])
def test_sort_numbers_extra_spaces(input_str, expected_output):
    assert sort_numbers(input_str) == expected_output

@pytest.mark.parametrize("input_str", [
    "one two three four five six seven eight nine zero invalid",
    "invalid word",
    "one two invalid three",
    "zero one 123 four",
])
def test_sort_numbers_invalid_input_raises_keyerror(input_str):
    with pytest.raises(KeyError):
        sort_numbers(input_str)

@pytest.mark.parametrize("input_str,expected_output", [
    ("one two one", "one one two"),
    ("three three three", "three three three"),
    ("zero zero", "zero zero"),
])
def test_sort_numbers_duplicate_words(input_str, expected_output):
    assert sort_numbers(input_str) == expected_output
