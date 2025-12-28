# Test cases for HumanEval/46
# Generated using Claude API



def fib4(n: int):
    """The Fib4 number sequence is a sequence similar to the Fibbonacci sequnece that's defined as follows:
    fib4(0) -> 0
    fib4(1) -> 0
    fib4(2) -> 2
    fib4(3) -> 0
    fib4(n) -> fib4(n-1) + fib4(n-2) + fib4(n-3) + fib4(n-4).
    Please write a function to efficiently compute the n-th element of the fib4 number sequence.  Do not use recursion.
    >>> fib4(5)
    4
    >>> fib4(6)
    8
    >>> fib4(7)
    14
    """

    results = [0, 0, 2, 0]
    if n < 4:
        return results[n]

    for _ in range(4, n + 1):
        results.append(results[-1] + results[-2] + results[-3] + results[-4])
        results.pop(0)

    return results[-1]


# Generated test cases:
import pytest
from typing import Any

@pytest.mark.parametrize("n,expected", [
    (0, 0),
    (1, 0),
    (2, 2),
    (3, 0),
    (4, 2),
    (5, 4),
    (6, 8),
    (7, 14),
    (8, 28),
    (9, 54),
    (10, 104),
    (15, 2764),
    (20, 73552),
    (25, 1957204),
    (30, 52080824),
])
def test_fib4_normal_cases(n, expected):
    assert fib4(n) == expected

@pytest.mark.parametrize("n", [
    -5,
    -100,
])
def test_fib4_negative_indices(n):
    with pytest.raises(IndexError):
        fib4(n)

@pytest.mark.parametrize("n", [
    3.5,
    "5",
    None,
    [4],
    {5: 1},
    (6,),
])
def test_fib4_invalid_types(n: Any):
    with pytest.raises((TypeError, IndexError)):
        fib4(n)