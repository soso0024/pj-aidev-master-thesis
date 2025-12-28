# Test cases for HumanEval/147
# Generated using Claude API


def get_max_triples(n):
    """
    You are given a positive integer n. You have to create an integer array a of length n.
        For each i (1 ≤ i ≤ n), the value of a[i] = i * i - i + 1.
        Return the number of triples (a[i], a[j], a[k]) of a where i < j < k, 
    and a[i] + a[j] + a[k] is a multiple of 3.

    Example :
        Input: n = 5
        Output: 1
        Explanation: 
        a = [1, 3, 7, 13, 21]
        The only valid triple is (1, 7, 13).
    """

    A = [i*i - i + 1 for i in range(1,n+1)]
    ans = []
    for i in range(n):
        for j in range(i+1,n):
            for k in range(j+1,n):
                if (A[i]+A[j]+A[k])%3 == 0:
                    ans += [(A[i],A[j],A[k])]
    return len(ans)


# Generated test cases:
import pytest
from math import comb

@pytest.mark.parametrize("n,expected", [
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 1),
    (5, 1),
    (6, 4),
    (7, 10),
    (8, 11),
    (9, 21),
    (10, 36),
])
def test_get_max_triples_basic(n, expected):
    assert get_max_triples(n) == expected

@pytest.mark.parametrize("n", [
    -1,
    -10,
])
def test_get_max_triples_negative(n):
    assert get_max_triples(n) == 0

@pytest.mark.parametrize("n", [
    0,
    1,
    2,
])
def test_get_max_triples_small(n):
    assert get_max_triples(n) == 0

def test_get_max_triples_large():
    # Just check it runs and returns an int for a large n
    result = get_max_triples(100)
    assert isinstance(result, int)
    assert result >= 0

@pytest.mark.parametrize("n", [
    3.5,
    "5",
    None,
    [3],
    {3: 1},
])
def test_get_max_triples_invalid_type(n):
    with pytest.raises(TypeError):
        get_max_triples(n)