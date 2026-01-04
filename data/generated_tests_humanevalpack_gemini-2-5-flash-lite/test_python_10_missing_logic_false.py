# Test cases for Python/10
# Generated using Claude API



def is_palindrome(string: str) -> bool:
    """ Test if given string is a palindrome """
    return string == string[::-1]


def make_palindrome(string: str) -> str:
    """ Find the shortest palindrome that begins with a supplied string.
    Algorithm idea is simple:
    - Find the longest postfix of supplied string that is a palindrome.
    - Append to the end of the string reverse of a string prefix that comes before the palindromic suffix.
    >>> make_palindrome('')
    ''
    >>> make_palindrome('cat')
    'catac'
    >>> make_palindrome('cata')
    'catac'
    """

    if not string:
        return ''

    beginning_of_suffix = 0

    while not is_palindrome(string[beginning_of_suffix:]):
        beginning_of_suffix += 1

    return string + string[:beginning_of_suffix][::-1]


# Generated test cases:
import pytest

def is_palindrome(s: str) -> bool:
    return s == s[::-1]

def make_palindrome(string: str) -> str:
    if not string:
        return ''

    beginning_of_suffix = 0

    while not is_palindrome(string[beginning_of_suffix:]):
        beginning_of_suffix += 1

    return string + string[:beginning_of_suffix][::-1]

@pytest.mark.parametrize("input_string,expected_output", [
    ("", ""),
    ("a", "a"),
    ("aa", "aa"),
    ("aba", "aba"),
    ("ab", "aba"),
    ("race", "racecar"),
    ("google", "googlelgoog"),
    ("level", "level"),
    ("madam", "madam"),
    ("aabbaa", "aabbaa"),
    ("abc", "abcba"),
    ("abcd", "abcdcba"),
    ("aaaaa", "aaaaa"),
    ("abacaba", "abacaba"),
    ("topcoder", "topcoderedocpot"),
    ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
    ("abcdefghijklmnopqrstuvwxyz", "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzyxwvutsrqponmlkjihgfedcba"),
    ("abracadabra", "abracadabracadabra"),
    ("zzza", "zzzaazzz"),
    ("azzz", "azzza"),
    ("aabba", "aabbaa"),
    ("baabb", "baabbaabb"),
])
def test_make_palindrome(input_string, expected_output):
    assert make_palindrome(input_string) == expected_output

@pytest.mark.parametrize("input_string,expected_output", [
    ("a b", "a b a"),
    ("121", "121"),
    ("123", "12321"),
    ("!@#", "!@#@!"),
    ("a!b", "a!b!a"),
    ("race car", "race car rac ecar"),
])
def test_make_palindrome_special_chars(input_string, expected_output):
    assert make_palindrome(input_string) == expected_output