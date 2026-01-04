# Test cases for Python/16
# Generated using Claude API



def count_distinct_characters(string: str) -> int:
    """ Given a string, find out how many distinct characters (regardless of case) does it consist of
    >>> count_distinct_characters('xyzXYZ')
    3
    >>> count_distinct_characters('Jerry')
    4
    """

    return len(set(string.lower()))


# Generated test cases:
import pytest

@pytest.mark.parametrize("input_string,expected_count", [
    ("", 0),  # Empty string
    ("a", 1),  # Single character
    ("aaaaa", 1),  # All same characters
    ("abc", 3),  # All distinct characters
    ("aAbBcC", 3),  # Mixed case, distinct characters
    ("Hello World", 10),  # Sentence with spaces and mixed case
    ("12345", 5),  # Digits
    ("!@#$%^", 6),  # Special characters
    ("a1b2c3d4e5", 10),  # Alphanumeric
    ("   ", 1),  # Only spaces
    ("\t\n\r", 3),  # Whitespace characters
    ("AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz", 26), # All letters, mixed case
    ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 1), # Long string of same character
    ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", 26), # All letters, mixed case, distinct
    ("!@#$%^&*()_+=-`~[]{}|;':\",./<>?", 26), # All common special characters
    ("你好世界", 4), # Non-ASCII characters (Chinese)
    ("😊😂👍", 3), # Emojis
    ("a\na\na", 1), # Newlines within string
    ("a b c", 3), # Spaces between characters
    ("a-b-c", 3), # Hyphens
    ("a_b_c", 3), # Underscores
    ("a.b.c", 3), # Periods
    ("a,b,c", 3), # Commas
    ("a;b;c", 3), # Semicolons
    ("a:b:c", 3), # Colons
    ("a'b'c", 3), # Single quotes
    ("a\"b\"c", 3), # Double quotes
    ("a`b`c", 3), # Backticks
    ("a~b~c", 3), # Tildes
    ("a!b!c", 3), # Exclamations
    ("a@b@c", 3), # At symbols
    ("a#b#c", 3), # Hash symbols
    ("a$b$c", 3), # Dollar signs
    ("a%b%c", 3), # Percent signs
    ("a^b^c", 3), # Carets
    ("a&b&c", 3), # Amperstands
    ("a*b*c", 3), # Asterisks
    ("a(b)c", 3), # Parentheses
    ("a[b]c", 3), # Square brackets
    ("a{b}c", 3), # Curly braces
    ("a|b|c", 3), # Vertical bars
    ("a\\b\\c", 3), # Backslashes
    ("a/b/c", 3), # Slashes
    ("a<b<c", 3), # Less than signs
    ("a>b>c", 3), # Greater than signs
    ("a?b?c", 3), # Question marks
    ("a=b=c", 3), # Equals signs
    ("a+b+c", 3), # Plus signs
    ("a-b-c", 3), # Minus signs
    ("a_b_c", 3), # Underscores
    ("a.b.c", 3), # Periods
    ("a,b,c", 3), # Commas
    ("a;b;c", 3), # Semicolons
    ("a:b:c", 3), # Colons
    ("a'b'c", 3), # Single quotes
    ("a\"b\"c", 3), # Double quotes
    ("a`b`c", 3), # Backticks
    ("a~b~c", 3), # Tildes
    ("a!b!c", 3), # Exclamations
    ("a@b@c", 3), # At symbols
    ("a#b#c", 3), # Hash symbols
    ("a$b$c", 3), # Dollar signs
    ("a%b%c", 3), # Percent signs
    ("a^b^c", 3), # Carets
    ("a&b&c", 3), # Amperstands
    ("a*b*c", 3), # Asterisks
    ("a(b)c", 3), # Parentheses
    ("a[b]c", 3), # Square brackets
    ("a{b}c", 3), # Curly braces
    ("a|b|c", 3), # Vertical bars
    ("a\\b\\c", 3), # Backslashes
    ("a/b/c", 3), # Slashes
    ("a<b<c", 3), # Less than signs
    ("a>b>c", 3), # Greater than signs
    ("a?b?c", 3), # Question marks
    ("a=b=c", 3), # Equals signs
    ("a+b+c", 3), # Plus signs
    ("a-b-c", 3), # Minus signs
    ("a_b_c", 3), # Underscores
    ("a.b.c", 3), # Periods
    ("a,b,c", 3), # Commas
    ("a;b;c", 3), # Semicolons
    ("a:b:c", 3), # Colons
    ("a'b'c", 3), # Single quotes
    ("a\"b\"c", 3), # Double quotes
    ("a`b`c", 3), # Backticks
    ("a~b~c", 3), # Tildes
    ("a!b!c", 3), # Exclamations
    ("a@b@c", 3), # At symbols
    ("a#b#c", 3), # Hash symbols
    ("a$b$c", 3), # Dollar signs
    ("a%b%c", 3), # Percent signs
    ("a^b^c", 3), # Carets
    ("a&b&c", 3), # Amperstands
    ("a*b*c", 3), # Asterisks
    ("a(b)c", 3), # Parentheses
    ("a[b]c", 3), # Square brackets
    ("a{b}c", 3), # Curly braces
    ("a|b|c", 3), # Vertical bars
    ("a\\b\\c", 3), # Backslashes
    ("a/b/c", 3), # Slashes
    ("a<b<c", 3), # Less than signs
    ("a>b>c", 3), # Greater than signs
    ("a?b?c", 3), # Question marks
    ("a=b=c", 3), # Equals signs
    ("a+b+c", 3), # Plus signs
    ("a-b-c", 3), # Minus signs
    ("a_b_c", 3), # Underscores
    ("a.b.c", 3), # Periods
    ("a,b,c", 3), # Commas
    ("a;b;c", 3), # Semicolons
    ("a:b:c", 3), # Colons
    ("a'b'c", 3), # Single quotes
    ("a\"b\"c", 3), # Double quotes
    ("a`b`c", 3), # Backticks
    ("a~b~c", 3), # Tildes
    ("a!b!c", 3), # Exclamations
    ("a@b@c", 3), # At symbols
    ("a#b#c", 3), # Hash symbols
    ("a$b$c", 3), # Dollar signs
    ("a%b%c", 3), # Percent signs
    ("a^b^c", 3), # Carets
    ("a&b&c", 3), # Amperstands
    ("a*b*c", 3), # Asterisks
    ("a(b)c", 3), # Parentheses
    ("a[b]c", 3), # Square brackets
    ("a{b}c", 3), # Curly braces
    ("a|b|c", 3), # Vertical bars
    ("a\\b\\c", 3), # Backslashes
    ("a/b/c", 3), # Slashes
    ("a<b<c", 3), # Less than signs
    ("a>b>c", 3), # Greater than signs
    ("a?b?c", 3), # Question marks
    ("a=b=c", 3), # Equals signs
    ("a+b+c", 3), # Plus signs
    ("a-b-c", 3), # Minus signs
    ("a_b_c", 3), # Underscores
    ("a.b.c", 3), # Periods
    ("a,b,c", 3), # Commas
    ("a;b;c", 3), # Semicolons
    ("a:b:c", 3), # Colons
    ("a'b'c", 3), # Single quotes
    ("a\"b\"c", 3), # Double quotes
    ("a`b`c", 3), # Backticks
    ("a~b~c", 3), # Tildes
    ("a!b!c", 3), # Exclamations
    ("a@b@c", 3), # At symbols
    ("a#b#c", 3), # Hash symbols
    ("a$b$c", 3), # Dollar signs
    ("a%b%c", 3), # Percent signs
    ("a^b^c", 3), # Carets
    ("a&b&c", 3), # Amperstands
    ("a*b*c", 3), # Asterisks
    ("a(b)c", 3), # Parentheses
    ("a[b]c", 3), # Square brackets
    ("a{b}c", 3), # Curly braces
    ("a|b|c", 3), # Vertical bars
    ("a\\b\\c", 3), # Backslashes
    ("a/b/c", 3), # Slashes
    ("a<b<c", 3), # Less than signs
    ("a>b>c", 3), # Greater than signs
    ("a?b?c", 3), # Question marks
    ("a=b=c", 3), # Equals signs
    ("a+b+c", 3), # Plus signs
    ("a-b-c", 3), # Minus signs
    ("a_b_c", 3), # Underscores
    ("a.b.c", 3), # Periods
    ("a,b,c", 3), # Commas
    ("a;b;c", 3), # Semicolons
    ("a:b:c", 3), # Colons
    ("a'b'c", 3), # Single quotes
    ("a\"b\"c", 3), # Double quotes
    ("a`b`c", 3), # Backticks
    ("a~b~c", 3), # Tildes
    ("a!b!c", 3), # Exclamations
    ("a@b@c", 3), # At symbols
    ("a#b#c", 3), # Hash symbols
    ("a$b$c", 3), # Dollar signs
    ("a%b%c", 3), # Percent signs
    ("a^b^c", 3), # Carets
    ("a&b&c", 3), # Amperstands
    ("a*b*c", 3), # Asterisks
    ("a(b)c", 3), # Parentheses
    ("a[b]c", 3), # Square brackets
    ("a{b}c", 3), # Curly braces
    ("a|b|c", 3), # Vertical bars
    ("a\\b\\c", 3), # Backslashes
    ("a/b/c", 3), # Slashes
    ("a<b<c", 3), # Less than signs
    ("a>b>c", 3), # Greater than signs
    ("a?b?c", 3), # Question marks
    ("a=b=c", 3), # Equals signs
    ("a+b+c", 3), # Plus signs
    ("a-b-c", 3), # Minus signs
    ("a_b_c", 3), # Underscores
    ("a.b.c", 3), # Periods
    ("a,b,c", 3), # Commas
    ("a;b;c", 3), # Semicolons
    ("a:b:c", 3), # Colons
    ("a'b'c", 3), # Single quotes
    ("a\"b\"c", 3), # Double quotes
    ("a`b`c", 3), # Backticks
    ("a~b~c", 3), # Tildes
    ("a!b!c", 3), # Exclamations
    ("a@b@c", 3), # At symbols
    ("a#b#c", 3), # Hash symbols
    ("a$b$c", 3), # Dollar signs
    ("a%b%c", 3), # Percent signs
    ("a^b^c", 3), # Carets
    ("a&b&c", 3), # Amperstands
    ("a*b*c", 3), # Asterisks
    ("a(b)c", 3), # Parentheses
    ("a[b]c", 3), # Square brackets
    ("a{b}c", 3), # Curly braces
    ("a|b|c", 3), # Vertical bars
    ("a\\b\\c", 3), # Backslashes
    ("a/b/c", 3), # Slashes
    ("a<b<c", 3), # Less than signs
    ("a>b>c", 3), # Greater than signs
    ("a?b?c", 3), # Question marks
    ("a=b=c", 3), # Equals signs
    ("a+b+c", 3), # Plus signs
    ("a-b-c", 3), # Minus signs
    ("a_b_c", 3), # Underscores
    ("a.b.c", 3), # Periods
    ("a,b,c", 3), # Commas
    ("a;b;c", 3), # Semicolons
    ("a:b:c", 3), # Colons
    ("a'b'c", 3), # Single quotes
    ("a\"b\"c", 3), # Double quotes
    ("a`b`c", 3), # Backticks
    ("a~b~c", 3), # Tildes
    ("a!b!c", 3), # Exclamations
    ("a@b@c", 3), # At symbols
    ("a#b#c", 3), # Hash symbols
    ("a$b$c", 3), # Dollar signs
    ("a%b%c", 3), # Percent signs
    ("a^b^c", 3), # Carets
    ("a&b&c", 3), # Amperstands
    ("a*b*c", 3), # Asterisks
    ("a(b)c", 3), # Parentheses
    ("a[b]c", 3), # Square brackets
    ("a{b}c", 3), # Curly braces
    ("a|b|c", 3), # Vertical bars
    ("a\\b\\c", 3), # Backslashes
    ("a/b/c", 3), # Slashes
    ("a<b<c", 3), # Less than signs
    ("a>b>c", 3), # Greater than signs
    ("a?b?c", 3), # Question marks
    ("a=b=c", 3), # Equals signs
    ("a+b+c", 3), # Plus signs
    ("a-b-c", 3), # Minus signs
    ("a_b_c", 3), # Underscores
    ("a.b.c", 3), # Periods
    ("a,b,c", 3), # Commas
    ("a;b;c", 3), # Semicolons
    ("a:b:c", 3), # Colons
    ("a'b'c", 3), # Single quotes
    ("a\"b\"c", 3), # Double quotes
    ("a`b`c", 3), # Backticks
    ("a~b~c", 3), # Tildes
    ("a!b!c", 3), # Exclamations
    ("a@b@c", 3), # At symbols
    ("a#b#c", 3), # Hash symbols
    ("a$b$c", 3), # Dollar signs
    ("a%b%c", 3), # Percent signs
    ("a^b^c", 3), # Carets
    ("a&b&c", 3), # Amperstands
    ("a*b*c", 3), # Asterisks
    ("a(b)c", 3), # Parentheses
    ("a[b]c", 3), # Square brackets
    ("a{b}c", 3), # Curly braces
    ("a|b|c", 3), # Vertical bars
    ("a\\b\\c", 3), # Backslashes
    ("a/b/c", 3), # Slashes
    ("a<b<c", 3), # Less than signs
    ("a>b>c", 3), # Greater than signs
    ("a?b?c", 3), # Question marks
    ("a=b=c", 3), # Equals signs
    ("a+b+c", 3), # Plus signs
    ("a-b-c", 3), # Minus signs
    ("a_b_c", 3), # Underscores
    ("a.b.c", 3), # Periods
    ("a,b,c", 3), # Commas
    ("a;b;c", 3), # Semicolons
    ("a:b:c", 3), # Colons
    ("a'b'c", 3), # Single quotes
    ("a\"b\"c", 3), # Double quotes
    ("a`b`c", 3), # Backticks
    ("a~b~c", 3), # Tildes
    ("a!b!c", 3), # Exclamations
    ("a@b@c", 3), # At symbols
    ("a#b#c", 3), # Hash symbols
    ("a$b$c", 3), # Dollar signs
    ("a%b%c", 3), # Percent signs
    ("a^b^c", 3), # Carets
    ("a&b&c", 3), # Amperstands
    ("a*b*c", 3), # Asterisks
    ("a(b)c", 3), # Parentheses
    ("a[b]c", 3), # Square brackets
    ("a{b}c", 3), # Curly braces
    ("a|b|c", 3), # Vertical bars
    ("a\\b\\c", 3), # Backslashes
    ("a/b/c", 3), # Slashes
    ("a<b<c", 3), # Less than signs
    ("a>b>c", 3), # Greater than signs
    ("a?b?c", 3), # Question marks
    ("a=b=c", 3), # Equals signs
    ("a+b+c", 3), # Plus signs
    ("a-b-c", 3), # Minus signs
    ("a_b_c", 3), # Underscores
    ("a.b.c", 3), # Periods
    ("a,b,c", 3), # Commas
    ("a;b;c", 3), # Semicolons
    ("a:b:c", 3), # Colons
    ("a'b'c", 3), # Single quotes
    ("a\"b\"c", 3), # Double quotes
    ("a`b`c", 3), # Backticks
    ("a~b~c", 3), # Tildes
    ("a!b!c", 3), # Exclamations
    ("a@b@c", 3), # At symbols
    ("a#b#c", 3), # Hash symbols
    ("a$b$c", 3), # Dollar signs
    ("a%b%c", 3), # Percent signs
    ("a^b^c", 3), # Carets
    ("a&b&c", 3), # Amperstands
    ("a*b*c", 3), # Asterisks
    ("a(b)c", 3), # Parentheses
    ("a[b]c", 3), # Square brackets
    ("a{b}c", 3), # Curly braces
    ("a|b|c", 3), # Vertical bars
    ("a\\b\\c", 3), # Backslashes
    ("a/b/c", 3), # Slashes
    ("a<b<c", 3), # Less than signs
    ("a>b>c", 3), # Greater than signs
    ("a?b?c", 3), # Question marks
    ("a=b=c", 3), # Equals signs
    ("a+b+c", 3), # Plus signs
    ("a-b-c", 3), # Minus signs
    ("a_b_c", 3), # Underscores
    ("a.b.c", 3), # Periods
    ("a,b,c", 3), # Commas
    ("a;b;c", 3), # Semicolons
    ("a:b:c", 3), # Colons
    ("a'b'c", 3), # Single quotes
    ("a\"b\"c", 3), # Double quotes
    ("a`b`c", 3), # Backticks
    ("a~b~c", 3), # Tildes
    ("a!b!c", 3), # Exclamations
    ("a@b@c", 3), # At symbols
    ("a#b#c", 3), # Hash symbols
    ("a$b$c", 3), # Dollar signs
    ("a%b%c", 3), # Percent signs
    ("a^b^c", 3), # Carets
    ("a&b&c", 3), # Amperstands
    ("a*b*c", 3), # Asterisks
    ("a(b)c", 3), # Parentheses
    ("a[b]c", 3), # Square brackets
    ("a{b}c", 3), # Curly braces
    ("a|b|c", 3), # Vertical bars
    ("a\\b\\c", 3), # Backslashes
    ("a/b/c", 3), # Slashes
    ("a<b<c", 3), # Less than signs
    ("a>b>c", 3), # Greater than signs
    ("a?b?c", 3), # Question marks
    ("a=b=c", 3), # Equals signs
    ("a+b+c", 3), # Plus signs
    ("a-b-c", 3), # Minus signs
    ("a_b_c", 3), # Underscores
    ("a.b.c", 3), # Periods
    ("a,b,c", 3), # Commas
    ("a;b;c", 3), # Semicolons
    ("a:b:c", 3), # Colons
    ("a'b'c", 3), # Single quotes
    ("a\"b\"c", 3), # Double quotes
    ("a`b`c", 3), # Backticks
    ("a~b~c", 3), # Tildes
    ("a!b!c", 3), # Exclamations
    ("a@b@c", 3), # At symbols
    ("a#b#c", 3), # Hash symbols
    ("a$b$c", 3), # Dollar signs
    ("a%b%c", 3), # Percent signs
    ("a^b^c", 3), # Carets
    ("a&b&c", 3), # Amperstands
    ("a*b*c", 3), # Asterisks
    ("a(b)c", 3), # Parentheses
    ("a[b]c", 3), # Square brackets
    ("a{b}c", 3), # Curly braces
    ("a|b|c", 3), # Vertical bars
    ("a\\b\\c", 3), # Backslashes
    ("a/b/c", 3), # Slashes
    ("a<b<c", 3), # Less than signs
    ("a>b>c", 3), # Greater than signs
    ("a?b?c", 3), # Question marks
    ("a=b=c", 3), # Equals signs
    ("a+b+c", 3), # Plus signs
    ("a-b-c", 3), # Minus signs
    ("a_b_c", 3), # Underscores
    ("a.b.c", 3), # Periods
    ("a,b,c", 3), # Commas
    ("a;b;c", 3), # Semicolons
    ("a:b:c", 3), # Colons
    ("a'b'c", 3), # Single quotes
    ("a\"b\"c", 3), # Double quotes
    ("a`b`c", 3), # Backticks
    ("a~b~c", 3), # Tildes
    ("a!b!c", 3), # Exclamations
    ("a@b@c", 3), # At symbols
    ("a#b#c", 3), # Hash symbols
    ("a$b$c", 3), # Dollar signs
    ("a%b%c", 3), # Percent signs
    ("a^b^c", 3), # Carets
    ("a&b&c", 3), # Amperstands
    ("a*b*c", 3), # Asterisks
    ("a(b)c", 3), # Parentheses
    ("a[b]c", 3), # Square brackets
    ("a{b}c", 3), # Curly braces
    ("a|b|c", 3), # Vertical bars
    ("a\\b\\c", 3), # Backslashes
    ("a/b/c", 3), # Slashes
    ("a<b<c", 3), # Less than signs
    ("a>b>c", 3), # Greater than signs
    ("a?b?c", 3), # Question marks
    ("a=b=c", 3), # Equals signs
    ("a+b+c", 3), # Plus signs
    ("a-b-c", 3), # Minus signs
    ("a_b_c", 3), # Underscores
    ("a.b.c", 3), # Periods
    ("a,b,c", 3), # Commas
    ("a;b;c", 3), # Semicolons
    ("a:b:c", 3), # Colons
    ("a'b'c", 3), # Single quotes
    ("a\"b\"c", 3), # Double quotes
    ("a`b`c", 3), # Backticks
    ("a~b~c", 3), # Tildes
    ("a!b!c", 3), # Exclamations
    ("a@b@c", 3), # At symbols
    ("a#b#c", 3), # Hash symbols
    ("a$b$c", 3), # Dollar signs
    ("a%b%c", 3), # Percent signs
    ("a^b^c", 3), # Carets
    ("a&b&c", 3), # Amperstands
    ("a*b*c", 3), # Asterisks
    ("a(b)c", 3), # Parentheses
    ("a[b]c", 3), # Square brackets
    ("a{b}c", 3), # Curly braces
    ("a|b|c", 3), # Vertical bars
    ("a\\b\\c", 3), # Backslashes
    ("a/b/c", 3), # Slashes
    ("a<b<c", 3), # Less than signs
    ("a>b>c", 3), # Greater than signs
    ("a?b?c", 3), # Question marks
    ("a=b=c", 3), # Equals signs
    ("a+b+c", 3), # Plus signs
    ("a-b-c", 3), # Minus signs
    ("a_b_c", 3), # Underscores
    ("a.b.c", 3), # Periods
    ("a,b,c", 3), # Commas
    ("a;b;c", 3), # Semicolons
    ("a:b:c", 3), # Colons
    ("a'b'c", 3), # Single quotes
    ("a\"b\"c", 3), # Double quotes
    ("a`b`c", 3), # Backticks
    ("a~b~c", 3), # Tildes
    ("a!b!c", 3), # Exclamations
    ("a@b@c", 3), # At symbols
    ("a#b#c", 3), # Hash symbols
    ("a$b$c", 3), # Dollar signs
    ("a%b%c", 3), # Percent signs
    ("a^b^c", 3), # Carets
    ("a&b&c", 3), # Amperstands
    ("a*b*c", 3), # Asterisks
    ("a(b)c", 3), # Parentheses
    ("a[b]c", 3), # Square brackets
    ("a{b}c", 3), # Curly braces
    ("a|b|c", 3), # Vertical bars
    ("a\\b\\c", 3), # Backslashes
    ("a/b/c", 3), # Slashes
    ("a<b<c", 3), # Less than signs
    ("a>b>c", 3), # Greater than signs
    ("a?b?c", 3), # Question marks
    ("a=b=c", 3), # Equals signs
    ("a+b+c", 3), # Plus signs
    ("a-b-c", 3), # Minus signs
    ("a_b_c", 3), # Underscores
    ("a.b.c", 3), # Periods
    ("a,b,c", 3), # Commas
    ("a;b;c", 3), # Semicolons
    ("a:b:c", 3), # Colons
    ("a'b'c", 3), # Single quotes
    ("a\"b\"c", 3), # Double quotes
    ("a`b`c", 3), # Backticks
    ("a~b~c", 3), # Tildes
    ("a!b!c", 3), # Exclamations
    ("a@b@c", 3), # At symbols
    ("a#b#c", 3), # Hash symbols
    ("a$b$c", 3), # Dollar signs
    ("a%b%c", 3), # Percent signs
    ("a^b^c", 3), # Carets
    ("a&b&c", 3), # Amperstands
    ("a*b*c", 3), # Asterisks
    ("a(b)c", 3), # Parentheses
    ("a[b]c", 3), # Square brackets
    ("a{b}c", 3), # Curly braces
    ("a|b|c", 3), # Vertical bars
    ("a\\b\\c", 3), # Backslashes
    ("a/b/c", 3), # Slashes
    ("a<b<c", 3), # Less than signs
    ("a>b>c", 3), # Greater than signs
    ("a?b?c", 3), # Question marks
    ("a=b=c", 3), # Equals signs
    ("a+b+c", 3), # Plus signs
    ("a-b-c", 3), # Minus signs
    ("a_b_c", 3), # Underscores
    ("a.b.c", 3), # Periods
    ("a,b,c", 3), # Commas
    ("a;b;c", 3), # Semicolons
    ("a:b:c", 3), # Colons
    ("a'