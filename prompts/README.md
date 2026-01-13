# Prompts

This directory contains the prompts used in the research.

## Prompt Variants

### 1. `basic.txt`

**Basic prompt without docstring or AST**

- Only includes function signature and canonical implementation
- Minimal context for the LLM
- Used as baseline for comparison

**Command line flags:**

```bash
python run_test_case_generator.py --task-id Python/0
# (default behavior - no flags needed)
# Note: Uses HumanEvalPack dataset by default
```

### 2. `ast.txt`

**Prompt with AST representation**

- Includes AST (Abstract Syntax Tree) representation of the canonical solution
- Provides structural information about the code
- May help LLM understand code logic more deeply

**Command line flags:**

```bash
python run_test_case_generator.py --task-id Python/0 --include-ast
```

### 3. `docstring.txt`

**Prompt with docstring**

- Includes function signature with full docstring
- Provides natural language description of function behavior
- May improve test coverage by highlighting edge cases mentioned in docs

**Command line flags:**

```bash
python run_test_case_generator.py --task-id Python/0 --include-docstring
```

### 4. `docstring_ast.txt`

**Prompt with both docstring and AST**

- Maximum context: combines natural language description and structural information
- Most comprehensive prompt variant
- Tests whether combining both types of information improves results

**Command line flags:**

```bash
python run_test_case_generator.py --task-id Python/0 --include-docstring --include-ast
```

## Dataset Support

The system supports two datasets:

### HumanEvalPack (Default)

```bash
python run_test_case_generator.py --task-id Python/0 --dataset-type humanevalpack
# or simply (default):
python run_test_case_generator.py --task-id Python/0
```

- 164 Python problems with intentionally introduced bugs
- Each problem includes both canonical and buggy solutions
- Enables bug detection evaluation
- Task IDs: `Python/0` to `Python/163`

### HumanEval

```bash
python run_test_case_generator.py --task-id HumanEval/0 --dataset-type humaneval
```

- Original 164 problems from OpenAI's HumanEval
- Task IDs: `HumanEval/0` to `HumanEval/163`
- Focus on test generation quality without bug detection

**Shortened Task ID format:**
You can also use just the number (e.g., `--task-id 0`), which will automatically add the appropriate prefix based on the dataset type.

## Template Variables

The prompt templates use the following placeholders that are filled in at runtime:

### Test Generation Templates
- `{function_signature}`: Extracted function signature (e.g., `def has_close_elements(numbers: List[float], threshold: float) -> bool:`)
- `{canonical_solution}`: The reference implementation body
- `{function_with_docstring}`: Full function definition including docstring from the dataset
- `{ast_representation}`: AST dump of the canonical solution

### Fix Templates
- `{function_info}`: Function signature or full function with docstring (based on `include_docstring` flag)
- `{canonical_solution}`: The reference implementation body
- `{ast_section}`: Optional AST representation section (based on `include_ast` flag)
- `{test_code}`: Current test code with errors
- `{error_output}`: Pytest error output
- `{fix_attempt_line}`: Information about current fix attempt number

## Fix Prompt Templates

When generated tests fail to run, the system attempts to fix them automatically using specialized fix prompts. These prompts provide the LLM with context about the error and ask it to generate corrected code.

### 5. `fix_basic.txt`

**Basic fix prompt without docstring or AST**

- Includes function signature and canonical implementation
- Shows the current test code with errors
- Provides pytest error output
- Minimal additional context

### 6. `fix_docstring.txt`

**Fix prompt with docstring**

- Includes full function definition with docstring
- Emphasizes using documented behavior for fixes
- Helps understand expected behavior from natural language description

### 7. `fix_ast.txt`

**Fix prompt with AST representation**

- Includes AST (Abstract Syntax Tree) representation
- Provides structural information about the implementation
- Helps understand control flow and operations

### 8. `fix_docstring_ast.txt`

**Fix prompt with both docstring and AST**

- Maximum context for error fixing
- Combines natural language and structural understanding
- Most comprehensive fix prompt variant

## Template System Benefits

1. **Reproducibility**: Template hashes are stored in `.stats.json` files for experiment tracking
2. **Maintainability**: Prompts can be modified without touching code
3. **Experimentation**: Easy to test different prompt variations
4. **Consistency**: Ensures all experiments use the same prompt structure
5. **Documentation**: Templates serve as self-documenting prompt specifications

## Additional Features

### Bug Detection (HumanEvalPack only)

When using HumanEvalPack dataset, the system automatically evaluates whether generated tests can detect bugs:

- **Canonical Solution Test**: Runs tests against the correct implementation
- **Buggy Solution Test**: Runs tests against the buggy implementation
- **True Bug Detection**: Tests pass on canonical solution AND fail on buggy solution with assertion errors
- **False Positive Detection**: Tests fail on buggy solution but with non-assertion errors (timeout, syntax, etc.)

Bug detection results are stored in `.stats.json` files:
- `bug_detection_success`: Whether tests correctly detected the bug
- `canonical_solution_passed`: Whether tests passed on correct implementation
- `buggy_solution_failed`: Whether tests failed on buggy implementation
- `buggy_failure_type`: Type of failure (assertion, timeout, syntax, etc.)
- `bug_type`: Category of bug from HumanEvalPack

### Code Coverage

The system automatically measures code coverage for all generated tests:

- **C0 Coverage (Statement Coverage)**: Percentage of code lines executed
- **C1 Coverage (Branch Coverage)**: Percentage of code branches executed

Coverage metrics are stored in `.stats.json` files and used for analysis.

### Automatic Test Fixing

When generated tests fail to run, the system automatically attempts to fix them:

```bash
python run_test_case_generator.py --task-id Python/0 --max-fix-attempts 3
```

- Default: 3 fix attempts
- Uses the same prompt variant (basic/ast/docstring/docstring_ast) for fixes
- Tracks number of fix attempts used in `.stats.json`
- Can be disabled with `--disable-evaluation`

### Verbose Output Control

```bash
# Default: verbose output showing all fix attempts and errors
python run_test_case_generator.py --task-id Python/0

# Quiet mode: minimal output
python run_test_case_generator.py --task-id Python/0 --quiet-evaluation

# Disable prompt preview
python run_test_case_generator.py --task-id Python/0 --no-show-prompt
```
