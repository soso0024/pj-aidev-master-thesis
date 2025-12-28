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
python run_test_case_generator.py --task-id HumanEval/0
# (default behavior - no flags needed)
```

### 2. `ast.txt`

**Prompt with AST representation**

- Includes AST (Abstract Syntax Tree) representation of the canonical solution
- Provides structural information about the code
- May help LLM understand code logic more deeply

**Command line flags:**

```bash
python run_test_case_generator.py --task-id HumanEval/0 --include-ast
```

### 3. `docstring.txt`

**Prompt with docstring**

- Includes function signature with full docstring
- Provides natural language description of function behavior
- May improve test coverage by highlighting edge cases mentioned in docs

**Command line flags:**

```bash
python run_test_case_generator.py --task-id HumanEval/0 --include-docstring
```

### 4. `docstring_ast.txt`

**Prompt with both docstring and AST**

- Maximum context: combines natural language description and structural information
- Most comprehensive prompt variant
- Tests whether combining both types of information improves results

**Command line flags:**

```bash
python run_test_case_generator.py --task-id HumanEval/0 --include-docstring --include-ast
```

## Template Variables

The prompt templates use the following placeholders that are filled in at runtime:

- `{function_signature}`: Extracted function signature (e.g., `def has_close_elements(numbers: List[float], threshold: float) -> bool:`)
- `{canonical_solution}`: The reference implementation body
- `{function_with_docstring}`: Full function definition including docstring from the HumanEval dataset
- `{ast_representation}`: AST dump of the canonical solution
