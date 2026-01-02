# HumanEval Test Case Generator

Automatically generates comprehensive pytest test cases for HumanEval problems using multiple LLM providers with evaluation, error fixing, and detailed analysis.

## Features

- **Multi-model support**: Claude (Opus, Sonnet, Haiku), Gemini, OpenAI models
- **Automatic evaluation**: Pytest execution with LLM-powered error fixing
- **Batch processing**: Generate tests for multiple problems simultaneously
- **Prompt engineering**: 4 different prompt strategies for comparison
- **Comprehensive analysis**: Dataset-aware visualizations with model comparisons
- **Cost tracking**: Token usage and API cost monitoring
- **Coverage analysis**: Test coverage percentage tracking

## Quick Start

1. Install dependencies: `uv sync` or `pip install -r requirements.txt`
2. Set API key: `export ANTHROPIC_API_KEY="your-key"` (or `GOOGLE_API_KEY` for Gemini)
3. Generate test: `python run_test_case_generator.py` (**requires Python 3.10+**)

## Supported Models

Models configured in `models_config.json`:

### Claude Models (Anthropic)

- **Claude Opus 4.1** - Most capable, highest cost
- **Claude Sonnet 4.5** - Latest Sonnet, best balance
- **Claude Sonnet 4** - Balanced performance/cost
- **Claude Haiku 4.5** - Latest Haiku, fast and capable
- **Claude 3.5 Haiku** - Fast, low cost
- **Claude 3 Haiku** - Legacy fast model

### Gemini Models (Google)

- **Gemini 3 Flash Preview** - Latest preview model
- **Gemini 2.5 Pro** - Most capable Gemini (free during preview)
- **Gemini 2.5 Flash** - Fast and capable (free during preview)
- **Gemini 2.5 Flash Lite** - Lightweight, default model (free)
- **Gemini 2.0 Flash Exp** - Experimental (free)
- **Gemini 1.5 Pro** - Stable production model
- **Gemini 1.5 Flash** - Fast production model

### Other Models

- **GPT-4.1** - OpenAI's latest model

**Default Model**: `gemini-2.5-flash-lite`

## Usage

### Single Test Generation

```bash
# Random problem with default model
python run_test_case_generator.py

# Specific problem with specific model
python run_test_case_generator.py --task-id "HumanEval/0" --models claude-sonnet-4-5

# With Gemini model
python run_test_case_generator.py --task-id "HumanEval/0" --models gemini-2.5-flash

# Full context generation
python run_test_case_generator.py --include-docstring --include-ast
```

### Batch Processing

```bash
# Generate tests for problems 0-10
cd batch
python run_batch_test_case_generator.py --start 0 --end 10

# With specific model
python run_batch_test_case_generator.py --start 0 --end 10 --models claude-haiku-4-5

# Multiple models for comparison
python run_batch_test_case_generator.py --start 0 --end 5 --models claude-sonnet-4-5 gemini-2.5-flash

# See batch/README.md for more options
```

### Prompt Engineering Comparison

```bash
# Compare different prompt strategies
python prompt_engineering_comparison.py --task-id "HumanEval/0"
```

### Analysis & Visualization

```bash
# Generate analysis plots (requires Python 3.8.20+)
python run_analysis.py
```

Creates visualizations in `vizs/` folder:

- Success rates and coverage analysis
- Cost vs. performance metrics
- Algorithm complexity analysis
- Dataset-aware problem classification

## Project Structure

```
├── analysis/                           # Analysis modules
│   ├── analysis_reporter.py           # Report generation
│   ├── data_loader.py                 # Data loading utilities
│   ├── dataset_aware_plots.py         # Dataset-specific visualizations
│   ├── problem_classifier.py          # Problem classification logic
│   └── traditional_plots.py           # General visualizations
├── batch/                             # Batch processing
│   ├── README.md                      # Batch processing documentation
│   └── run_batch_test_case_generator.py
├── data/                              # Generated test outputs
│   └── generated_tests_[dataset]_[model]/
├── dataset/                           # HumanEval dataset files
│   ├── HumanEval.jsonl               # Original format
│   ├── HumanEval_formatted.json      # Formatted version
│   └── HumanEval_formatted.yaml      # YAML version
├── prompts/                           # Prompt templates
│   ├── basic.txt                     # Basic prompt
│   ├── docstring.txt                 # With docstring
│   ├── ast.txt                       # With AST
│   ├── docstring_ast.txt             # Full context
│   └── README.md                     # Prompt documentation
├── prompt_comparison_results/         # Prompt engineering results
├── vizs/                             # Analysis visualizations
├── run_test_case_generator.py        # Main script
├── run_analysis.py                   # Generate visualizations
├── prompt_engineering_comparison.py   # Compare prompt strategies
├── models_config.json                # Model configuration
└── pyproject.toml                    # Project dependencies
```

## File Outputs

- **Test files**: `test_python_X_[config]_[status].py`
- **Statistics**: `test_python_X_[config]_[status].stats.json`
- **Visualizations**: Analysis plots in `vizs/`
- **Prompt results**: Comparison data in `prompt_comparison_results/`

## Running Tests

```bash
# Run generated tests
cd data/generated_tests_[dataset]_[model]
pytest test_python_0_*.py -v --cov

# Run specific test
pytest test_python_0_missing_logic_success.py -v
```

## Cost Guide

| Model                 | Input/1K  | Output/1K | Use Case             |
| --------------------- | --------- | --------- | -------------------- |
| Claude Opus 4.1       | $0.015    | $0.075    | Complex problems     |
| Claude Sonnet 4.5     | $0.003    | $0.015    | Best balance         |
| Claude Sonnet 4       | $0.003    | $0.015    | Balanced choice      |
| Claude Haiku 4.5      | $0.001    | $0.005    | Fast, capable        |
| Claude 3.5 Haiku      | $0.00025  | $0.00125  | Fast iteration       |
| Claude 3 Haiku        | $0.00025  | $0.00125  | Legacy fast          |
| Gemini 2.5 Pro        | Free\*    | Free\*    | Best Gemini          |
| Gemini 2.5 Flash      | Free\*    | Free\*    | Fast Gemini          |
| Gemini 2.5 Flash Lite | Free\*    | Free\*    | Default, lightweight |
| Gemini 1.5 Pro        | $0.00125  | $0.005    | Stable production    |
| Gemini 1.5 Flash      | $0.000075 | $0.0003   | Fast production      |
| GPT-4.1               | $0.002    | $0.008    | OpenAI latest        |

\*Free during preview period

## Requirements

- **Python 3.10+** required for test generation (`run_test_case_generator.py`)
- **Python 3.8.20+** required for analysis scripts (`run_analysis.py`)
- `uv sync` or `pip install -r requirements.txt`
- API keys:
  - `ANTHROPIC_API_KEY` for Claude models
  - `GOOGLE_API_KEY` for Gemini models
  - `OPENAI_API_KEY` for GPT models

> **Note**: Due to different Python version requirements, you may need separate virtual environments for test generation (3.10+) and analysis (3.8.20+).

## Environment Variables

```bash
# For Claude models
export ANTHROPIC_API_KEY="your-anthropic-key"

# For Gemini models
export GOOGLE_API_KEY="your-google-key"

# For OpenAI models
export OPENAI_API_KEY="your-openai-key"
```

## Prompt Strategies

Four prompt strategies available in `prompts/`:

1. **basic.txt** - Minimal context, function signature only
2. **docstring.txt** - Includes function docstring
3. **ast.txt** - Includes AST of canonical solution
4. **docstring_ast.txt** - Full context (docstring + AST)

Use `--include-docstring` and `--include-ast` flags to select strategy.
