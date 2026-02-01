# HumanEval Test Case Generator

Automatically generates comprehensive pytest test cases for HumanEval problems using multiple LLM providers with evaluation, error fixing, and detailed analysis.

## Features

- **Multi-model support**: Claude (Opus, Sonnet, Haiku), Gemini, OpenAI models
- **Multiple datasets**: HumanEval and HumanEvalPack support (loaded from HuggingFace 🤗)
- **Automatic evaluation**: Pytest execution with LLM-powered error fixing
- **Batch processing**: Generate tests for multiple problems simultaneously
- **Prompt engineering**: 4 different prompt strategies for comparison
- **Comprehensive analysis**: Dataset-aware visualizations with model comparisons
- **Cost tracking**: Token usage and API cost monitoring
- **Coverage analysis**: Test coverage percentage tracking

## Quick Start

1. Install dependencies: `uv sync` or `pip install -r requirements.txt`
2. Set API key: `export ANTHROPIC_API_KEY="your-key"` (or `GOOGLE_API_KEY` for Gemini, `OPENAI_API_KEY` for GPT)
3. Generate test: `python run_test_case_generator.py` (**requires Python 3.10+**)

**Note**: Datasets (HumanEval and HumanEvalPack) are automatically downloaded from HuggingFace 🤗 on first run. No manual dataset setup required!

## Supported Models

Models configured in `models_config.json`:

### Claude Models (Anthropic)

- **Claude Opus 4.5** - Most capable, highest cost
- **Claude Sonnet 4.5** - Best balance of intelligence and speed
- **Claude Haiku 4.5** - Fast and capable

### Gemini Models (Google)

- **Gemini 3 Pro Preview** - Latest preview model
- **Gemini 2.5 Flash** - Fast and capable (Free in some tiers)

### GPT Models (OpenAI)

- **GPT-5.2** - High intelligence
- **GPT-5.1** - Balanced
- **GPT-5 Mini** - Small and fast
- **GPT-5 Nano** - Extremely fast and lightweight

**Default Model**: `gemini-2.5-flash`

## Usage

### Single Test Generation

```bash
# Random problem with default model (HumanEval)
python run_test_case_generator.py

# Specific problem with specific model (unified ID format - recommended)
python run_test_case_generator.py --task-id 0 --models claude-sonnet-4-5

# With Gemini model
python run_test_case_generator.py --task-id 5 --models gemini-2.5-flash

# Full context generation
python run_test_case_generator.py --task-id 10 --include-docstring --include-ast

# Use HumanEvalPack dataset (same unified ID format!)
python run_test_case_generator.py --dataset-type humanevalpack --task-id 0

# Legacy format also supported
python run_test_case_generator.py --task-id "HumanEval/0"
python run_test_case_generator.py --dataset-type humanevalpack --task-id "Python/0"
```

### Batch Processing

```bash
# Generate tests for problems 0-10 (works for both HumanEval and HumanEvalPack!)
cd batch
python run_batch_test_case_generator.py --start 0 --end 10

# With specific model
python run_batch_test_case_generator.py --start 0 --end 10 --models claude-haiku-4-5

# HumanEvalPack dataset (same command format!)
python run_batch_test_case_generator.py --start 0 --end 10 --dataset-type humanevalpack

# Specific task IDs (unified format)
python run_batch_test_case_generator.py --task-ids "0,5,10,15"

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

# With specific results directory
python run_analysis.py --results-dir data/generated_tests_claude-haiku-4-5/ --output-dir vizs/
```

Creates visualizations in `vizs/` folder:

- Success rates and coverage analysis
- Cost vs. performance metrics
- Algorithm complexity analysis
- Dataset-aware problem classification
- **Efficiency metrics comparison** (CCE-C0, CCE-C1, SCCE)

## Efficiency Metrics

The analysis includes custom efficiency metrics for evaluating cost-performance trade-offs:

| Metric     | Formula                                     | Description                          |
| ---------- | ------------------------------------------- | ------------------------------------ |
| **CCE-C0** | C0 Coverage / (Cost × 1000)                 | Statement coverage efficiency        |
| **CCE-C1** | C1 Coverage / (Cost × 1000)                 | Branch coverage efficiency           |
| **SCCE**   | Success × (0.3×C0 + 0.7×C1) / (Cost × 1000) | Success-weighted coverage efficiency |

### Metric Interpretation

- **CCE-C0 / CCE-C1**: Higher values indicate better cost-efficiency. A value of 10.0 means achieving 10% coverage per $0.001 spent.
- **SCCE**: Combines success rate with weighted coverage. Only successful test cases contribute to the score, making it the most comprehensive metric.
- **Weighted Coverage**: `0.3×C0 + 0.7×C1` - C1 (branch coverage) is weighted higher because achieving C1 implies C0 is also achieved (C1 ⊃ C0).

### Example Interpretation

| Model         | CCE-C0 | CCE-C1 | SCCE | Interpretation                                         |
| ------------- | ------ | ------ | ---- | ------------------------------------------------------ |
| Claude Haiku  | 7.1    | 8.0    | 7.73 | High efficiency, good for budget-limited               |
| Claude Sonnet | 3.2    | 3.5    | 3.4  | Lower efficiency but may have higher absolute coverage |

> **Note**: Compare SCCE across models to identify the best cost-performance trade-off for your use case.


## Project Structure

```
├── analysis/                           # Analysis modules
│   ├── analysis_reporter.py           # Report generation
│   ├── cross_model_plots.py           # Model comparison plots
│   ├── data_loader.py                 # Data loading utilities
│   ├── dataset_aware_plots.py         # Dataset-specific visualizations
│   ├── humanevalpack_plots.py         # HumanEvalPack specific visualizations
│   ├── problem_classifier.py          # Problem classification logic
│   └── traditional_plots.py           # General visualizations
├── batch/                             # Batch processing
│   ├── README.md                      # Batch processing documentation
│   └── run_batch_test_case_generator.py
├── data/                              # Generated test outputs
│   └── generated_tests_[dataset]_[model]/
├── evaluator/                         # Test evaluation logic
├── generator/                         # Test case generation logic
├── llm_clients/                       # LLM client implementations
├── problem_classification/            # Detailed classification data
├── prompts/                           # Prompt templates
│   ├── basic.txt                     # Basic prompt
│   ├── docstring.txt                 # With docstring
│   ├── ast.txt                       # With AST
│   ├── docstring_ast.txt             # Full context
│   └── README.md                     # Prompt documentation
├── utils/                             # detailed utilities
├── vizs/                             # Analysis visualizations
├── config.py                         # Configuration settings
├── model_utils.py                    # Model utility functions
├── prompt_engineering_comparison.py   # Compare prompt strategies
├── remove_duplicates.py              # Utility to remove duplicate files
├── run_test_case_generator.py        # Main script
├── run_analysis.py                   # Generate visualizations
├── run_cross_model_analysis.py       # Cross-model analysis script
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

| Model | Input/1K | Output/1K | Use Case |
| ----- | -------- | --------- | -------- |
| Claude Opus 4.5 | $0.005 | $0.025 | Complex problems |
| Claude Sonnet 4.5 | $0.003 | $0.015 | Best balance |
| Claude Haiku 4.5 | $0.001 | $0.005 | Fast, capable |
| Gemini 3 Pro Preview | $0.002 | $0.012 | High capability |
| Gemini 2.5 Flash | Free* | Free* | Fast & efficient |
| GPT-5.2 | $0.00175 | $0.014 | High intelligence |
| GPT-5.1 | $0.00125 | $0.010 | Balanced |
| GPT-5 Mini | $0.00025 | $0.002 | Small & fast |
| GPT-5 Nano | $0.00005 | $0.0004 | Extremely lightweight |

\*Free in some tiers / preview

## Requirements

- **Python 3.10+** required for test generation (`run_test_case_generator.py`)
- **Python 3.8.20+** required for analysis scripts (`run_analysis.py`)
- `uv sync` or `pip install -r requirements.txt`
- **Internet connection** for first run (to download datasets from HuggingFace 🤗)
- API keys:
  - `ANTHROPIC_API_KEY` for Claude models
  - `GOOGLE_API_KEY` for Gemini models
  - `OPENAI_API_KEY` for GPT models

> **Note**:
>
> - Datasets are automatically cached after first download
> - Due to different Python version requirements, you may need separate virtual environments for test generation (3.10+) and analysis (3.8.20+)

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
