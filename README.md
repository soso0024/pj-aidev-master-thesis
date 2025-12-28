# HumanEval Test Case Generator

Automatically generates comprehensive pytest test cases for HumanEval problems using multiple LLM providers with evaluation, error fixing, and detailed analysis.

## Features

- **Multi-model support**: Claude (Opus, Sonnet, Haiku) and Ollama models
- **Automatic evaluation**: Pytest execution with LLM-powered error fixing
- **Batch processing**: Generate tests for multiple problems simultaneously
- **Comprehensive analysis**: 11 visualizations with model comparisons
- **Cost tracking**: Token usage and API cost monitoring
- **Coverage analysis**: Test coverage percentage tracking

## Quick Start

1. Install dependencies: `uv sync` or `pip install -r requirements.txt`
2. Set API key: `export ANTHROPIC_API_KEY="your-key"`
3. Generate test: `python run_test_case_generator.py` (**requires Python 3.10+**)

## Supported Models

Models configured in `models_config.json`:

- **Claude Opus 4.1** - Most capable, highest cost
- **Claude Sonnet 4.5** - Latest Sonnet, best balance
- **Claude Sonnet 4** - Balanced performance/cost
- **Claude 3.5 Haiku** - Fast, low cost
- **Claude 3 Haiku** - Legacy fast model
- **Ollama GPT-OSS** - Local, free (requires Ollama)

## Usage

### Single Test Generation

```bash
# Random problem
python run_test_case_generator.py

# Specific problem with best model
python run_test_case_generator.py --task-id "HumanEval/0" --models claude-sonnet-4-5

# Full context generation
python run_test_case_generator.py --include-docstring --include-ast
```

### Analysis & Visualization

```bash
# Generate 11 analysis plots per model (requires Python 3.8.20+)
python run_analysis.py

# Cross-model comparison (requires Python 3.8.20+)
cd all_comparison
python best_model_comparison.py
python comprehensive_table_generator.py
```

Creates visualizations in `[model]_viz/` folders:

- Success rates and coverage analysis
- Cost vs. performance metrics
- Algorithm complexity analysis
- Model comparison charts

## Project Structure

```
├── analysis/                    # Analysis modules
├── all_comparison/              # Cross-model comparison tools
├── batch/                       # Batch processing
├── dataset/                     # HumanEval dataset files
├── generated_tests_[model]/     # Model-specific test outputs
├── [model]_viz/                # Model-specific visualizations
├── tests/                       # Test suite (52 tests)
├── run_test_case_generator.py   # Main script
├── run_analysis.py             # Generate visualizations
├── models_config.json          # Model configuration
└── requirements.txt
```

## File Outputs

- **Test files**: `test_humaneval_X_[config]_[status].py`
- **Statistics**: `test_humaneval_X_[config]_[status].stats.json`
- **Visualizations**: 11 plots per model in `[model]_viz/`
- **Comparisons**: Cross-model analysis in `all_comparison/`

## Running Tests

```bash
# Run generated tests
cd generated_tests_[model]
pytest test_humaneval_0.py -v --cov

# Run test suite
pytest tests/ -v
```

## Cost Guide

| Model             | Input/1K | Output/1K | Use Case         |
| ----------------- | -------- | --------- | ---------------- |
| Claude Opus 4.1   | $0.015   | $0.075    | Complex problems |
| Claude Sonnet 4.5 | $0.003   | $0.015    | Latest, best     |
| Claude Sonnet 4   | $0.003   | $0.015    | Balanced choice  |
| Claude 3.5 Haiku  | $0.00025 | $0.00125  | Fast iteration   |
| Claude 3 Haiku    | $0.00025 | $0.00125  | Legacy fast      |
| Ollama GPT-OSS    | Free     | Free      | Local testing    |

## Requirements

- **Python 3.10+** required for test generation (`run_test_case_generator.py`)
- **Python 3.8.20+** required for analysis scripts (`run_analysis.py` and `all_comparison/`)
- `uv sync` or `pip install -r requirements.txt`
- Anthropic API key for Claude models
- Ollama running locally (for local models)

> **Note**: Due to different Python version requirements, you may need separate virtual environments for test generation (3.10+) and analysis (3.8.20+).
