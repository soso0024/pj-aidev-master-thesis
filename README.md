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
2. Set API key: `export ANTHROPIC_API_KEY="your-key"` (or `GOOGLE_API_KEY` for Gemini)
3. Generate test: `python run_test_case_generator.py` (**requires Python 3.10+**)

**Note**: Datasets (HumanEval and HumanEvalPack) are automatically downloaded from HuggingFace 🤗 on first run. No manual dataset setup required!

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

## Statistical Significance Testing

For research paper validation, the analysis includes comprehensive statistical tests.

### Why Non-Parametric Tests?

Test generation results often violate normality assumptions (success rates are bounded 0-100%, distributions may be skewed). Non-parametric tests make no assumptions about the underlying distribution.

### Tests Performed

| Test                      | Purpose                              | Output               |
| ------------------------- | ------------------------------------ | -------------------- |
| **Kruskal-Wallis H-test** | Overall comparison of 3+ groups      | H-statistic, p-value |
| **Mann-Whitney U test**   | Pairwise comparison (non-parametric) | U-statistic, p-value |
| **Bonferroni correction** | Multiple comparison adjustment       | Adjusted α threshold |

#### Understanding the Values

- **H-statistic (Kruskal-Wallis)**: Measures the degree of separation between groups. Higher H indicates greater differences between configurations.
- **U-statistic (Mann-Whitney)**: Counts how often values from one group exceed values from another. Useful for determining which configuration performs better.
- **p-value**: Probability of observing the result by chance.
  - `p < 0.05 (*)`: Statistically significant
  - `p < 0.01 (**)`: Highly significant
  - `p < 0.001 (***)`: Very highly significant
- **Bonferroni α**: When comparing multiple pairs, divide α (0.05) by the number of comparisons to reduce false positives. For 3 comparisons: α = 0.05/3 ≈ 0.0167.

### Effect Size Measures

Statistical significance alone doesn't indicate practical importance. Effect size quantifies the magnitude of differences.

| Measure       | Type           | Interpretation                                                                  |
| ------------- | -------------- | ------------------------------------------------------------------------------- |
| **Cohen's d** | Parametric     | \|d\| < 0.2: negligible, 0.2-0.5: small, 0.5-0.8: medium, ≥0.8: large           |
| **Cliff's δ** | Non-parametric | \|δ\| < 0.147: negligible, 0.147-0.33: small, 0.33-0.474: medium, ≥0.474: large |

#### Interpreting Effect Sizes

- **Cohen's d**: Standardized difference between two means. `d = 0.5` means the groups differ by 0.5 standard deviations.
- **Cliff's δ**: Probability that a random value from group A exceeds a random value from group B, minus the reverse. `δ = 0.6` means 80% dominance (calculated as (1+δ)/2).

#### Example Interpretation

```
basic vs docstring: Cohen's d = 0.65 (medium), Cliff's δ = 0.42 (medium)
→ docstring configuration shows a medium practical improvement over basic
```

### Confidence Intervals

- **95% CI**: The true population mean lies within this range with 95% confidence.
- **Bootstrap method**: Resamples data 5,000 times to estimate the sampling distribution.
- **Overlapping CIs**: If confidence intervals overlap substantially, the difference may not be practically significant.

#### Example Interpretation

```
basic:     78.5% (95% CI: [72.1%, 84.9%])
docstring: 82.3% (95% CI: [77.1%, 87.5%])
→ CIs overlap (77.1-84.9%), suggesting the difference may not be robust
```

### Example Output

```
📊 Success Rate
────────────────────────────────────────────────────────────────────
  📈 Descriptive Statistics (with 95% CI):
  Config          |       Mean |      Std |          95% CI     |     n
  basic           |      78.5% |    12.3% | [72.1%, 84.9%]      |    50
  docstring       |      82.3% |    10.8% | [77.1%, 87.5%]      |    50

  🔬 Kruskal-Wallis H-test (Overall Comparison):
      H-statistic: 8.234
      p-value:     0.0163 *
      → Significant difference exists between groups

  🔍 Pairwise Mann-Whitney U Tests (Bonferroni α = 0.0167):
  basic vs docstring   |      890.5 |     0.0234 * |        Yes

  📏 Effect Sizes:
  basic vs docstring   |      0.342 |        small |      0.218 |        small
```

> **Requirement**: Statistical tests require at least 2 configuration types (e.g., basic, docstring, ast) for comparison.

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
├── dataset/                           # Legacy dataset files (optional)
│   ├── HumanEval.jsonl               # Legacy format (loaded from HuggingFace by default)
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
