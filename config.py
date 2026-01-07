"""
Central Configuration Module

This module provides centralized configuration management for the project,
including constants, default values, and path management.
"""

from pathlib import Path
from typing import Final

# ===== Project Paths =====
PROJECT_ROOT: Final[Path] = Path(__file__).parent
DATASET_DIR: Final[Path] = PROJECT_ROOT / "dataset"
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
PROMPTS_DIR: Final[Path] = PROJECT_ROOT / "prompts"
VISUALIZATIONS_DIR: Final[Path] = PROJECT_ROOT / "vizs"
BATCH_DIR: Final[Path] = PROJECT_ROOT / "batch"
ANALYSIS_DIR: Final[Path] = PROJECT_ROOT / "analysis"

# ===== Dataset Files =====
HUMANEVAL_JSONL: Final[Path] = DATASET_DIR / "HumanEval.jsonl"
HUMANEVAL_JSON: Final[Path] = DATASET_DIR / "HumanEval_formatted.json"
HUMANEVAL_YAML: Final[Path] = DATASET_DIR / "HumanEval_formatted.yaml"

# ===== Model Configuration =====
MODELS_CONFIG_FILE: Final[Path] = PROJECT_ROOT / "models_config.json"

# ===== Test Generation Constants =====

# Pytest execution constants
PYTEST_TIMEOUT_SECONDS: Final[int] = 60  # Timeout for pytest execution

# LLM API constants
DEFAULT_MAX_TOKENS: Final[int] = 8000  # Max tokens for LLM response
DEFAULT_TEMPERATURE: Final[float] = 0.0  # Temperature for deterministic responses

# Display and output constants
DISPLAY_LINE_LIMIT: Final[int] = 20  # Max lines to display without truncation
TRUNCATE_HEAD_LINES: Final[int] = 10  # Lines to show at start when truncating
TRUNCATE_TAIL_LINES: Final[int] = 10  # Lines to show at end when truncating

# Test file markers
GENERATED_TEST_MARKER: Final[str] = "# Generated test cases:\n"

# ===== Batch Processing Constants =====
DEFAULT_BATCH_START_ID: Final[int] = 0
DEFAULT_BATCH_END_ID: Final[int] = 50
DEFAULT_TASK_TIMEOUT: Final[int] = 300  # Timeout in seconds for each task

# ===== Evaluation Constants =====
DEFAULT_MAX_FIX_ATTEMPTS: Final[int] = 3

# ===== Analysis Constants =====
# Configuration order for consistent plotting
CONFIG_ORDER: Final[list[str]] = ["basic", "ast", "docstring", "docstring_ast"]

# Statistical significance thresholds
STATISTICAL_SIGNIFICANCE_ALPHA: Final[float] = 0.05
BONFERRONI_COMPARISON_THRESHOLD: Final[int] = (
    2  # Minimum comparisons for Bonferroni correction
)

# Bootstrap parameters
BOOTSTRAP_ITERATIONS: Final[int] = 5000
BOOTSTRAP_CONFIDENCE_LEVEL: Final[float] = 0.95

# ===== Visualization Constants =====
FIGURE_DPI: Final[int] = 300
DEFAULT_FIGURE_SIZE: Final[tuple[int, int]] = (10, 6)

# Color palettes for plots
COLOR_PALETTE_PRIMARY: Final[list[str]] = [
    "#3498db",  # Blue
    "#e74c3c",  # Red
    "#2ecc71",  # Green
    "#f39c12",  # Orange
]

COLOR_PALETTE_EXTENDED: Final[list[str]] = [
    "#3498db",  # Blue
    "#e74c3c",  # Red
    "#2ecc71",  # Green
    "#f39c12",  # Orange
    "#9b59b6",  # Purple
    "#1abc9c",  # Turquoise
    "#e67e22",  # Carrot
    "#34495e",  # Wet Asphalt
]

# ===== HumanEvalPack Constants =====
BUG_TYPES: Final[list[str]] = [
    "value_misuse",
    "function_misuse",
    "excess_logic",
    "variable_misuse",
    "operator_misuse",
    "missing_condition",
    "missing_logic",
]

# ===== Environment Variable Names =====
ENV_ANTHROPIC_API_KEY: Final[str] = "ANTHROPIC_API_KEY"
ENV_GOOGLE_API_KEY: Final[str] = "GOOGLE_API_KEY"
ENV_GEMINI_API_KEY: Final[str] = "GEMINI_API_KEY"
ENV_OPENAI_API_KEY: Final[str] = "OPENAI_API_KEY"


# ===== Helper Functions =====
def get_generated_tests_dir(model: str, dataset_type: str = "humaneval") -> Path:
    """
    Get the output directory path for generated tests.

    Args:
        model: Model name (e.g., 'claude-sonnet-4-5')
        dataset_type: Type of dataset ('humaneval' or 'humanevalpack')

    Returns:
        Path to the generated tests directory
    """
    if dataset_type == "humanevalpack":
        return DATA_DIR / f"generated_tests_humanevalpack_{model}"
    return DATA_DIR / f"generated_tests_{model}"


def ensure_directories_exist() -> None:
    """Create necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        VISUALIZATIONS_DIR,
        PROJECT_ROOT / "prompt_comparison_results",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
