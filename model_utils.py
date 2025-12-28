"""
Utility functions for model configuration management.
"""

import json
from typing import Dict, List, Any
from pathlib import Path


def load_model_config(config_path: str = "models_config.json") -> Dict[str, Any]:
    """Load model configuration from JSON file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in model configuration file: {e}")


def get_available_models(config_path: str = "models_config.json") -> List[str]:
    """Get list of available model names."""
    config = load_model_config(config_path)
    return list(config["models"].keys())


def get_default_model(config_path: str = "models_config.json") -> str:
    """Get the default model name."""
    config = load_model_config(config_path)
    return config["default_model"]
