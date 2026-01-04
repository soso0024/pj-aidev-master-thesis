"""
File Utility Functions

Provides safe file I/O operations with error handling.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Union

from .exceptions import ProjectError


def ensure_dir_exists(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for the directory
        
    Raises:
        ProjectError: If directory creation fails
    """
    path = Path(directory)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        raise ProjectError(
            f"Failed to create directory: {directory}",
            details={"error": str(e)}
        ) from e


def safe_read_json(filepath: Union[str, Path]) -> dict[str, Any]:
    """
    Safely read a JSON file with error handling.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Parsed JSON data as dictionary
        
    Raises:
        ProjectError: If file reading or JSON parsing fails
    """
    path = Path(filepath)
    
    if not path.exists():
        raise ProjectError(
            f"File not found: {filepath}",
            details={"filepath": str(path)}
        )
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ProjectError(
            f"Invalid JSON in file: {filepath}",
            details={"filepath": str(path), "error": str(e)}
        ) from e
    except Exception as e:
        raise ProjectError(
            f"Failed to read file: {filepath}",
            details={"filepath": str(path), "error": str(e)}
        ) from e


def safe_write_json(filepath: Union[str, Path], data: dict[str, Any], indent: int = 2) -> None:
    """
    Safely write data to a JSON file with error handling.
    
    Args:
        filepath: Path to the JSON file
        data: Data to write
        indent: JSON indentation level
        
    Raises:
        ProjectError: If file writing fails
    """
    path = Path(filepath)
    
    # Ensure parent directory exists
    ensure_dir_exists(path.parent)
    
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except Exception as e:
        raise ProjectError(
            f"Failed to write file: {filepath}",
            details={"filepath": str(path), "error": str(e)}
        ) from e


def safe_read_text(filepath: Union[str, Path]) -> str:
    """
    Safely read a text file with error handling.
    
    Args:
        filepath: Path to the text file
        
    Returns:
        File contents as string
        
    Raises:
        ProjectError: If file reading fails
    """
    path = Path(filepath)
    
    if not path.exists():
        raise ProjectError(
            f"File not found: {filepath}",
            details={"filepath": str(path)}
        )
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise ProjectError(
            f"Failed to read file: {filepath}",
            details={"filepath": str(path), "error": str(e)}
        ) from e


def safe_write_text(filepath: Union[str, Path], content: str) -> None:
    """
    Safely write text to a file with error handling.
    
    Args:
        filepath: Path to the text file
        content: Content to write
        
    Raises:
        ProjectError: If file writing fails
    """
    path = Path(filepath)
    
    # Ensure parent directory exists
    ensure_dir_exists(path.parent)
    
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        raise ProjectError(
            f"Failed to write file: {filepath}",
            details={"filepath": str(path), "error": str(e)}
        ) from e

