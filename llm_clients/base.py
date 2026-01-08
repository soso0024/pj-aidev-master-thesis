#!/usr/bin/env python3
"""
Base LLM Client Interface

Defines the abstract interface for all LLM clients and common data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LLMResponse:
    """Represents a response from an LLM API."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str
    raw_response: Optional[Any] = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used in this response."""
        return self.input_tokens + self.output_tokens


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    # Default generation parameters
    DEFAULT_MAX_TOKENS = 8000
    DEFAULT_TEMPERATURE = 0.0

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM client.

        Args:
            api_key: API key for the provider. If None, will try to get from environment.
        """
        self.api_key = api_key
        self._client = None
        self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize the underlying API client."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'anthropic', 'gemini', 'openai')."""
        pass

    @property
    def is_available(self) -> bool:
        """Check if the client is properly initialized and available."""
        return self._client is not None

    @abstractmethod
    def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: The input prompt
            model: The model identifier to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 for deterministic)

        Returns:
            LLMResponse containing the generated content and metadata

        Raises:
            ValueError: If the client is not properly initialized
            RuntimeError: If the API call fails
        """
        pass

    def _check_availability(self) -> None:
        """Check if client is available, raise error if not."""
        if not self.is_available:
            raise ValueError(
                f"{self.provider_name.title()} API key required but client not initialized. "
                f"Please set the appropriate environment variable or provide the API key."
            )
