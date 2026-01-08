#!/usr/bin/env python3
"""
OpenAI GPT LLM Client

Provides integration with OpenAI's GPT API.
"""

import os
from typing import Optional

from openai import OpenAI

from .base import LLMClient, LLMResponse


class OpenAIClient(LLMClient):
    """LLM client for OpenAI GPT models."""

    ENV_KEY = "OPENAI_API_KEY"

    # OpenAI-specific parameters for reproducibility
    DEFAULT_TOP_P = 1.0
    DEFAULT_FREQUENCY_PENALTY = 0.0
    DEFAULT_PRESENCE_PENALTY = 0.0
    DEFAULT_SEED = 42

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key. If None, will try to get from OPENAI_API_KEY env var.
        """
        super().__init__(api_key)

    def _initialize_client(self) -> None:
        """Initialize the OpenAI client."""
        key = self.api_key or os.getenv(self.ENV_KEY)
        if key:
            self._client = OpenAI(api_key=key)
        else:
            self._client = None

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "openai"

    def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int = LLMClient.DEFAULT_MAX_TOKENS,
        temperature: float = LLMClient.DEFAULT_TEMPERATURE,
    ) -> LLMResponse:
        """
        Generate a response using OpenAI GPT.

        Args:
            prompt: The input prompt
            model: The GPT model identifier (e.g., 'gpt-4.1')
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            LLMResponse with the generated content
        """
        self._check_availability()

        try:
            # GPT-5+ models use max_completion_tokens instead of max_tokens
            # Determine which parameter to use based on model name
            uses_max_completion_tokens = any(
                model.startswith(prefix) for prefix in ["gpt-5", "o1", "o3"]
            )

            # Build the API call parameters
            api_params = {
                "model": model,
                "temperature": temperature,
                "top_p": self.DEFAULT_TOP_P,
                "frequency_penalty": self.DEFAULT_FREQUENCY_PENALTY,
                "presence_penalty": self.DEFAULT_PRESENCE_PENALTY,
                "seed": self.DEFAULT_SEED,
                "messages": [{"role": "user", "content": prompt}],
            }

            # Add the appropriate max tokens parameter
            if uses_max_completion_tokens:
                api_params["max_completion_tokens"] = max_tokens
            else:
                api_params["max_tokens"] = max_tokens

            response = self._client.chat.completions.create(**api_params)

            return LLMResponse(
                content=response.choices[0].message.content,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model=model,
                provider=self.provider_name,
                raw_response=response,
            )

        except Exception as e:
            raise RuntimeError(f"Error generating response with OpenAI: {e}")
