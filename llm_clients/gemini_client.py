#!/usr/bin/env python3
"""
Google Gemini LLM Client

Provides integration with Google's Gemini API.
"""

import os
from typing import Optional

import google.generativeai as genai

from .base import LLMClient, LLMResponse


class GeminiClient(LLMClient):
    """LLM client for Google Gemini models."""

    ENV_KEY = "GEMINI_API_KEY"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini client.

        Args:
            api_key: Google API key. If None, will try to get from GEMINI_API_KEY env var.
        """
        super().__init__(api_key)

    def _initialize_client(self) -> None:
        """Initialize the Gemini client."""
        key = self.api_key or os.getenv(self.ENV_KEY)
        if key:
            genai.configure(api_key=key)
            self._client = genai
        else:
            self._client = None

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "gemini"

    def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int = LLMClient.DEFAULT_MAX_TOKENS,
        temperature: float = LLMClient.DEFAULT_TEMPERATURE,
    ) -> LLMResponse:
        """
        Generate a response using Gemini.

        Args:
            prompt: The input prompt
            model: The Gemini model identifier (e.g., 'gemini-2.5-flash-lite')
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            LLMResponse with the generated content

        Raises:
            RuntimeError: If the API call fails or content is blocked
        """
        self._check_availability()

        try:
            gemini_model = self._client.GenerativeModel(model)
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )

            # Check if response was blocked or has no candidates
            if not response.candidates:
                if response.prompt_feedback and hasattr(
                    response.prompt_feedback, "block_reason"
                ):
                    raise RuntimeError(
                        f"Gemini API blocked content: {response.prompt_feedback.block_reason}"
                    )
                else:
                    raise RuntimeError("Gemini API returned no candidates.")

            return LLMResponse(
                content=response.text,
                input_tokens=response.usage_metadata.prompt_token_count,
                output_tokens=response.usage_metadata.candidates_token_count,
                model=model,
                provider=self.provider_name,
                raw_response=response,
            )

        except Exception as e:
            if "blocked" in str(e).lower():
                raise RuntimeError(f"Gemini content blocked: {e}")
            raise RuntimeError(f"Error generating response with Gemini: {e}")
