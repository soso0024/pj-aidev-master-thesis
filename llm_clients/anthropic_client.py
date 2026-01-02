#!/usr/bin/env python3
"""
Anthropic Claude LLM Client

Provides integration with Anthropic's Claude API.
"""

import os
from typing import Optional

import anthropic

from .base import LLMClient, LLMResponse


class AnthropicClient(LLMClient):
    """LLM client for Anthropic Claude models."""
    
    ENV_KEY = "ANTHROPIC_API_KEY"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Anthropic client.
        
        Args:
            api_key: Anthropic API key. If None, will try to get from ANTHROPIC_API_KEY env var.
        """
        super().__init__(api_key)
    
    def _initialize_client(self) -> None:
        """Initialize the Anthropic client."""
        key = self.api_key or os.getenv(self.ENV_KEY)
        if key:
            self._client = anthropic.Anthropic(api_key=key)
        else:
            self._client = None
    
    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "anthropic"
    
    def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int = LLMClient.DEFAULT_MAX_TOKENS,
        temperature: float = LLMClient.DEFAULT_TEMPERATURE,
    ) -> LLMResponse:
        """
        Generate a response using Claude.
        
        Args:
            prompt: The input prompt
            model: The Claude model identifier (e.g., 'claude-sonnet-4-5-20250929')
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            LLMResponse with the generated content
        """
        self._check_availability()
        
        try:
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            
            return LLMResponse(
                content=response.content[0].text,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                model=model,
                provider=self.provider_name,
                raw_response=response,
            )
            
        except anthropic.APIError as e:
            raise RuntimeError(f"Anthropic API error: {e}")
        except Exception as e:
            raise RuntimeError(f"Error generating response with Claude: {e}")

