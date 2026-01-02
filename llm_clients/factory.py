#!/usr/bin/env python3
"""
LLM Client Factory

Provides factory functions for creating LLM clients based on provider configuration.
"""

from typing import Any, Optional

from .base import LLMClient
from .anthropic_client import AnthropicClient
from .gemini_client import GeminiClient
from .openai_client import OpenAIClient


# Provider to client class mapping
PROVIDER_CLIENTS = {
    "anthropic": AnthropicClient,
    "gemini": GeminiClient,
    "openai": OpenAIClient,
}


def create_client(
    provider: str, api_key: Optional[str] = None
) -> LLMClient:
    """
    Create an LLM client for the specified provider.
    
    Args:
        provider: Provider name ('anthropic', 'gemini', 'openai')
        api_key: Optional API key. If not provided, will use environment variable.
        
    Returns:
        LLMClient instance for the specified provider
        
    Raises:
        ValueError: If the provider is not supported
    """
    if provider not in PROVIDER_CLIENTS:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {list(PROVIDER_CLIENTS.keys())}"
        )
    
    client_class = PROVIDER_CLIENTS[provider]
    return client_class(api_key=api_key)


def create_clients_for_models(
    models_config: dict[str, Any],
    selected_models: list[str],
    anthropic_api_key: Optional[str] = None,
) -> dict[str, LLMClient]:
    """
    Create LLM clients for all providers needed by the selected models.
    
    Args:
        models_config: Model configuration dictionary from models_config.json
        selected_models: List of model names to create clients for
        anthropic_api_key: Optional Anthropic API key (passed directly)
        
    Returns:
        Dictionary mapping provider names to client instances
    """
    # Determine which providers are needed
    providers_needed = set()
    for model in selected_models:
        if model in models_config["models"]:
            provider = models_config["models"][model].get("provider", "anthropic")
            providers_needed.add(provider)
    
    # Create clients for each provider
    clients = {}
    for provider in providers_needed:
        if provider == "anthropic":
            clients[provider] = create_client(provider, anthropic_api_key)
        else:
            clients[provider] = create_client(provider)
    
    return clients


def get_client_for_model(
    clients: dict[str, LLMClient],
    model: str,
    models_config: dict[str, Any],
) -> LLMClient:
    """
    Get the appropriate client for a specific model.
    
    Args:
        clients: Dictionary of initialized clients
        model: Model name
        models_config: Model configuration dictionary
        
    Returns:
        LLMClient instance for the model's provider
        
    Raises:
        ValueError: If no client is available for the model's provider
    """
    provider = models_config["models"][model].get("provider", "anthropic")
    
    if provider not in clients:
        raise ValueError(
            f"No client available for provider '{provider}' "
            f"(required by model '{model}')"
        )
    
    client = clients[provider]
    if not client.is_available:
        raise ValueError(
            f"{provider.title()} API key required for model '{model}'"
        )
    
    return client

