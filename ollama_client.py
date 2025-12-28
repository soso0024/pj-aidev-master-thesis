"""
Ollama client wrapper that mimics the Anthropic API interface for compatibility.
"""

import json
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class Usage:
    """Token usage information."""

    input_tokens: int
    output_tokens: int


@dataclass
class MessageContent:
    """Message content."""

    text: str


@dataclass
class Message:
    """Response message."""

    content: List[MessageContent]
    usage: Usage


class Messages:
    """Messages API wrapper for Ollama."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/generate"

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 2000,
        temperature: float = 0.0,
        **kwargs,
    ) -> Message:
        """Create a completion using Ollama API.

        Args:
            model: The model name to use (e.g., "llama2", "mistral", etc.)
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            **kwargs: Additional parameters (ignored for compatibility)

        Returns:
            Message object with content and usage information
        """
        # Convert messages to a single prompt
        # Ollama expects a simple prompt string, not chat messages
        prompt = self._convert_messages_to_prompt(messages)

        # Prepare the request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            # Make the request to Ollama
            response = requests.post(self.api_endpoint, json=payload, timeout=60)
            response.raise_for_status()

            data = response.json()

            # Extract the generated text
            generated_text = data.get("response", "")

            # Handle case where response is empty but thinking contains content
            if not generated_text and "thinking" in data:
                thinking_content = data.get("thinking", "")
                # Try to extract code from thinking content
                generated_text = self._extract_code_from_thinking(thinking_content)

            # Apply thinking mode cleaning regardless
            generated_text = self._remove_thinking_mode(generated_text)

            # Remove thinking tags if present (common in some models like deepseek-r1)
            if "<think>" in generated_text and "</think>" in generated_text:
                # Extract content after </think>
                parts = generated_text.split("</think>")
                if len(parts) > 1:
                    generated_text = parts[1].strip()

            # Token counting for Ollama (simplified since it's free)
            # We just return 0 for both since Ollama is free and we don't need accurate counts
            usage = Usage(input_tokens=0, output_tokens=0)

            content = [MessageContent(text=generated_text)]

            return Message(content=content, usage=usage)

        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API request failed: {str(e)}") from e
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse Ollama response: {str(e)}") from e

    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt string."""
        prompt_parts = []

        for message in messages:
            role = message.get("role")
            content = message.get("content", "")

            if role and content:
                # Include role information for better context
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"Human: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")

        if not prompt_parts:
            return ""

        # Join all messages with newlines for context
        full_prompt = "\n\n".join(prompt_parts)

        # Add a final prompt indicator if the last message was from user
        if messages and messages[-1].get("role") == "user":
            full_prompt += "\n\nAssistant:"

        return full_prompt

    def _remove_thinking_mode(self, text: str) -> str:
        """Remove thinking mode patterns from generated text.

        This method handles various thinking patterns such as:
        - "Thinking..." ... "...done thinking."
        - Multiple thinking blocks
        - Nested thinking patterns

        Args:
            text: The raw generated text that may contain thinking patterns

        Returns:
            Cleaned text with thinking patterns removed
        """
        import re

        # Pattern 1: "Thinking..." ... "...done thinking."
        thinking_pattern = r"Thinking\.{3}.*?\.{3}done thinking\."
        text = re.sub(thinking_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

        # Pattern 2: Alternative thinking patterns
        # "Thinking..." ... "...thinking complete."
        alt_thinking_pattern = r"Thinking\.{3}.*?\.{3}thinking complete\."
        text = re.sub(alt_thinking_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

        # Pattern 3: Single line thinking patterns
        # "Thinking: ..." (single line)
        single_thinking_pattern = r"Thinking:.*?\n"
        text = re.sub(single_thinking_pattern, "", text, flags=re.IGNORECASE)

        # Clean up any remaining artifacts
        # Remove multiple consecutive newlines
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def _extract_code_from_thinking(self, thinking_content: str) -> str:
        """Extract executable code from thinking content.

        This method tries to find and extract Python code blocks from
        the thinking content when the response field is empty.

        Args:
            thinking_content: The thinking content from the model

        Returns:
            Extracted Python code or empty string if no valid code found
        """
        import re

        # Pattern 1: Look for code blocks marked with ```python or ```
        python_code_pattern = r"```(?:python)?\s*\n(.*?)\n```"
        matches = re.findall(
            python_code_pattern, thinking_content, re.DOTALL | re.IGNORECASE
        )

        if matches:
            # Combine all code blocks
            extracted_code = "\n\n".join(matches)
            # Clean up any remaining artifacts
            return extracted_code.strip()

        # Pattern 2: Look for import statements and test functions
        # This is more aggressive - extract Python-like code
        lines = thinking_content.split("\n")
        code_lines = []
        in_code_block = False

        for line in lines:
            stripped = line.strip()

            # Start collecting if we see Python keywords
            if (
                stripped.startswith("import ")
                or stripped.startswith("from ")
                or stripped.startswith("def test_")
                or stripped.startswith("def ")
                or stripped.startswith("@pytest")
            ):
                in_code_block = True

            # Continue if we're in a code block and line looks like Python
            if in_code_block:
                # Skip lines that are clearly explanatory text
                if (
                    not stripped
                    or stripped.startswith("import ")
                    or stripped.startswith("from ")
                    or stripped.startswith("def ")
                    or stripped.startswith("@")
                    or stripped.startswith("assert ")
                    or stripped.startswith("pytest.")
                    or stripped.startswith("    ")  # indented lines
                    or stripped.startswith("\t")  # tab indented
                    or stripped in ["", "pass"]
                ):
                    code_lines.append(line)
                elif len(stripped) > 0 and not any(
                    word in stripped.lower()
                    for word in [
                        "so ",
                        "the ",
                        "this ",
                        "we ",
                        "let",
                        "now",
                        "then",
                        "here",
                        "that",
                    ]
                ):
                    # Likely still code if it doesn't contain common English words
                    code_lines.append(line)
                else:
                    # Probably moved to explanation, stop collecting
                    in_code_block = False

        if code_lines:
            extracted_code = "\n".join(code_lines).strip()
            # Basic validation - should have import and at least one test function
            if "import" in extracted_code and "def test_" in extracted_code:
                return extracted_code

        # Pattern 3: Last resort - return the original content and let cleaning handle it
        return thinking_content


class Ollama:
    """Ollama client that mimics Anthropic's interface."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize Ollama client.

        Args:
            base_url: Base URL for Ollama API (default: http://localhost:11434)
        """
        self.base_url = base_url
        self.messages = Messages(base_url)

    def test_connection(self) -> bool:
        """Test if Ollama server is accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> List[str]:
        """List available models on the Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except (requests.exceptions.RequestException, json.JSONDecodeError):
            return []
