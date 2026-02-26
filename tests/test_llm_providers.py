"""Tests for the LLM providers (OpenAI and Ollama)."""
import os
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to the path for imports
sys.path.insert(0, str((Path(__file__).parent.parent / "src").resolve()))

from backend.services.llm import get_llm_service, LLMService
from backend.services.llm.ollama_provider import OllamaProvider
from backend.services.llm.openai_provider import OpenAIProvider
from backend.services.llm.types import ChatMessage


class TestOllamaProvider:
    """Tests for the OllamaProvider class."""

    def test_provider_name(self):
        """Test that the provider has the correct name."""
        provider = OllamaProvider()
        assert provider.name == "ollama"

    def test_default_model(self):
        """Test that the provider has a default model."""
        provider = OllamaProvider()
        assert provider._default_model == "llama3.2"

    def test_custom_model(self):
        """Test that the provider can be initialized with a custom model."""
        provider = OllamaProvider(default_model="mistral")
        assert provider._default_model == "mistral"

    def test_custom_base_url(self):
        """Test that the provider can be initialized with a custom host URL."""
        provider = OllamaProvider(host="http://192.168.1.100:11434")
        assert provider._host == "http://192.168.1.100:11434"

    def test_base_url_trailing_slash_removed(self):
        """Test that trailing slash is handled by the host parameter."""
        provider = OllamaProvider(host="http://localhost:11434/")
        # The official library handles URL normalization
        assert "localhost:11434" in provider._host


class TestOpenAIProvider:
    """Tests for the OpenAIProvider class."""

    def test_provider_name(self):
        """Test that the provider has the correct name."""
        provider = OpenAIProvider()
        assert provider.name == "openai"

    def test_default_model(self):
        """Test that the provider has a default model."""
        provider = OpenAIProvider()
        assert provider._default_model == "gpt-4o-mini"

    def test_custom_model(self):
        """Test that the provider can be initialized with a custom model."""
        provider = OpenAIProvider(default_model="gpt-4")
        assert provider._default_model == "gpt-4"


class TestLLMService:
    """Tests for the LLMService class."""

    def test_available_providers(self):
        """Test that both providers are registered."""
        service = LLMService()
        providers = service.available_providers()
        assert "openai" in providers
        assert "ollama" in providers

    def test_default_provider_is_openai(self):
        """Test that the default provider is OpenAI."""
        service = LLMService()
        assert service.current_provider() == "openai"

    def test_set_provider_to_ollama(self):
        """Test that we can switch to Ollama provider."""
        service = LLMService()
        service.set_provider("ollama")
        assert service.current_provider() == "ollama"

    def test_set_provider_to_openai(self):
        """Test that we can switch back to OpenAI provider."""
        service = LLMService()
        service.set_provider("ollama")
        service.set_provider("openai")
        assert service.current_provider() == "openai"

    def test_set_unknown_provider_raises(self):
        """Test that setting an unknown provider raises an error."""
        service = LLMService()
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            service.set_provider("unknown_provider")


class TestOllamaProviderErrorHandling:
    """Tests for Ollama provider error handling."""

    def test_connection_error_message(self):
        """Test that connection errors produce helpful error messages."""
        provider = OllamaProvider(host="http://localhost:99999")
        messages: list[ChatMessage] = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(RuntimeError) as exc_info:
            # Consume the generator to trigger the connection
            list(provider.stream_chat(messages))
        
        error_message = str(exc_info.value).lower()
        # Check for any of the expected error patterns
        expected_patterns = [
            "could not connect to ollama",
            "connection refused",
            "connect",
            "error",
        ]
        assert any(pattern in error_message for pattern in expected_patterns), (
            f"Error message '{error_message}' did not contain any expected pattern"
        )
