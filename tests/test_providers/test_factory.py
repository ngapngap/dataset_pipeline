# -*- coding: utf-8 -*-
"""
Test Suite cho providers/factory.py

Test perspectives table:
| Case ID  | Input / Precondition           | Perspective                  | Expected Result                          | Notes |
|----------|--------------------------------|------------------------------|------------------------------------------|-------|
| TC-N-01  | Create Gemini provider         | Equivalence - normal         | GeminiProvider created                   |       |
| TC-N-02  | Create provider with preset    | Equivalence - preset         | CustomLLMProvider with preset URL        |       |
| TC-N-03  | Create provider with base_url  | Equivalence - custom         | CustomLLMProvider created                |       |
| TC-A-01  | Invalid provider name          | Boundary - invalid           | Raises ValueError                        |       |
| TC-A-02  | Create multiple providers      | Equivalence - batch          | List of providers created                |       |
"""

import os
import sys

import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from providers.factory import (
    create_provider,
    create_providers_from_config,
    get_available_providers,
    PROVIDER_REGISTRY,
    PROVIDER_PRESETS
)
from providers.base import BaseLLMProvider


class TestProviderRegistry:
    """Test provider registry"""
    
    def test_registry_has_standard_providers(self):
        """
        Given: PROVIDER_REGISTRY
        When: Checking contents
        Then: Should have all standard providers
        """
        # Then
        assert "gemini" in PROVIDER_REGISTRY
        assert "openai" in PROVIDER_REGISTRY
        assert "anthropic" in PROVIDER_REGISTRY
        assert "custom" in PROVIDER_REGISTRY
    
    def test_presets_has_common_providers(self):
        """
        Given: PROVIDER_PRESETS
        When: Checking contents
        Then: Should have common custom providers
        """
        # Then
        assert "ollama" in PROVIDER_PRESETS
        assert "groq" in PROVIDER_PRESETS
        assert "together" in PROVIDER_PRESETS
        assert "deepseek" in PROVIDER_PRESETS


class TestGetAvailableProviders:
    """Test get_available_providers function"""
    
    def test_returns_all_providers(self):
        """
        Given: Nothing
        When: Getting available providers
        Then: Should return combined list
        """
        # When
        providers = get_available_providers()
        
        # Then
        assert "gemini" in providers
        assert "openai" in providers
        assert "ollama" in providers
        assert "groq" in providers


class TestCreateProvider:
    """Test create_provider function"""
    
    @patch('providers.gemini.genai', create=True)
    def test_create_gemini_provider(self, mock_genai):
        """
        Given: Gemini provider name
        When: Creating provider
        Then: Should return GeminiProvider
        """
        # Given
        mock_genai.configure = MagicMock()
        mock_genai.GenerativeModel = MagicMock()
        
        # When
        provider = create_provider(
            provider_name="gemini",
            api_key="test-key",
            model="gemini-2.0-flash"
        )
        
        # Then
        assert provider.provider_name == "Gemini"
        assert provider.model == "gemini-2.0-flash"
    
    def test_create_invalid_provider_raises(self):
        """
        Given: Invalid provider name
        When: Creating provider
        Then: Should raise ValueError
        """
        # When/Then
        with pytest.raises(ValueError) as exc_info:
            create_provider(
                provider_name="invalid_provider",
                api_key="key",
                model="model"
            )
        
        assert "không hỗ trợ" in str(exc_info.value).lower() or "not" in str(exc_info.value).lower()
    
    def test_create_preset_provider(self):
        """
        Given: Preset provider name (ollama)
        When: Creating provider
        Then: Should return CustomLLMProvider with preset URL
        """
        # Given
        with patch('openai.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            # When
            provider = create_provider(
                provider_name="ollama",
                api_key="ollama",
                model="llama2"
            )
            
            # Then
            assert provider.base_url == "http://localhost:11434/v1"
    
    def test_create_custom_provider_with_base_url(self):
        """
        Given: Custom base_url
        When: Creating provider
        Then: Should return CustomLLMProvider with that URL
        """
        # Given
        with patch('openai.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            # When
            provider = create_provider(
                provider_name="custom",
                api_key="test-key",
                model="test-model",
                base_url="http://localhost:8000/v1"
            )
            
            # Then
            assert provider.base_url == "http://localhost:8000/v1"


class TestCreateProvidersFromConfig:
    """Test create_providers_from_config function"""
    
    @patch('providers.gemini.genai', create=True)
    def test_create_multiple_providers(self, mock_genai):
        """
        Given: Multiple API keys
        When: Creating providers
        Then: Should return list of providers
        """
        # Given
        mock_genai.configure = MagicMock()
        mock_genai.GenerativeModel = MagicMock()
        
        api_keys = ["key1", "key2", "key3"]
        
        # When
        providers = create_providers_from_config(
            provider_name="gemini",
            api_keys=api_keys,
            model="gemini-2.0-flash"
        )
        
        # Then
        assert len(providers) == 3
        for provider in providers:
            assert isinstance(provider, BaseLLMProvider)
    
    def test_create_empty_list_for_no_keys(self):
        """
        Given: Empty API keys list
        When: Creating providers
        Then: Should return empty list
        """
        # When
        providers = create_providers_from_config(
            provider_name="gemini",
            api_keys=[],
            model="test"
        )
        
        # Then
        assert providers == []

