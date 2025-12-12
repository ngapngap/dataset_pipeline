# -*- coding: utf-8 -*-
"""
Test Suite cho providers/base.py

Test perspectives table:
| Case ID  | Input / Precondition           | Perspective                  | Expected Result                          | Notes |
|----------|--------------------------------|------------------------------|------------------------------------------|-------|
| TC-N-01  | Valid API call                 | Equivalence - normal         | Returns response                         |       |
| TC-N-02  | Rate limit error               | Equivalence - rate limit     | Retries with backoff                     |       |
| TC-A-01  | Auth error                     | Boundary - no retry          | Fails immediately                        |       |
| TC-A-02  | Max retries exceeded           | Boundary - max retry         | Returns None                             |       |
| TC-B-01  | Timeout error                  | Boundary - network           | Retries                                  |       |
"""

import os
import sys
import time

import pytest
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from providers.base import BaseLLMProvider


class ConcreteProvider(BaseLLMProvider):
    """Concrete implementation for testing"""
    
    def __init__(self, *args, **kwargs):
        self._mock_response = kwargs.pop('mock_response', "Test response")
        self._mock_error = kwargs.pop('mock_error', None)
        self._call_count = 0
        super().__init__(*args, **kwargs)
    
    def _initialize(self):
        pass
    
    def _call_api(self, prompt: str):
        self._call_count += 1
        if self._mock_error:
            raise self._mock_error
        return self._mock_response
    
    @property
    def provider_name(self):
        return "TestProvider"


class TestBaseLLMProvider:
    """Test BaseLLMProvider class"""
    
    def test_initialization(self):
        """
        Given: Valid parameters
        When: Creating provider
        Then: Should initialize correctly
        """
        # Given/When
        provider = ConcreteProvider(
            api_key="test-key",
            model="test-model",
            max_tokens=1000,
            temperature=0.5
        )
        
        # Then
        assert provider.api_key == "test-key"
        assert provider.model == "test-model"
        assert provider.max_tokens == 1000
        assert provider.temperature == 0.5
        assert provider.consecutive_failures == 0
    
    def test_generate_success(self):
        """
        Given: Provider that returns response
        When: Calling generate
        Then: Should return response
        """
        # Given
        provider = ConcreteProvider(
            api_key="key",
            model="model",
            mock_response="Test output"
        )
        
        # When
        result = provider.generate("Test prompt")
        
        # Then
        assert result == "Test output"
        assert provider.consecutive_failures == 0
    
    def test_generate_with_retry_on_error(self):
        """
        Given: Provider that fails first then succeeds
        When: Calling generate
        Then: Should retry and return response
        """
        # Given
        call_count = [0]
        
        class RetryProvider(ConcreteProvider):
            def _call_api(self, prompt):
                call_count[0] += 1
                if call_count[0] < 2:
                    raise Exception("Temporary error")
                return "Success after retry"
        
        provider = RetryProvider(
            api_key="key",
            model="model",
            max_retries=3,
            retry_delay=0.01
        )
        
        # When
        result = provider.generate("Test")
        
        # Then
        assert result == "Success after retry"
        assert call_count[0] == 2
    
    def test_generate_max_retries_exceeded(self):
        """
        Given: Provider that always fails
        When: Calling generate
        Then: Should return None after max retries
        """
        # Given
        provider = ConcreteProvider(
            api_key="key",
            model="model",
            mock_error=Exception("Always fail"),
            max_retries=2,
            retry_delay=0.01
        )
        
        # When
        result = provider.generate("Test")
        
        # Then
        assert result is None
        assert provider._call_count == 2
    
    def test_categorize_error_rate_limit(self):
        """
        Given: Rate limit error
        When: Categorizing
        Then: Should identify as rate_limit
        """
        # Given
        provider = ConcreteProvider(api_key="key", model="model")
        
        # When/Then
        error_type, should_retry = provider._categorize_error(
            Exception("Rate limit exceeded (429)")
        )
        assert error_type == "rate_limit"
        assert should_retry is True
    
    def test_categorize_error_auth(self):
        """
        Given: Auth error
        When: Categorizing
        Then: Should identify as auth_error and not retry
        """
        # Given
        provider = ConcreteProvider(api_key="key", model="model")
        
        # When/Then
        error_type, should_retry = provider._categorize_error(
            Exception("401 Unauthorized - Invalid API key")
        )
        assert error_type == "auth_error"
        assert should_retry is False
    
    def test_categorize_error_timeout(self):
        """
        Given: Timeout error
        When: Categorizing
        Then: Should identify as timeout and retry
        """
        # Given
        provider = ConcreteProvider(api_key="key", model="model")
        
        # When/Then
        error_type, should_retry = provider._categorize_error(
            Exception("Connection timed out")
        )
        assert error_type == "timeout"
        assert should_retry is True
    
    def test_categorize_error_server(self):
        """
        Given: Server error
        When: Categorizing
        Then: Should identify as server_error and retry
        """
        # Given
        provider = ConcreteProvider(api_key="key", model="model")
        
        # When/Then
        error_type, should_retry = provider._categorize_error(
            Exception("500 Internal Server Error")
        )
        assert error_type == "server_error"
        assert should_retry is True
    
    def test_repr(self):
        """
        Given: Provider
        When: Getting repr
        Then: Should include name and model
        """
        # Given
        provider = ConcreteProvider(api_key="key", model="test-model")
        
        # When
        result = repr(provider)
        
        # Then
        assert "TestProvider" in result
        assert "test-model" in result
    
    def test_get_stats(self):
        """
        Given: Provider with activity
        When: Getting stats
        Then: Should return correct stats
        """
        # Given
        provider = ConcreteProvider(
            api_key="key",
            model="test-model",
            max_retries=5
        )
        provider.consecutive_failures = 2
        
        # When
        stats = provider.get_stats()
        
        # Then
        assert stats["provider"] == "TestProvider"
        assert stats["model"] == "test-model"
        assert stats["consecutive_failures"] == 2
        assert stats["max_retries"] == 5

