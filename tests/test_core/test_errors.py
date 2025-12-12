# -*- coding: utf-8 -*-
"""
Test Suite cho core/errors.py

Test perspectives table:
| Case ID  | Input / Precondition           | Perspective                  | Expected Result                          | Notes |
|----------|--------------------------------|------------------------------|------------------------------------------|-------|
| TC-N-01  | Create PipelineError           | Equivalence - normal         | Error created with correct attrs         |       |
| TC-N-02  | Create ConfigurationError      | Equivalence - normal         | Non-recoverable error                    |       |
| TC-N-03  | Create RateLimitError          | Equivalence - normal         | Recoverable warning                      |       |
| TC-N-04  | ErrorSummary tracking          | Equivalence - normal         | Counts and stores correctly              |       |
| TC-A-01  | to_dict serialization          | Boundary - serialization     | All fields serialized                    |       |
| TC-A-02  | ErrorSummary max_recent        | Boundary - limit             | Keeps only max_recent errors             |       |
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.errors import (
    PipelineError,
    ConfigurationError,
    ValidationError,
    APIError,
    RateLimitError,
    AuthenticationError,
    NetworkError,
    FileIOError,
    ParseError,
    ErrorSummary,
    ErrorSeverity,
    ErrorCategory
)


class TestPipelineError:
    """Test base PipelineError class"""
    
    def test_basic_creation(self):
        """
        Given: Error message
        When: Creating PipelineError
        Then: Should have correct attributes
        """
        # Given/When
        error = PipelineError("Test error message")
        
        # Then
        assert error.message == "Test error message"
        assert error.category == ErrorCategory.INTERNAL
        assert error.severity == ErrorSeverity.ERROR
        assert error.recoverable is True
        assert error.timestamp is not None
    
    def test_custom_attributes(self):
        """
        Given: Custom attributes
        When: Creating PipelineError
        Then: Should use custom values
        """
        # Given/When
        error = PipelineError(
            message="Custom error",
            category=ErrorCategory.API,
            severity=ErrorSeverity.CRITICAL,
            details={"extra": "info"},
            recoverable=False
        )
        
        # Then
        assert error.category == ErrorCategory.API
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.details == {"extra": "info"}
        assert error.recoverable is False
    
    def test_to_dict(self):
        """
        Given: PipelineError
        When: Converting to dict
        Then: Should serialize all fields
        """
        # Given
        error = PipelineError(
            message="Test error",
            category=ErrorCategory.CONFIG,
            details={"key": "value"}
        )
        
        # When
        result = error.to_dict()
        
        # Then
        assert result["error_type"] == "PipelineError"
        assert result["message"] == "Test error"
        assert result["category"] == "config"
        assert result["details"] == {"key": "value"}
        assert "timestamp" in result
    
    def test_str_representation(self):
        """
        Given: PipelineError
        When: Converting to string
        Then: Should include category and message
        """
        # Given
        error = PipelineError("Something failed", category=ErrorCategory.API)
        
        # When
        result = str(error)
        
        # Then
        assert "[API]" in result
        assert "Something failed" in result


class TestConfigurationError:
    """Test ConfigurationError class"""
    
    def test_default_attributes(self):
        """
        Given: ConfigurationError
        When: Created
        Then: Should be critical and non-recoverable
        """
        # Given/When
        error = ConfigurationError("Config invalid")
        
        # Then
        assert error.category == ErrorCategory.CONFIG
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.recoverable is False
    
    def test_with_field_info(self):
        """
        Given: Field information
        When: Creating ConfigurationError
        Then: Should include in details
        """
        # Given/When
        error = ConfigurationError(
            message="Invalid value",
            field="processing.chunk_size",
            value=-100,
            suggestion="Use positive value"
        )
        
        # Then
        assert error.field == "processing.chunk_size"
        assert error.value == -100
        assert error.suggestion == "Use positive value"
        assert error.details["field"] == "processing.chunk_size"


class TestValidationError:
    """Test ValidationError class"""
    
    def test_basic_creation(self):
        """
        Given: Validation error info
        When: Creating ValidationError
        Then: Should have correct attributes
        """
        # Given/When
        error = ValidationError(
            message="Value out of range",
            field="config.temperature",
            value=2.5,
            expected="0.0 - 1.0"
        )
        
        # Then
        assert error.category == ErrorCategory.VALIDATION
        assert error.field == "config.temperature"
        assert error.value == 2.5
        assert error.expected == "0.0 - 1.0"


class TestAPIError:
    """Test APIError class"""
    
    def test_with_provider_info(self):
        """
        Given: API error info
        When: Creating APIError
        Then: Should be recoverable with details
        """
        # Given/When
        error = APIError(
            message="Server error",
            provider="gemini",
            status_code=500,
            response_body="Internal error"
        )
        
        # Then
        assert error.category == ErrorCategory.API
        assert error.recoverable is True
        assert error.provider == "gemini"
        assert error.status_code == 500


class TestRateLimitError:
    """Test RateLimitError class"""
    
    def test_with_retry_info(self):
        """
        Given: Rate limit info
        When: Creating RateLimitError
        Then: Should be warning with retry info
        """
        # Given/When
        error = RateLimitError(
            message="Rate limited",
            provider="openai",
            retry_after=60.0
        )
        
        # Then
        assert error.category == ErrorCategory.RATE_LIMIT
        assert error.severity == ErrorSeverity.WARNING
        assert error.recoverable is True
        assert error.retry_after == 60.0


class TestAuthenticationError:
    """Test AuthenticationError class"""
    
    def test_non_recoverable(self):
        """
        Given: Auth error
        When: Created
        Then: Should be critical and non-recoverable
        """
        # Given/When
        error = AuthenticationError("Invalid API key", provider="anthropic")
        
        # Then
        assert error.category == ErrorCategory.AUTH
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.recoverable is False


class TestErrorSummary:
    """Test ErrorSummary class"""
    
    def test_initial_state(self):
        """
        Given: New ErrorSummary
        When: Created
        Then: Should have zero counts
        """
        # Given/When
        summary = ErrorSummary()
        
        # Then
        assert summary.total_errors == 0
        assert summary.errors_by_category == {}
        assert summary.errors_by_severity == {}
        assert summary.recent_errors == []
    
    def test_add_error(self):
        """
        Given: ErrorSummary
        When: Adding errors
        Then: Should track correctly
        """
        # Given
        summary = ErrorSummary()
        
        # When
        summary.add_error(APIError("Error 1", provider="test"))
        summary.add_error(RateLimitError("Rate limited", provider="test"))
        
        # Then
        assert summary.total_errors == 2
        assert summary.errors_by_category["api"] == 1
        assert summary.errors_by_category["rate_limit"] == 1
        assert summary.errors_by_severity["error"] == 1
        assert summary.errors_by_severity["warning"] == 1
    
    def test_max_recent_limit(self):
        """
        Given: ErrorSummary with max_recent=5
        When: Adding more than 5 errors
        Then: Should keep only 5 most recent
        """
        # Given
        summary = ErrorSummary(max_recent=5)
        
        # When
        for i in range(10):
            summary.add_error(APIError(f"Error {i}"))
        
        # Then
        assert len(summary.recent_errors) == 5
        assert "Error 9" in summary.recent_errors[-1]["message"]
    
    def test_has_critical_errors(self):
        """
        Given: ErrorSummary
        When: Adding critical error
        Then: has_critical_errors should return True
        """
        # Given
        summary = ErrorSummary()
        
        # Initially no critical
        assert summary.has_critical_errors() is False
        
        # When
        summary.add_error(AuthenticationError("Auth failed"))
        
        # Then
        assert summary.has_critical_errors() is True
    
    def test_to_dict(self):
        """
        Given: ErrorSummary with errors
        When: Converting to dict
        Then: Should serialize correctly
        """
        # Given
        summary = ErrorSummary()
        summary.add_error(APIError("Test error"))
        
        # When
        result = summary.to_dict()
        
        # Then
        assert result["total_errors"] == 1
        assert "by_category" in result
        assert "by_severity" in result
        assert "recent_errors" in result
    
    def test_get_report(self):
        """
        Given: ErrorSummary with errors
        When: Getting report
        Then: Should return formatted string
        """
        # Given
        summary = ErrorSummary()
        summary.add_error(APIError("API failed"))
        summary.add_error(ConfigurationError("Config invalid"))
        
        # When
        report = summary.get_report()
        
        # Then
        assert "ERROR SUMMARY" in report
        assert "Total errors: 2" in report
        assert "api:" in report or "config:" in report

