# -*- coding: utf-8 -*-
"""
Test Suite cho core/validators.py

Test perspectives table:
| Case ID  | Input / Precondition           | Perspective                  | Expected Result                          | Notes |
|----------|--------------------------------|------------------------------|------------------------------------------|-------|
| TC-N-01  | Valid config                   | Equivalence - normal         | Validation passes                        |       |
| TC-N-02  | Valid paths                    | Equivalence - normal         | No path errors                           |       |
| TC-N-03  | Valid API keys                 | Equivalence - normal         | No API key errors                        |       |
| TC-A-01  | Missing input_dir              | Boundary - missing           | Error returned                           |       |
| TC-A-02  | No API keys                    | Boundary - missing           | Error returned                           |       |
| TC-A-03  | Invalid ratios                 | Boundary - sum != 1          | Error returned                           |       |
| TC-B-01  | chunk_size = 0                 | Boundary - zero              | Error returned                           |       |
| TC-B-02  | chunk_size = -1                | Boundary - negative          | Error returned                           |       |
| TC-B-03  | overlap >= chunk_size          | Boundary - invalid           | Error returned                           |       |
"""

import os
import sys
import tempfile
import shutil

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.config import PipelineConfig
from core.validators import (
    ConfigValidator,
    ValidationResult,
    ValidationIssue,
    validate_config
)
from core.errors import ConfigurationError, ErrorSeverity


class TestValidationResult:
    """Test ValidationResult class"""
    
    def test_initial_state(self):
        """
        Given: New ValidationResult
        When: Created
        Then: Should be valid with no errors
        """
        # Given/When
        result = ValidationResult()
        
        # Then
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
    
    def test_add_error(self):
        """
        Given: ValidationResult
        When: Adding error
        Then: Should be invalid
        """
        # Given
        result = ValidationResult()
        
        # When
        result.add_error("field", "Error message", value="bad_value")
        
        # Then
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == "field"
        assert result.errors[0].message == "Error message"
    
    def test_add_warning(self):
        """
        Given: ValidationResult
        When: Adding warning
        Then: Should still be valid
        """
        # Given
        result = ValidationResult()
        
        # When
        result.add_warning("field", "Warning message")
        
        # Then
        assert result.is_valid is True
        assert len(result.warnings) == 1
    
    def test_merge(self):
        """
        Given: Two ValidationResults
        When: Merging
        Then: Errors and warnings combined
        """
        # Given
        result1 = ValidationResult()
        result1.add_error("field1", "Error 1")
        
        result2 = ValidationResult()
        result2.add_warning("field2", "Warning 1")
        
        # When
        result1.merge(result2)
        
        # Then
        assert result1.is_valid is False
        assert len(result1.errors) == 1
        assert len(result1.warnings) == 1
    
    def test_get_error_messages(self):
        """
        Given: Result with errors
        When: Getting error messages
        Then: Should return formatted messages
        """
        # Given
        result = ValidationResult()
        result.add_error("config.field", "Something wrong")
        
        # When
        messages = result.get_error_messages()
        
        # Then
        assert len(messages) == 1
        assert "[config.field]" in messages[0]
        assert "Something wrong" in messages[0]


class TestConfigValidator:
    """Test ConfigValidator class"""
    
    def test_validate_valid_config(self, config_file: str):
        """
        Given: Valid config
        When: Validating all
        Then: Should pass
        """
        # Given
        config = PipelineConfig(config_file)
        validator = ConfigValidator(config)
        
        # When
        result = validator.validate_all()
        
        # Then
        assert result.is_valid is True
    
    def test_validate_paths_input_not_exists(self, temp_dir: str):
        """
        Given: Config with non-existent input_dir
        When: Validating paths
        Then: Should return error
        """
        # Given
        import yaml
        
        config_dict = {
            "general": {
                "input_dir": "/nonexistent/path",
                "output_dir": os.path.join(temp_dir, "output")
            },
            "llm": {"provider": "gemini", "providers": {}},
            "processing": {},
            "output": {}
        }
        os.makedirs(config_dict["general"]["output_dir"], exist_ok=True)
        
        config_path = os.path.join(temp_dir, "test.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        
        config = PipelineConfig(config_path)
        validator = ConfigValidator(config)
        
        # When
        result = validator.validate_paths()
        
        # Then
        assert result.is_valid is False
        assert any("input_dir" in e.field for e in result.errors)
    
    def test_validate_paths_empty_input(self, temp_dir: str, sample_config_dict):
        """
        Given: Empty input directory
        When: Validating paths
        Then: Should return warning
        """
        # Given
        import yaml
        
        input_dir = os.path.join(temp_dir, "empty_input")
        os.makedirs(input_dir)
        
        sample_config_dict["general"]["input_dir"] = input_dir
        sample_config_dict["general"]["output_dir"] = os.path.join(temp_dir, "output")
        sample_config_dict["general"]["log_file"] = os.path.join(temp_dir, "log.txt")
        os.makedirs(sample_config_dict["general"]["output_dir"], exist_ok=True)
        
        config_path = os.path.join(temp_dir, "test.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(sample_config_dict, f)
        
        config = PipelineConfig(config_path)
        validator = ConfigValidator(config)
        
        # When
        result = validator.validate_paths()
        
        # Then
        # Empty dir should trigger warning, not error
        assert any("rỗng" in w.message or "empty" in w.message.lower() 
                   for w in result.warnings)
    
    def test_validate_api_keys_missing(self, temp_dir: str):
        """
        Given: Config without API keys
        When: Validating API keys
        Then: Should return error
        """
        # Given
        import yaml
        
        config_dict = {
            "general": {
                "input_dir": os.path.join(temp_dir, "input"),
                "output_dir": os.path.join(temp_dir, "output")
            },
            "llm": {
                "provider": "gemini",
                "providers": {
                    "gemini": {
                        "model": "gemini-2.0-flash"
                        # No api_key or api_keys_file
                    }
                }
            },
            "processing": {},
            "output": {}
        }
        os.makedirs(config_dict["general"]["input_dir"], exist_ok=True)
        os.makedirs(config_dict["general"]["output_dir"], exist_ok=True)
        
        config_path = os.path.join(temp_dir, "test.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        
        config = PipelineConfig(config_path)
        validator = ConfigValidator(config)
        
        # When
        result = validator.validate_api_keys()
        
        # Then
        assert result.is_valid is False
        assert any("api" in e.message.lower() or "key" in e.message.lower() 
                   for e in result.errors)
    
    def test_validate_api_keys_local_provider(self, temp_dir: str):
        """
        Given: Local provider (ollama)
        When: Validating API keys
        Then: Should pass without API key
        """
        # Given
        import yaml
        
        config_dict = {
            "general": {
                "input_dir": os.path.join(temp_dir, "input"),
                "output_dir": os.path.join(temp_dir, "output")
            },
            "llm": {
                "provider": "ollama",
                "providers": {
                    "ollama": {
                        "model": "llama2",
                        "base_url": "http://localhost:11434/v1",
                        "api_key": "ollama"
                    }
                }
            },
            "processing": {},
            "output": {}
        }
        os.makedirs(config_dict["general"]["input_dir"], exist_ok=True)
        os.makedirs(config_dict["general"]["output_dir"], exist_ok=True)
        
        config_path = os.path.join(temp_dir, "test.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        
        config = PipelineConfig(config_path)
        validator = ConfigValidator(config)
        
        # When
        result = validator.validate_api_keys()
        
        # Then
        # Local provider doesn't need real API key
        assert result.is_valid is True
    
    def test_validate_processing_negative_chunk_size(self, config_file: str):
        """
        Given: Negative chunk_size
        When: Validating processing
        Then: Should return error
        """
        # Given
        config = PipelineConfig(config_file)
        config.processing.chunk_size = -100
        validator = ConfigValidator(config)
        
        # When
        result = validator.validate_processing()
        
        # Then
        assert result.is_valid is False
        assert any("chunk_size" in e.field for e in result.errors)
    
    def test_validate_processing_overlap_too_large(self, config_file: str):
        """
        Given: chunk_overlap >= chunk_size
        When: Validating processing
        Then: Should return error
        """
        # Given
        config = PipelineConfig(config_file)
        config.processing.chunk_size = 1000
        config.processing.chunk_overlap = 1500
        validator = ConfigValidator(config)
        
        # When
        result = validator.validate_processing()
        
        # Then
        assert result.is_valid is False
        assert any("overlap" in e.field for e in result.errors)
    
    def test_validate_ratios_sum_not_one(self, temp_dir: str):
        """
        Given: Ratios that don't sum to 1
        When: Validating ratios
        Then: Should return error
        """
        # Given
        import yaml
        
        config_dict = {
            "general": {
                "input_dir": os.path.join(temp_dir, "input"),
                "output_dir": os.path.join(temp_dir, "output")
            },
            "llm": {"provider": "gemini", "providers": {}},
            "processing": {},
            "output": {
                "train_ratio": 0.5,
                "val_ratio": 0.5,
                "test_ratio": 0.5  # Sum = 1.5
            }
        }
        os.makedirs(config_dict["general"]["input_dir"], exist_ok=True)
        os.makedirs(config_dict["general"]["output_dir"], exist_ok=True)
        
        config_path = os.path.join(temp_dir, "test.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        
        config = PipelineConfig(config_path)
        validator = ConfigValidator(config)
        
        # When
        result = validator.validate_ratios()
        
        # Then
        assert result.is_valid is False
        assert any("ratio" in e.field.lower() for e in result.errors)


class TestValidateConfigFunction:
    """Test validate_config convenience function"""
    
    def test_valid_config_passes(self, config_file: str):
        """
        Given: Valid config
        When: Using validate_config
        Then: Should not raise
        """
        # Given
        config = PipelineConfig(config_file)
        
        # When/Then - should not raise
        result = validate_config(config)
        assert result.is_valid
    
    def test_invalid_config_raises(self, invalid_config_file: str):
        """
        Given: Invalid config
        When: Using validate_config
        Then: Should raise ConfigurationError
        """
        # Given
        config = PipelineConfig(invalid_config_file)
        
        # When/Then
        with pytest.raises(ConfigurationError):
            validate_config(config)

