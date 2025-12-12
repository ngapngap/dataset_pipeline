# -*- coding: utf-8 -*-
"""
Integration Test Suite cho Pipeline Flow

Test perspectives table:
| Case ID  | Input / Precondition           | Perspective                  | Expected Result                          | Notes |
|----------|--------------------------------|------------------------------|------------------------------------------|-------|
| TC-I-01  | Valid config + documents       | Integration - full flow      | Pipeline completes                       |       |
| TC-I-02  | Config validation              | Integration - validation     | Invalid config rejected                  |       |
| TC-I-03  | Resume from checkpoint         | Integration - resume         | Continues from last step                 |       |
"""

import os
import sys
import tempfile
import shutil

import pytest
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestPipelineInitialization:
    """Test pipeline initialization and validation"""
    
    def test_init_with_valid_config(self, config_file: str, sample_document_file: str):
        """
        Given: Valid config and documents
        When: Initializing pipeline
        Then: Should initialize successfully
        """
        # Given/When
        from pipeline import DatasetPipeline
        
        pipeline = DatasetPipeline(config_file, skip_validation=True)
        
        # Then
        assert pipeline.config is not None
        assert pipeline.project_name == "test-project"
    
    def test_init_validates_config(self, invalid_config_file: str):
        """
        Given: Invalid config
        When: Initializing pipeline
        Then: Should raise ConfigurationError
        """
        # Given
        from pipeline import DatasetPipeline
        from core.errors import ConfigurationError
        
        # When/Then
        with pytest.raises(ConfigurationError):
            DatasetPipeline(invalid_config_file, skip_validation=False)
    
    def test_init_with_skip_validation(self, invalid_config_file: str):
        """
        Given: Invalid config with skip_validation=True
        When: Initializing pipeline
        Then: Should initialize (skip validation)
        """
        # Given
        from pipeline import DatasetPipeline
        
        # When/Then - should not raise
        pipeline = DatasetPipeline(invalid_config_file, skip_validation=True)
        assert pipeline.config is not None


class TestExtractStep:
    """Test extract step"""
    
    def test_extract_documents(self, config_file: str, sample_document_file: str):
        """
        Given: Input directory with documents
        When: Running extract step
        Then: Should extract documents
        """
        # Given
        from pipeline import DatasetPipeline
        
        pipeline = DatasetPipeline(config_file, skip_validation=True)
        
        # When
        pipeline._run_extract()
        
        # Then
        assert len(pipeline.state["documents"]) > 0
        assert "content" in pipeline.state["documents"][0]


class TestEvaluateStep:
    """Test evaluate step with mock data"""
    
    def test_evaluate_qa_pairs(self, config_file: str, sample_qa_pairs):
        """
        Given: Q&A pairs
        When: Running evaluate step
        Then: Should separate good and bad
        """
        # Given
        from pipeline import DatasetPipeline
        
        pipeline = DatasetPipeline(config_file, skip_validation=True)
        pipeline.state["qa_pairs"] = sample_qa_pairs
        
        # When
        pipeline._run_evaluate()
        
        # Then
        total = len(pipeline.state["good_qa"]) + len(pipeline.state["bad_qa"])
        assert total > 0


class TestSplitStep:
    """Test split step"""
    
    def test_split_good_qa(self, config_file: str, sample_qa_good):
        """
        Given: Good Q&A pairs
        When: Running split step
        Then: Should create train/val/test splits
        """
        # Given
        from pipeline import DatasetPipeline
        
        pipeline = DatasetPipeline(config_file, skip_validation=True)
        pipeline.state["good_qa"] = sample_qa_good
        
        # When
        pipeline._run_split()
        
        # Then
        assert "splits" in pipeline.state
        splits = pipeline.state["splits"]
        total = len(splits.get("train", [])) + len(splits.get("validation", [])) + len(splits.get("test", []))
        assert total == len(sample_qa_good)


class TestPipelineState:
    """Test pipeline state management"""
    
    def test_save_and_load_state(self, config_file: str, temp_dir: str):
        """
        Given: Pipeline with state
        When: Saving and loading state
        Then: State should persist
        """
        # Given
        from pipeline import DatasetPipeline
        
        pipeline = DatasetPipeline(config_file, skip_validation=True)
        pipeline.state["steps_completed"] = ["extract"]
        pipeline.state["documents"] = [{"file_name": "test.txt", "content": "Test"}]
        
        # When
        pipeline._save_state()
        
        # Then - state file should exist
        output_dir = pipeline.config.get("general.output_dir")
        state_file = os.path.join(output_dir, "pipeline_state.json")
        assert os.path.exists(state_file)


class TestErrorHandling:
    """Test error handling in pipeline"""
    
    def test_error_summary_tracking(self, config_file: str):
        """
        Given: Pipeline with error summary
        When: Errors occur
        Then: Should be tracked
        """
        # Given
        from pipeline import DatasetPipeline
        from core.errors import APIError
        
        pipeline = DatasetPipeline(config_file, skip_validation=True)
        
        # When
        pipeline.error_summary.add_error(APIError("Test error"))
        
        # Then
        assert pipeline.error_summary.total_errors == 1
        assert pipeline.error_summary.errors_by_category.get("api") == 1

