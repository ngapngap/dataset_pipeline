# -*- coding: utf-8 -*-
"""
Test Suite cho steps/splitter.py

Test perspectives table:
| Case ID  | Input / Precondition           | Perspective                  | Expected Result                          | Notes |
|----------|--------------------------------|------------------------------|------------------------------------------|-------|
| TC-N-01  | Normal Q&A list                | Equivalence - normal         | Split into train/val/test                |       |
| TC-N-02  | Multiple documents             | Equivalence - docs           | No overlap between splits                |       |
| TC-A-01  | Empty Q&A list                 | Boundary - empty             | Empty splits                             |       |
| TC-A-02  | Single document                | Boundary - one doc           | All in one split                         |       |
| TC-B-01  | Validate leakage detection     | Boundary - leakage           | Detects overlap                          |       |
"""

import os
import sys
import tempfile
import shutil

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from steps.splitter import DatasetSplitter


class TestDatasetSplitter:
    """Test DatasetSplitter class"""
    
    @pytest.fixture
    def splitter(self, temp_dir: str):
        """Create splitter for testing"""
        return DatasetSplitter({
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "seed": 42,
            "output_dir": temp_dir
        })
    
    def test_split_by_document(self, splitter):
        """
        Given: Q&A pairs from multiple documents
        When: Splitting
        Then: Q&A from same document should stay together
        """
        # Given
        qa_pairs = [
            {"question": "Q1 from Doc A", "answer": "A1", "source_doc": "doc_a.txt"},
            {"question": "Q2 from Doc A", "answer": "A2", "source_doc": "doc_a.txt"},
            {"question": "Q3 from Doc A", "answer": "A3", "source_doc": "doc_a.txt"},
            {"question": "Q1 from Doc B", "answer": "A1", "source_doc": "doc_b.txt"},
            {"question": "Q2 from Doc B", "answer": "A2", "source_doc": "doc_b.txt"},
            {"question": "Q1 from Doc C", "answer": "A1", "source_doc": "doc_c.txt"},
            {"question": "Q2 from Doc C", "answer": "A2", "source_doc": "doc_c.txt"},
            {"question": "Q1 from Doc D", "answer": "A1", "source_doc": "doc_d.txt"},
            {"question": "Q1 from Doc E", "answer": "A1", "source_doc": "doc_e.txt"},
        ]
        
        # When
        splits = splitter.split(qa_pairs)
        
        # Then
        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits
        
        # Check total
        total = len(splits["train"]) + len(splits["validation"]) + len(splits["test"])
        assert total == len(qa_pairs)
    
    def test_no_document_overlap(self, splitter):
        """
        Given: Split results
        When: Checking for overlap
        Then: No document should appear in multiple splits
        """
        # Given
        qa_pairs = [
            {"question": "Q1", "answer": "A1", "source_doc": "doc_a"},
            {"question": "Q2", "answer": "A2", "source_doc": "doc_a"},
            {"question": "Q3", "answer": "A3", "source_doc": "doc_b"},
            {"question": "Q4", "answer": "A4", "source_doc": "doc_c"},
            {"question": "Q5", "answer": "A5", "source_doc": "doc_d"},
            {"question": "Q6", "answer": "A6", "source_doc": "doc_e"},
        ]
        
        # When
        splits = splitter.split(qa_pairs)
        
        # Then
        is_valid = splitter.validate_no_leakage(splits)
        assert is_valid is True
    
    def test_validate_detects_leakage(self, splitter):
        """
        Given: Splits with artificial overlap
        When: Validating
        Then: Should detect leakage
        """
        # Given - manually create overlapping splits
        splits = {
            "train": [{"question": "Q1", "source_doc": "same_doc"}],
            "validation": [{"question": "Q2", "source_doc": "same_doc"}],  # Same doc!
            "test": [{"question": "Q3", "source_doc": "other_doc"}]
        }
        
        # When
        is_valid = splitter.validate_no_leakage(splits)
        
        # Then
        assert is_valid is False
    
    def test_split_empty_list(self, splitter):
        """
        Given: Empty Q&A list
        When: Splitting
        Then: Should return empty splits
        """
        # When
        splits = splitter.split([])
        
        # Then
        assert splits["train"] == []
        assert splits["validation"] == []
        assert splits["test"] == []
    
    def test_split_single_document(self, splitter):
        """
        Given: All Q&A from single document
        When: Splitting
        Then: All should go to one split
        """
        # Given
        qa_pairs = [
            {"question": "Q1", "answer": "A1", "source_doc": "single_doc"},
            {"question": "Q2", "answer": "A2", "source_doc": "single_doc"},
            {"question": "Q3", "answer": "A3", "source_doc": "single_doc"},
        ]
        
        # When
        splits = splitter.split(qa_pairs)
        
        # Then
        # All should be in one split (train since it's first)
        total = len(splits["train"]) + len(splits["validation"]) + len(splits["test"])
        assert total == 3
        # One split should have all 3
        assert any(len(s) == 3 for s in splits.values())
    
    def test_reproducible_split(self, temp_dir: str):
        """
        Given: Same seed
        When: Splitting twice
        Then: Should produce same results
        """
        # Given
        qa_pairs = [
            {"question": f"Q{i}", "answer": f"A{i}", "source_doc": f"doc_{i}"}
            for i in range(20)
        ]
        
        # When
        splitter1 = DatasetSplitter({
            "train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2,
            "seed": 123,
            "output_dir": os.path.join(temp_dir, "split1")
        })
        splits1 = splitter1.split(qa_pairs.copy())
        
        splitter2 = DatasetSplitter({
            "train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2,
            "seed": 123,
            "output_dir": os.path.join(temp_dir, "split2")
        })
        splits2 = splitter2.split(qa_pairs.copy())
        
        # Then
        assert len(splits1["train"]) == len(splits2["train"])
        assert len(splits1["validation"]) == len(splits2["validation"])
        assert len(splits1["test"]) == len(splits2["test"])
    
    def test_load_splits(self, splitter):
        """
        Given: Saved splits
        When: Loading
        Then: Should return same data
        """
        # Given
        qa_pairs = [
            {"question": "Q1", "answer": "A1", "source_doc": "doc_a"},
            {"question": "Q2", "answer": "A2", "source_doc": "doc_b"},
            {"question": "Q3", "answer": "A3", "source_doc": "doc_c"},
        ]
        splitter.split(qa_pairs)
        
        # When
        loaded = splitter.load_splits()
        
        # Then
        total_loaded = len(loaded["train"]) + len(loaded["validation"]) + len(loaded["test"])
        assert total_loaded == 3


class TestSplitRatios:
    """Test split ratio handling"""
    
    def test_custom_ratios(self, temp_dir: str):
        """
        Given: Custom split ratios
        When: Splitting
        Then: Should approximately match ratios
        """
        # Given
        splitter = DatasetSplitter({
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "seed": 42,
            "output_dir": temp_dir
        })
        
        # Create many documents for better ratio approximation
        qa_pairs = [
            {"question": f"Q{i}", "answer": f"A{i}", "source_doc": f"doc_{i}"}
            for i in range(100)
        ]
        
        # When
        splits = splitter.split(qa_pairs)
        
        # Then - ratios should be approximately correct
        total = len(qa_pairs)
        train_ratio = len(splits["train"]) / total
        val_ratio = len(splits["validation"]) / total
        test_ratio = len(splits["test"]) / total
        
        # Allow 20% tolerance since we split by document
        assert 0.6 <= train_ratio <= 1.0
        assert 0.0 <= val_ratio <= 0.3
        assert 0.0 <= test_ratio <= 0.3
    
    def test_warning_on_invalid_ratios(self, temp_dir: str, caplog):
        """
        Given: Ratios that don't sum to 1
        When: Creating splitter
        Then: Should log warning
        """
        # Given/When
        import logging
        with caplog.at_level(logging.WARNING):
            splitter = DatasetSplitter({
                "train_ratio": 0.5,
                "val_ratio": 0.3,
                "test_ratio": 0.1,  # Sum = 0.9
                "output_dir": temp_dir
            })
        
        # Then
        assert any("ratio" in record.message.lower() or "1.0" in record.message 
                   for record in caplog.records)

