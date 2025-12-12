# -*- coding: utf-8 -*-
"""
Test Suite cho core/utils.py

Test perspectives table:
| Case ID  | Input / Precondition           | Perspective                  | Expected Result                          | Notes |
|----------|--------------------------------|------------------------------|------------------------------------------|-------|
| TC-N-01  | Normal text, normal chunk_size | Equivalence - normal         | Correct chunks with overlap              |       |
| TC-N-02  | Valid JSON data                | Equivalence - normal         | Save and load correctly                  |       |
| TC-N-03  | JSONL data                     | Equivalence - normal         | Save and load line by line               |       |
| TC-A-01  | Empty text                     | Boundary - empty             | Returns empty list                       |       |
| TC-A-02  | Text < chunk_size              | Boundary - small             | Returns single chunk                     |       |
| TC-A-03  | Empty file                     | Boundary - empty file        | Returns empty list                       |       |
| TC-A-04  | File not found                 | Boundary - missing file      | Returns empty or raises                  |       |
| TC-B-01  | chunk_size = 0                 | Boundary - zero              | Handle edge case                         |       |
| TC-B-02  | overlap >= chunk_size          | Boundary - invalid overlap   | Handle gracefully                        |       |
| TC-B-03  | Unicode text                   | Boundary - encoding          | Handle correctly                         |       |
"""

import os
import sys
import json
import tempfile
import shutil

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.utils import (
    chunk_text,
    save_json,
    load_json,
    save_jsonl,
    load_jsonl,
    load_api_keys,
    normalize_text,
    compute_hash,
    deduplicate_qa,
    get_file_list,
    CacheManager
)


class TestChunkText:
    """Test chunk_text function"""
    
    def test_normal_chunking(self):
        """
        Given: Text longer than chunk_size
        When: Calling chunk_text
        Then: Should return multiple chunks with overlap
        """
        # Given
        text = "A" * 5000
        
        # When
        chunks = chunk_text(text, chunk_size=1000, overlap=100)
        
        # Then
        assert len(chunks) > 1
        # Each chunk should not exceed chunk_size + some buffer
        for chunk in chunks:
            assert len(chunk) <= 1200  # Allow buffer for sentence boundary
    
    def test_short_text(self):
        """
        Given: Text shorter than chunk_size
        When: Calling chunk_text
        Then: Should return single chunk
        """
        # Given
        text = "Đây là văn bản ngắn."
        
        # When
        chunks = chunk_text(text, chunk_size=1000)
        
        # Then
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_empty_text(self):
        """
        Given: Empty text
        When: Calling chunk_text
        Then: Should return empty list
        """
        # Given/When
        chunks = chunk_text("")
        
        # Then
        assert chunks == []
    
    def test_none_text(self):
        """
        Given: None as text
        When: Calling chunk_text
        Then: Should return empty list
        """
        # Given/When
        chunks = chunk_text(None)
        
        # Then
        assert chunks == []
    
    def test_unicode_text(self):
        """
        Given: Vietnamese text
        When: Calling chunk_text
        Then: Should handle unicode correctly
        """
        # Given
        text = "Điều 1. Luật BHXH quy định về chế độ bảo hiểm. " * 100
        
        # When
        chunks = chunk_text(text, chunk_size=500)
        
        # Then
        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk, str)
    
    def test_sentence_boundary_respected(self):
        """
        Given: Text with sentences
        When: Calling chunk_text
        Then: Should prefer breaking at sentence boundaries
        """
        # Given
        text = "Câu thứ nhất. Câu thứ hai. Câu thứ ba. " * 50
        
        # When
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        
        # Then
        # Most chunks should end with sentence markers
        sentence_endings = sum(1 for c in chunks if c.rstrip().endswith('.'))
        assert sentence_endings > len(chunks) // 2


class TestJsonIO:
    """Test JSON save/load functions"""
    
    def test_save_and_load_json(self, temp_dir: str):
        """
        Given: Python data structure
        When: Saving and loading JSON
        Then: Data should be identical
        """
        # Given
        data = [{"question": "Test?", "answer": "Answer"}]
        filepath = os.path.join(temp_dir, "test.json")
        
        # When
        save_json(data, filepath)
        loaded = load_json(filepath)
        
        # Then
        assert data == loaded
    
    def test_save_json_creates_directory(self, temp_dir: str):
        """
        Given: Path with non-existent directory
        When: Saving JSON
        Then: Directory should be created
        """
        # Given
        filepath = os.path.join(temp_dir, "subdir", "nested", "test.json")
        data = {"key": "value"}
        
        # When
        save_json(data, filepath)
        
        # Then
        assert os.path.exists(filepath)
    
    def test_save_json_unicode(self, temp_dir: str):
        """
        Given: Data with Vietnamese text
        When: Saving JSON with ensure_ascii=False
        Then: Unicode should be preserved
        """
        # Given
        data = {"message": "Xin chào Việt Nam"}
        filepath = os.path.join(temp_dir, "unicode.json")
        
        # When
        save_json(data, filepath)
        
        # Then
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "Việt Nam" in content  # Not escaped
    
    def test_save_and_load_jsonl(self, temp_dir: str):
        """
        Given: List of dicts
        When: Saving and loading JSONL
        Then: Data should be identical
        """
        # Given
        data = [
            {"q": "Q1", "a": "A1"},
            {"q": "Q2", "a": "A2"}
        ]
        filepath = os.path.join(temp_dir, "test.jsonl")
        
        # When
        save_jsonl(data, filepath)
        loaded = load_jsonl(filepath)
        
        # Then
        assert data == loaded


class TestLoadApiKeys:
    """Test load_api_keys function"""
    
    def test_load_keys(self, temp_dir: str):
        """
        Given: File with API keys
        When: Loading keys
        Then: Should return list of keys
        """
        # Given
        keys_file = os.path.join(temp_dir, "keys.txt")
        with open(keys_file, 'w') as f:
            f.write("key1\n")
            f.write("# comment\n")
            f.write("key2\n")
            f.write("\n")  # Empty line
            f.write("key3\n")
        
        # When
        keys = load_api_keys(keys_file)
        
        # Then
        assert keys == ["key1", "key2", "key3"]
    
    def test_load_keys_file_not_found(self):
        """
        Given: Non-existent file
        When: Loading keys
        Then: Should return empty list
        """
        # Given/When
        keys = load_api_keys("/nonexistent/path/keys.txt")
        
        # Then
        assert keys == []
    
    def test_load_keys_empty_file(self, temp_dir: str):
        """
        Given: Empty file
        When: Loading keys
        Then: Should return empty list
        """
        # Given
        keys_file = os.path.join(temp_dir, "empty.txt")
        with open(keys_file, 'w') as f:
            f.write("")
        
        # When
        keys = load_api_keys(keys_file)
        
        # Then
        assert keys == []


class TestDeduplicateQA:
    """Test deduplicate_qa function"""
    
    def test_remove_duplicate_questions(self):
        """
        Given: List with duplicate questions
        When: Deduplicating
        Then: Should keep only unique questions
        """
        # Given
        qa_list = [
            {"question": "Câu hỏi 1?", "answer": "A1"},
            {"question": "Câu hỏi 1?", "answer": "A2"},  # Duplicate
            {"question": "Câu hỏi 2?", "answer": "A3"},
        ]
        
        # When
        unique = deduplicate_qa(qa_list)
        
        # Then
        assert len(unique) == 2
    
    def test_empty_list(self):
        """
        Given: Empty list
        When: Deduplicating
        Then: Should return empty list
        """
        # Given/When
        unique = deduplicate_qa([])
        
        # Then
        assert unique == []
    
    def test_no_duplicates(self):
        """
        Given: List with no duplicates
        When: Deduplicating
        Then: Should return same list
        """
        # Given
        qa_list = [
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2"},
        ]
        
        # When
        unique = deduplicate_qa(qa_list)
        
        # Then
        assert len(unique) == 2


class TestCacheManager:
    """Test CacheManager class"""
    
    def test_cache_set_and_get(self, temp_dir: str):
        """
        Given: Cache manager
        When: Setting and getting value
        Then: Should return cached value
        """
        # Given
        cache = CacheManager(cache_dir=temp_dir, ttl_days=30, enabled=True)
        content = "Test content"
        result = [{"q": "Q1", "a": "A1"}]
        
        # When
        cache.set(content=content, result=result, model="test")
        cached = cache.get(content=content, model="test")
        
        # Then
        assert cached == result
        assert cache.hits == 1
    
    def test_cache_miss(self, temp_dir: str):
        """
        Given: Empty cache
        When: Getting non-existent key
        Then: Should return None
        """
        # Given
        cache = CacheManager(cache_dir=temp_dir, enabled=True)
        
        # When
        result = cache.get(content="nonexistent", model="test")
        
        # Then
        assert result is None
        assert cache.misses == 1
    
    def test_cache_disabled(self, temp_dir: str):
        """
        Given: Disabled cache
        When: Setting and getting
        Then: Should always return None
        """
        # Given
        cache = CacheManager(cache_dir=temp_dir, enabled=False)
        
        # When
        cache.set(content="test", result=["data"], model="test")
        result = cache.get(content="test", model="test")
        
        # Then
        assert result is None
    
    def test_cache_exists(self, temp_dir: str):
        """
        Given: Cached item
        When: Checking exists
        Then: Should return True
        """
        # Given
        cache = CacheManager(cache_dir=temp_dir, enabled=True)
        cache.set(content="test", result=["data"], model="test")
        
        # When/Then
        assert cache.exists(content="test", model="test") is True
        assert cache.exists(content="other", model="test") is False
    
    def test_cache_invalidate(self, temp_dir: str):
        """
        Given: Cached item
        When: Invalidating
        Then: Should be removed
        """
        # Given
        cache = CacheManager(cache_dir=temp_dir, enabled=True)
        cache.set(content="test", result=["data"], model="test")
        
        # Get cache key
        cache_key = cache._get_cache_key("test", "", "test")
        
        # When
        cache.invalidate(cache_key)
        
        # Then
        assert cache.get(content="test", model="test") is None
    
    def test_cache_clear(self, temp_dir: str):
        """
        Given: Cache with items
        When: Clearing cache
        Then: All items should be removed
        """
        # Given
        cache = CacheManager(cache_dir=temp_dir, enabled=True)
        cache.set(content="test1", result=["data1"], model="test")
        cache.set(content="test2", result=["data2"], model="test")
        
        # When
        cache.clear()
        
        # Then
        assert cache.get(content="test1", model="test") is None
        assert cache.get(content="test2", model="test") is None
    
    def test_cache_stats(self, temp_dir: str):
        """
        Given: Cache with activity
        When: Getting stats
        Then: Should return correct stats
        """
        # Given
        cache = CacheManager(cache_dir=temp_dir, enabled=True)
        cache.set(content="test", result=["data"], model="test")
        cache.get(content="test", model="test")  # Hit
        cache.get(content="other", model="test")  # Miss
        
        # When
        stats = cache.get_stats()
        
        # Then
        assert stats["enabled"] is True
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["cache_files"] >= 1


class TestNormalizeText:
    """Test normalize_text function"""
    
    def test_lowercase(self):
        """
        Given: Mixed case text
        When: Normalizing
        Then: Should be lowercase
        """
        assert normalize_text("Hello World") == "hello world"
    
    def test_strip_whitespace(self):
        """
        Given: Text with extra whitespace
        When: Normalizing
        Then: Should strip and collapse whitespace
        """
        assert normalize_text("  hello   world  ") == "hello world"
    
    def test_unicode(self):
        """
        Given: Vietnamese text
        When: Normalizing
        Then: Should handle correctly
        """
        assert normalize_text("XIN CHÀO VIỆT NAM") == "xin chào việt nam"


class TestComputeHash:
    """Test compute_hash function"""
    
    def test_consistent_hash(self):
        """
        Given: Same text
        When: Computing hash multiple times
        Then: Should return same hash
        """
        text = "Test content"
        hash1 = compute_hash(text)
        hash2 = compute_hash(text)
        assert hash1 == hash2
    
    def test_different_hash(self):
        """
        Given: Different texts
        When: Computing hashes
        Then: Should be different
        """
        hash1 = compute_hash("Text 1")
        hash2 = compute_hash("Text 2")
        assert hash1 != hash2
    
    def test_hash_length(self):
        """
        Given: Any text
        When: Computing hash
        Then: Should return 12 character hash
        """
        assert len(compute_hash("test")) == 12

