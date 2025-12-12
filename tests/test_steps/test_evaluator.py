# -*- coding: utf-8 -*-
"""
Test Suite cho steps/evaluator.py

Test perspectives table:
| Case ID  | Input / Precondition           | Perspective                  | Expected Result                          | Notes |
|----------|--------------------------------|------------------------------|------------------------------------------|-------|
| TC-N-01  | Q&A with legal citation        | Equivalence - good           | High score (>= 7)                        |       |
| TC-N-02  | Q&A without citation           | Equivalence - bad            | Low score (< 7)                          |       |
| TC-N-03  | Duplicate Q&A pairs            | Equivalence - dedup          | Duplicates removed                       |       |
| TC-A-01  | Empty Q&A list                 | Boundary - empty             | Empty results                            |       |
| TC-A-02  | Very short answer              | Boundary - min length        | Low score                                |       |
| TC-B-01  | Vague question                 | Boundary - quality           | Low score                                |       |
"""

import os
import sys
import tempfile
import shutil

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from steps.evaluator import QualityEvaluator


class TestQualityEvaluator:
    """Test QualityEvaluator class"""
    
    @pytest.fixture
    def evaluator(self, temp_dir: str):
        """Create evaluator for testing"""
        return QualityEvaluator({
            "mode": "rule",
            "min_score": 7,
            "min_question_length": 10,
            "min_answer_length": 20,
            "max_answer_length": 5000,
            "output_dir": temp_dir
        })
    
    def test_rule_score_with_legal_citation(self, evaluator):
        """
        Given: Q&A with proper legal citation
        When: Scoring
        Then: Should get high score
        """
        # Given
        qa = {
            "question": "Mức đóng BHXH của người lao động là bao nhiêu phần trăm?",
            "answer": "Căn cứ Điều 5, Khoản 2, Luật BHXH số 58/2014/QH13, mức đóng BHXH bắt buộc của người lao động là 8% tiền lương hàng tháng."
        }
        
        # When
        score, reason = evaluator._rule_score(qa)
        
        # Then
        assert score >= 7, f"Score {score} should be >= 7. Reason: {reason}"
        assert "căn cứ" in reason.lower() or "✅" in reason
    
    def test_rule_score_without_legal_citation(self, evaluator):
        """
        Given: Q&A without legal citation
        When: Scoring
        Then: Should get low score
        """
        # Given
        qa = {
            "question": "Mức đóng BHXH là bao nhiêu?",
            "answer": "Mức đóng BHXH là 8% tiền lương hàng tháng của người lao động."
        }
        
        # When
        score, reason = evaluator._rule_score(qa)
        
        # Then
        assert score < 7, f"Score {score} should be < 7"
        assert "THIẾU" in reason or "căn cứ" in reason.lower()
    
    def test_rule_score_short_answer(self, evaluator):
        """
        Given: Q&A with very short answer
        When: Scoring
        Then: Should get low score
        """
        # Given
        qa = {
            "question": "Hỏi gì đó về BHXH?",
            "answer": "Ngắn"  # Too short
        }
        
        # When
        score, reason = evaluator._rule_score(qa)
        
        # Then
        assert score < 7
        assert "ngắn" in reason.lower()
    
    def test_rule_score_vague_question(self, evaluator):
        """
        Given: Vague generic question
        When: Scoring
        Then: Should get low score
        """
        # Given
        qa = {
            "question": "Thông tư này quy định về nội dung gì?",
            "answer": "Căn cứ Điều 1, Thông tư số 25/2025/TT-BYT quy định về chi tiết một số điều của Luật BHYT."
        }
        
        # When
        score, reason = evaluator._rule_score(qa)
        
        # Then
        assert score < 7, f"Score {score} should be < 7 for vague question"
        assert "chung" in reason.lower()
    
    def test_rule_score_with_document_number(self, evaluator):
        """
        Given: Q&A with proper document number
        When: Scoring
        Then: Should get good score
        """
        # Given
        qa = {
            "question": "Điều kiện hưởng lương hưu theo Luật BHXH mới như thế nào?",
            "answer": "Căn cứ Điều 54, Khoản 1, Luật BHXH số 41/2024/QH15, người lao động được hưởng lương hưu khi đủ tuổi nghỉ hưu (nam 62 tuổi, nữ 60 tuổi) và có thời gian đóng BHXH đủ 15 năm trở lên."
        }
        
        # When
        score, reason = evaluator._rule_score(qa)
        
        # Then
        assert score >= 7
    
    def test_evaluate_rule_mode(self, evaluator, temp_dir: str):
        """
        Given: List of Q&A pairs
        When: Evaluating in rule mode
        Then: Should separate good and bad
        """
        # Given
        qa_pairs = [
            {
                "question": "Mức đóng BHXH bắt buộc của người lao động là bao nhiêu?",
                "answer": "Căn cứ Điều 5, Khoản 1, Luật BHXH số 58/2014/QH13, mức đóng BHXH bắt buộc của người lao động là 8% tiền lương hàng tháng."
            },
            {
                "question": "Test?",
                "answer": "Ngắn quá"  # Will be bad
            }
        ]
        
        # When
        good, bad = evaluator.evaluate(qa_pairs)
        
        # Then
        assert len(good) == 1
        assert len(bad) == 1
        assert good[0]["eval_method"] == "rule"
    
    def test_evaluate_empty_list(self, evaluator):
        """
        Given: Empty Q&A list
        When: Evaluating
        Then: Should return empty lists
        """
        # When
        good, bad = evaluator.evaluate([])
        
        # Then
        assert good == []
        assert bad == []
    
    def test_deduplicate(self, evaluator):
        """
        Given: Q&A pairs with duplicates
        When: Deduplicating
        Then: Should remove duplicates
        """
        # Given
        qa_pairs = [
            {"question": "Câu hỏi 1?", "answer": "Trả lời ngắn"},
            {"question": "Câu hỏi 1?", "answer": "Trả lời dài hơn nhiều lần"},  # Duplicate
            {"question": "Câu hỏi 2?", "answer": "Trả lời khác"},
        ]
        
        # When
        unique, dups = evaluator._deduplicate(qa_pairs)
        
        # Then
        assert len(unique) == 2
        assert len(dups) == 1
        # Should keep the one with longer answer
        q1_item = [q for q in unique if "Câu hỏi 1" in q["question"]][0]
        assert "dài hơn" in q1_item["answer"]
    
    def test_deduplicate_by_answer(self, evaluator):
        """
        Given: Q&A pairs with same answer
        When: Deduplicating
        Then: Should remove duplicate answers
        """
        # Given
        qa_pairs = [
            {"question": "Câu hỏi A?", "answer": "Same answer here"},
            {"question": "Câu hỏi B?", "answer": "Same answer here"},  # Same answer
            {"question": "Câu hỏi C?", "answer": "Different answer"},
        ]
        
        # When
        unique, dups = evaluator._deduplicate(qa_pairs)
        
        # Then
        assert len(unique) == 2
        assert len(dups) == 1


class TestRuleScoreEdgeCases:
    """Test edge cases for rule scoring"""
    
    @pytest.fixture
    def evaluator(self, temp_dir: str):
        return QualityEvaluator({
            "mode": "rule",
            "min_score": 7,
            "output_dir": temp_dir
        })
    
    def test_missing_question_mark(self, evaluator):
        """
        Given: Question without ?
        When: Scoring
        Then: Should deduct points
        """
        # Given
        qa = {
            "question": "Mức đóng BHXH là bao nhiêu",  # No ?
            "answer": "Căn cứ Điều 5, Luật BHXH số 58/2014/QH13, mức đóng là 8% tiền lương."
        }
        
        # When
        score, reason = evaluator._rule_score(qa)
        
        # Then
        assert "?" in reason or score < 10  # Either mentioned or deducted
    
    def test_answer_with_numbers(self, evaluator):
        """
        Given: Answer with specific numbers
        When: Scoring
        Then: Should get bonus or no penalty
        """
        # Given
        qa = {
            "question": "Thời gian đóng BHXH tối thiểu là bao nhiêu năm?",
            "answer": "Căn cứ Điều 54, Luật BHXH số 41/2024/QH15, người lao động cần đóng BHXH tối thiểu 15 năm để được hưởng lương hưu."
        }
        
        # When
        score, reason = evaluator._rule_score(qa)
        
        # Then
        assert score >= 7  # Has number, should be OK
    
    def test_vague_reference_without_number(self, evaluator):
        """
        Given: Answer referencing "this law" without number
        When: Scoring
        Then: Should get penalty
        """
        # Given
        qa = {
            "question": "Ai là đối tượng áp dụng của luật này?",
            "answer": "Theo quy định của Luật này, đối tượng áp dụng bao gồm người lao động và người sử dụng lao động."
        }
        
        # When
        score, reason = evaluator._rule_score(qa)
        
        # Then
        assert score < 7
        assert "văn bản này" in reason.lower() or "số hiệu" in reason.lower()

