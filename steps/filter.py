# -*- coding: utf-8 -*-
"""
QA Filter - Lọc bỏ các QA có câu hỏi không có ngữ cảnh rõ ràng
"""

import re
from typing import List, Dict, Any, Tuple

from core.logger import get_logger

logger = get_logger(__name__)


class QAFilter:
    """
    Filter QA pairs để loại bỏ những câu hỏi không có ngữ cảnh rõ ràng.
    
    Loại bỏ:
    - Câu hỏi lý thuyết: "Trong văn bản này, BHXH được viết tắt cho cụm từ gì?"
    - Câu hỏi chung chung: "Văn bản này quy định về nội dung gì?"
    - Câu hỏi không có ngữ cảnh thực tế
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: Config cho filter (có thể mở rộng sau)
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        
        # Patterns cho câu hỏi chất lượng thấp
        self.low_quality_patterns = [
            # "Trong văn bản này, X được viết tắt cho cụm từ gì?"
            r'trong\s+văn\s+bản\s+này.*được\s+viết\s+tắt',
            r'được\s+viết\s+tắt\s+cho\s+cụm\s+từ',
            r'chữ\s+viết\s+tắt',
            
            # "Văn bản này quy định về nội dung gì?"
            r'văn\s+bản\s+này\s+quy\s+định\s+về',
            r'văn\s+bản\s+này\s+nói\s+về',
            r'văn\s+bản\s+này\s+có\s+nội\s+dung',
            r'nội\s+dung.*văn\s+bản\s+này',
            
            # "Văn bản này là gì?"
            r'văn\s+bản\s+này\s+là\s+gì',
            
            # "Điều X của văn bản này quy định gì?"
            r'điều.*của\s+văn\s+bản\s+này\s+quy\s+định',
            
            # Câu hỏi quá chung chung
            r'văn\s+bản\s+này.*là\s+gì',
            r'văn\s+bản\s+này.*quy\s+định\s+gì',
        ]
    
    def is_low_quality_question(self, question: str) -> bool:
        """
        Kiểm tra câu hỏi có phải là câu hỏi lý thuyết, không có ngữ cảnh rõ ràng không
        
        Args:
            question: Câu hỏi cần kiểm tra
            
        Returns:
            True nếu là câu hỏi chất lượng thấp (cần loại bỏ)
        """
        if not self.enabled:
            return False
        
        question_lower = question.lower()
        return any(re.search(pattern, question_lower) for pattern in self.low_quality_patterns)
    
    def filter(self, qa_pairs: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter QA list, loại bỏ những câu hỏi chất lượng thấp
        
        Args:
            qa_pairs: List Q&A pairs cần filter
            
        Returns:
            (good_qa, removed_qa): Tuple 2 lists
        """
        if not self.enabled:
            logger.info("Filter đã tắt, giữ lại tất cả QA")
            return qa_pairs, []
        
        good_qa = []
        removed_qa = []
        
        for qa in qa_pairs:
            question = qa.get("question", "")
            if self.is_low_quality_question(question):
                removed_qa.append(qa)
            else:
                good_qa.append(qa)
        
        if removed_qa:
            logger.info(f"Đã loại bỏ {len(removed_qa)} QA có câu hỏi không có ngữ cảnh rõ ràng")
            # Log một vài ví dụ
            for i, qa in enumerate(removed_qa[:3]):
                logger.debug(f"  [{i+1}] {qa.get('question', '')[:80]}...")
        
        return good_qa, removed_qa

