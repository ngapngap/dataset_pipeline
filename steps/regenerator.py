# -*- coding: utf-8 -*-
"""
Chunk Regenerator - Tái tạo Q&A cho các chunks có chất lượng kém
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

from core.logger import get_logger
from core.utils import save_json, load_json

logger = get_logger(__name__)


class ChunkRegenerator:
    """
    Phân tích và xác định chunks cần regenerate.
    
    Một chunk cần regenerate nếu:
    1. Tỷ lệ good Q&A < threshold (mặc định 50%)
    2. Chưa có Q&A nào được tạo
    3. Tất cả Q&A đều bị đánh giá bad
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Config cho regenerator
        """
        self.output_dir = config.get("output_dir", "./output")
        self.min_good_rate = config.get("min_good_rate", 0.5)  # 50%
        self.min_good_count = config.get("min_good_count", 1)  # Ít nhất 1 good Q&A
        
        # Tracking
        self._chunks_to_regenerate = []
        self._chunk_stats = {}
    
    def analyze(self, good_qa: List[Dict], bad_qa: List[Dict], 
                all_chunks: List[Dict] = None) -> Dict[str, Any]:
        """
        Phân tích chunks và xác định chunks cần regenerate
        
        Args:
            good_qa: List good Q&A pairs
            bad_qa: List bad Q&A pairs  
            all_chunks: List tất cả chunks (optional, để phát hiện chunks chưa được generate)
        
        Returns:
            Dict chứa thống kê và danh sách chunks cần regenerate
        """
        logger.info("Phân tích chunks để xác định cần regenerate...")
        
        # Gom Q&A theo source_doc và chunk_id
        chunk_qa = defaultdict(lambda: {"good": [], "bad": []})
        
        for qa in good_qa:
            key = (qa.get("source_doc", ""), qa.get("chunk_id", 0))
            chunk_qa[key]["good"].append(qa)
        
        for qa in bad_qa:
            key = (qa.get("source_doc", ""), qa.get("chunk_id", 0))
            chunk_qa[key]["bad"].append(qa)
        
        # Phân tích từng chunk
        self._chunks_to_regenerate = []
        self._chunk_stats = {}
        
        total_chunks = len(chunk_qa)
        good_chunks = 0
        bad_chunks = 0
        
        for (source_doc, chunk_id), qa_data in chunk_qa.items():
            good_count = len(qa_data["good"])
            bad_count = len(qa_data["bad"])
            total = good_count + bad_count
            
            if total == 0:
                continue
            
            good_rate = good_count / total
            
            # Lưu stats
            self._chunk_stats[(source_doc, chunk_id)] = {
                "source_doc": source_doc,
                "chunk_id": chunk_id,
                "good_count": good_count,
                "bad_count": bad_count,
                "total": total,
                "good_rate": good_rate
            }
            
            # Xác định chunks cần regenerate
            needs_regenerate = False
            reason = ""
            
            if good_count < self.min_good_count:
                needs_regenerate = True
                reason = f"Chỉ có {good_count} good Q&A (yêu cầu >= {self.min_good_count})"
            elif good_rate < self.min_good_rate:
                needs_regenerate = True
                reason = f"Tỷ lệ good chỉ {good_rate*100:.1f}% (yêu cầu >= {self.min_good_rate*100:.0f}%)"
            
            if needs_regenerate:
                bad_chunks += 1
                self._chunks_to_regenerate.append({
                    "source_doc": source_doc,
                    "chunk_id": chunk_id,
                    "good_count": good_count,
                    "bad_count": bad_count,
                    "good_rate": good_rate,
                    "reason": reason
                })
            else:
                good_chunks += 1
        
        # Kiểm tra chunks chưa được generate (nếu có danh sách all_chunks)
        missing_chunks = []
        if all_chunks:
            generated_keys = set(chunk_qa.keys())
            for chunk in all_chunks:
                key = (chunk.get("source_doc", ""), chunk.get("chunk_id", 0))
                if key not in generated_keys:
                    missing_chunks.append({
                        "source_doc": chunk.get("source_doc", ""),
                        "chunk_id": chunk.get("chunk_id", 0),
                        "reason": "Chưa được generate"
                    })
            
            self._chunks_to_regenerate.extend(missing_chunks)
        
        # Lưu danh sách chunks cần regenerate
        self._save_regenerate_list()
        
        stats = {
            "total_chunks_analyzed": total_chunks,
            "good_chunks": good_chunks,
            "bad_chunks": bad_chunks,
            "missing_chunks": len(missing_chunks),
            "chunks_to_regenerate": len(self._chunks_to_regenerate),
            "regenerate_rate": len(self._chunks_to_regenerate) / max(1, total_chunks) * 100
        }
        
        logger.info(f"Kết quả phân tích:")
        logger.info(f"  - Tổng chunks: {total_chunks}")
        logger.info(f"  - Chunks tốt: {good_chunks}")
        logger.info(f"  - Chunks cần regenerate: {len(self._chunks_to_regenerate)}")
        if missing_chunks:
            logger.info(f"  - Chunks chưa generate: {len(missing_chunks)}")
        
        return stats
    
    def get_chunks_to_regenerate(self) -> List[Dict]:
        """Trả về danh sách chunks cần regenerate"""
        return self._chunks_to_regenerate
    
    def get_regenerate_chunk_ids(self) -> List[Tuple[str, int]]:
        """Trả về list (source_doc, chunk_id) cần regenerate"""
        return [(c["source_doc"], c["chunk_id"]) for c in self._chunks_to_regenerate]
    
    def _save_regenerate_list(self):
        """Lưu danh sách chunks cần regenerate"""
        if not self._chunks_to_regenerate:
            return
        
        output_file = os.path.join(self.output_dir, "evaluated", "chunks_to_regenerate.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        save_json(self._chunks_to_regenerate, output_file)
        logger.info(f"Đã lưu danh sách {len(self._chunks_to_regenerate)} chunks cần regenerate -> {output_file}")
    
    def load_regenerate_list(self) -> List[Dict]:
        """Load danh sách chunks cần regenerate từ file"""
        input_file = os.path.join(self.output_dir, "evaluated", "chunks_to_regenerate.json")
        if os.path.exists(input_file):
            return load_json(input_file)
        return []
    
    def get_stats(self) -> Dict:
        """Trả về thống kê chi tiết"""
        return {
            "total_analyzed": len(self._chunk_stats),
            "to_regenerate": len(self._chunks_to_regenerate),
            "chunks": self._chunks_to_regenerate
        }


def get_chunks_needing_regenerate(good_file: str, bad_file: str, 
                                   min_good_rate: float = 0.5) -> List[Tuple[str, int]]:
    """
    Utility function để lấy danh sách chunks cần regenerate
    
    Args:
        good_file: Path tới file qa_good.json
        bad_file: Path tới file qa_bad.json (hoặc qa_still_bad.json)
        min_good_rate: Tỷ lệ good tối thiểu
    
    Returns:
        List of (source_doc, chunk_id) tuples
    """
    good_qa = load_json(good_file) if os.path.exists(good_file) else []
    bad_qa = load_json(bad_file) if os.path.exists(bad_file) else []
    
    regenerator = ChunkRegenerator({
        "min_good_rate": min_good_rate,
        "output_dir": os.path.dirname(os.path.dirname(good_file))
    })
    
    regenerator.analyze(good_qa, bad_qa)
    
    return regenerator.get_regenerate_chunk_ids()
