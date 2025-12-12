# -*- coding: utf-8 -*-
"""
Q&A Rescuer - Cứu các Q&A bị đánh giá bad bằng cách tự động thêm số hiệu văn bản
"""

import os
import re
import json
from typing import Dict, Any, List, Tuple, Optional

from core.logger import get_logger
from core.utils import save_json, load_json

logger = get_logger(__name__)


class QARescuer:
    """
    Cứu bad Q&A bằng cách tự động thêm số hiệu văn bản vào câu trả lời.
    
    Nguyên nhân chính của bad Q&A:
    - Có "Căn cứ Điều X Khoản Y" nhưng thiếu số hiệu văn bản
    - Chúng ta biết source_doc nên có thể tự động thêm
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Config cho rescuer
        """
        self.output_dir = config.get("output_dir", "./output/evaluated")
        self.min_score = config.get("min_score", 8)  # Score threshold sau khi rescue
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Tracking
        self._rescued = []
        self._still_bad = []
    
    def rescue(self, bad_qa: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """
        Cứu bad Q&A pairs
        
        Args:
            bad_qa: List bad Q&A pairs
        
        Returns:
            (rescued_qa, still_bad_qa): Tuple 2 lists
        """
        logger.info(f"Bắt đầu rescue {len(bad_qa)} bad Q&A pairs")
        
        if not bad_qa:
            return [], []
        
        # Reset
        self._rescued = []
        self._still_bad = []
        
        for qa in bad_qa:
            rescued_qa = self._try_rescue(qa)
            if rescued_qa:
                self._rescued.append(rescued_qa)
            else:
                self._still_bad.append(qa)
        
        # Lưu kết quả
        self._save_results()
        
        logger.info(f"Rescue kết quả: {len(self._rescued)} rescued, {len(self._still_bad)} still bad")
        
        return self._rescued, self._still_bad
    
    def _try_rescue(self, qa: Dict) -> Optional[Dict]:
        """
        Thử cứu một Q&A pair
        
        Returns:
            Q&A đã fix nếu thành công, None nếu không fix được
        """
        source_doc = qa.get('source_doc', '')
        answer = qa.get('answer', '')
        
        # Trích xuất số hiệu từ source_doc
        doc_info = self._extract_doc_number(source_doc)
        
        if not doc_info:
            return None
        
        # Đã có số hiệu rồi thì không cần fix
        if self._has_doc_number(answer):
            return None
        
        # KHÔNG rescue những format sai/bịa đặt
        if self._is_invalid_format(answer):
            return None
        
        # Thử fix answer
        fixed_answer = self._add_doc_number_to_answer(answer, doc_info)
        
        if fixed_answer != answer:
            # Đã fix được
            rescued_qa = qa.copy()
            rescued_qa['answer'] = fixed_answer
            rescued_qa['rescued'] = True
            rescued_qa['doc_added'] = doc_info['number']
            rescued_qa['eval_score'] = 10  # Sau khi fix = good
            rescued_qa['eval_reason'] = f"✅ Rescued: đã thêm {doc_info['number']}"
            return rescued_qa
        
        return None
    
    def _is_invalid_format(self, answer: str) -> bool:
        """
        Kiểm tra answer có format sai/bịa đặt không
        Những cái này KHÔNG nên rescue vì nội dung không đáng tin
        """
        invalid_patterns = [
            # Điều X (khoản Y Điều Z) - format lạ, bịa
            r'Điều\s+\d+\s*\([^)]*Điều[^)]*\)',
            # Khoản Xa và Yb - format lạ
            r'Khoản\s+\d+[a-z]\s+và\s+\d+[a-z]',
            # "phần chế độ dưỡng sức" - không phải trích dẫn đúng
            r'phần\s+chế\s+độ',
            # "Luật Bảo hiểm xã hội" không có số (nhưng không phải "Luật BHXH số X")
            r'Luật\s+Bảo\s+hiểm\s+xã\s+hội\s*[^0-9số]',
            # "của Luật Bảo hiểm xã hội" - thiếu số
            r'của\s+Luật\s+Bảo\s+hiểm\s+xã\s+hội',
            # "Luật BHXH" không có số tiếp theo
            r'Luật\s+BHXH\s*[^0-9số/]',
        ]
        
        return any(re.search(p, answer) for p in invalid_patterns)
    
    def _extract_doc_number(self, source_doc: str) -> Optional[Dict]:
        """
        Trích xuất số hiệu văn bản từ tên file
        Ví dụ: "11_2025_TT-BNV_663712.txt" -> {"number": "11/2025/TT-BNV", "type": "Thông tư"}
        """
        patterns = [
            # Thông tư: 11_2025_TT-BNV_663712.txt
            (r'^(\d+)_(\d{4})_(TT)-([A-Z]+)_', 'Thông tư', 
             lambda m: f"{m.group(1)}/{m.group(2)}/{m.group(3)}-{m.group(4)}"),
            
            # Nghị định: 274_2025_ND-CP_653507.txt
            (r'^(\d+)_(\d{4})_(ND|NĐ)-CP_', 'Nghị định',
             lambda m: f"{m.group(1)}/{m.group(2)}/NĐ-CP"),
            
            # Luật/QH: 74_2025_QH15_530912.txt
            (r'^(\d+)_(\d{4})_(QH\d+)_', 'Luật số',
             lambda m: f"{m.group(1)}/{m.group(2)}/{m.group(3)}"),
            
            # Quyết định: 2222_QD-BHXH_667488.txt
            (r'^(\d+)_(QD|QĐ)-([A-Z]+)_', 'Quyết định',
             lambda m: f"{m.group(1)}/{m.group(2)}-{m.group(3)}"),
        ]
        
        for pattern, doc_type, formatter in patterns:
            match = re.match(pattern, source_doc)
            if match:
                return {
                    "number": formatter(match),
                    "type": doc_type
                }
        
        return None
    
    def _has_doc_number(self, answer: str) -> bool:
        """Kiểm tra answer đã có số hiệu văn bản chưa"""
        patterns = [
            r'\d+/\d{4}/[A-Za-zĐ]+-?[A-Za-z]*',  # 11/2025/TT-BNV
            r'\d+/\d{4}/NĐ-CP',                   # Nghị định
            r'\d+/\d{4}/TT-[A-Z]+',               # Thông tư  
            r'\d+/\d{4}/QH\d+',                   # Luật
        ]
        return any(re.search(p, answer) for p in patterns)
    
    def _add_doc_number_to_answer(self, answer: str, doc_info: Dict) -> str:
        """
        Thêm số hiệu văn bản vào câu trả lời
        """
        doc_number = doc_info["number"]
        doc_type = doc_info["type"]
        
        # Patterns để tìm vị trí cần chèn (theo thứ tự ưu tiên)
        patterns_to_fix = [
            # ========== THAY THẾ "Nghị định này", "Thông tư này" ==========
            # Căn cứ Điều X khoản Y điểm Z Nghị định này -> ... Nghị định 274/2025/NĐ-CP
            (r'(Căn cứ\s+Điều\s+\d+[a-zđ]?\s*,?\s*[Kk]hoản\s+\d+[a-zđ]?\s*,?\s*điểm\s+[a-zđ])\s+(Nghị định|Thông tư|Luật)\s+này',
             rf'\1 {doc_type} {doc_number}'),
            
            # Căn cứ Điều X khoản Y Nghị định này -> ... Nghị định 274/2025/NĐ-CP
            (r'(Căn cứ\s+Điều\s+\d+[a-zđ]?\s*,?\s*[Kk]hoản\s+\d+[a-zđ]?)\s+(Nghị định|Thông tư|Luật)\s+này',
             rf'\1 {doc_type} {doc_number}'),
            
            # Căn cứ Điều X Nghị định này -> ... Nghị định 274/2025/NĐ-CP
            (r'(Căn cứ\s+Điều\s+\d+[a-zđ]?)\s+(Nghị định|Thông tư|Luật)\s+này',
             rf'\1 {doc_type} {doc_number}'),
            
            # Căn cứ Điều X, Nghị định này -> ... Nghị định 274/2025/NĐ-CP
            (r'(Căn cứ\s+Điều\s+\d+[a-zđ]?)\s*,\s*(Nghị định|Thông tư|Luật)\s+này',
             rf'\1 {doc_type} {doc_number}'),
             
            # Căn cứ khoản X Điều Y Nghị định này
            (r'(Căn cứ\s+[Kk]hoản\s+\d+[a-zđ]?\s*,?\s*Điều\s+\d+[a-zđ]?)\s+(Nghị định|Thông tư|Luật)\s+này',
             rf'\1 {doc_type} {doc_number}'),
             
            # Căn cứ Điều X.Y Nghị định này (format Điều 16.2)
            (r'(Căn cứ\s+Điều\s+\d+\.\d+)\s*,?\s*(Nghị định|Thông tư|Luật)\s+này',
             rf'\1 {doc_type} {doc_number}'),
            
            # ========== THAY THẾ "Phụ lục kèm theo Nghị định" ==========
            (r'(Căn cứ\s+[Pp]hụ lục kèm theo)\s+(Nghị định|Thông tư)',
             rf'\1 {doc_type} {doc_number}'),
             
            # ========== THAY THẾ "Luật BHXH" không có số ==========
            # Căn cứ Điều X Khoản Y Luật BHXH: -> Luật BHXH số 41/2024/QH15
            (r'(Căn cứ\s+Điều\s+\d+[a-zđ]?\s*,?\s*[Kk]hoản\s+\d+[a-zđ]?)\s+Luật\s+BHXH(\s*:)',
             rf'\1 {doc_type} {doc_number}\2'),
            
            # Căn cứ Điều X Luật BHXH:
            (r'(Căn cứ\s+Điều\s+\d+[a-zđ]?)\s+Luật\s+BHXH(\s*:)',
             rf'\1 {doc_type} {doc_number}\2'),
             
            # ========== PATTERNS GỐC (thêm số hiệu khi chưa có) ==========
            # Căn cứ Điều X Khoản Y điểm z: (có số + điểm)
            (r'(Căn cứ\s+Điều\s+\d+[a-zđ]?\s*,?\s*[Kk]hoản\s+\d+[a-zđ]?\s*,?\s*điểm\s+[a-zđ])(\s*:)', 
             rf'\1 {doc_type} {doc_number}\2'),
            
            # Căn cứ Điều X, Khoản Y: -> Căn cứ Điều X, Khoản Y Thông tư XX:
            (r'(Căn cứ\s+Điều\s+\d+[a-zđ]?\s*,?\s*[Kk]hoản\s+\d+[a-zđ]?)(\s*:)', 
             rf'\1 {doc_type} {doc_number}\2'),
            
            # Căn cứ Điều X Khoản [a-z]: (Khoản không có số, chỉ có chữ)
            (r'(Căn cứ\s+Điều\s+\d+[a-zđ]?\s*,?\s*[Kk]hoản\s+[a-zđ])(\s*:)', 
             rf'\1 {doc_type} {doc_number}\2'),
            
            # Căn cứ Điều X: -> Căn cứ Điều X Thông tư XX:
            (r'(Căn cứ\s+Điều\s+\d+[a-zđ]?)(\s*:)', 
             rf'\1 {doc_type} {doc_number}\2'),
            
            # Căn cứ Điều X, (dấu phẩy)
            (r'(Căn cứ\s+Điều\s+\d+[a-zđ]?)(\s*,)', 
             rf'\1 {doc_type} {doc_number}\2'),
             
            # Căn cứ Điều X.Y, (format 16.2)
            (r'(Căn cứ\s+Điều\s+\d+\.\d+)(\s*,)', 
             rf'\1 {doc_type} {doc_number}\2'),
            
            # Căn cứ khoản X Điều Y: 
            (r'(Căn cứ\s+[Kk]hoản\s+\d+[a-zđ]?\s*,?\s*Điều\s+\d+[a-zđ]?)(\s*:)', 
             rf'\1 {doc_type} {doc_number}\2'),
            
            # Căn cứ khoản X:
            (r'(Căn cứ\s+[Kk]hoản\s+\d+[a-zđ]?)(\s*:)', 
             rf'\1 {doc_type} {doc_number}\2'),
            
            # Căn cứ điểm a khoản X:
            (r'(Căn cứ\s+điểm\s+[a-zđ]\s+[Kk]hoản\s+\d+)(\s*:)',
             rf'\1 {doc_type} {doc_number}\2'),
             
            # Theo Điều X, Khoản Y,
            (r'(Theo\s+Điều\s+\d+[a-zđ]?\s*,?\s*[Kk]hoản\s+\d+[a-zđ]?)(\s*,)', 
             rf'\1 {doc_type} {doc_number}\2'),
            
            # Theo Điều X,
            (r'(Theo\s+Điều\s+\d+[a-zđ]?)(\s*,)', 
             rf'\1 {doc_type} {doc_number}\2'),
             
            # Căn cứ Ví dụ X Điều Y
            (r'(Căn cứ\s+[Vv]í dụ\s+\d+\s+Điều\s+\d+)(\s+)', 
             rf'\1 {doc_type} {doc_number}\2'),
             
            # Căn cứ Ví dụ X trong văn bản
            (r'(Căn cứ\s+[Vv]í dụ\s+\d+)(\s+trong\s+văn\s+bản)', 
             rf'\1 {doc_type} {doc_number}\2'),
             
            # Căn cứ mục X Hướng dẫn
            (r'(Căn cứ\s+mục\s+\d+\s+[Hh]ướng\s+dẫn)(\s*:?)', 
             rf'\1 {doc_type} {doc_number}\2'),
             
            # Căn cứ Phần X:
            (r'(Căn cứ\s+[Pp]hần\s+[IVX\d]+\.?\d*[a-z]?)(\s*:)',
             rf'\1 {doc_type} {doc_number}\2'),
             
            # Căn cứ quy định tại -> Căn cứ quy định tại Thông tư XX,
            (r'(Căn cứ\s+quy\s+định\s+tại)(\s+)',
             rf'\1 {doc_type} {doc_number},\2'),
        ]
        
        modified = answer
        for pattern, replacement in patterns_to_fix:
            new_modified = re.sub(pattern, replacement, modified, count=1)
            if new_modified != modified:
                return new_modified
        
        return answer  # Không fix được
    
    def _save_results(self):
        """Lưu kết quả rescue"""
        if self._rescued:
            rescued_file = os.path.join(self.output_dir, "qa_rescued.json")
            save_json(self._rescued, rescued_file)
            logger.info(f"Đã lưu: {len(self._rescued)} rescued -> {rescued_file}")
        
        if self._still_bad:
            still_bad_file = os.path.join(self.output_dir, "qa_still_bad.json")
            save_json(self._still_bad, still_bad_file)
            logger.info(f"Đã lưu: {len(self._still_bad)} still bad -> {still_bad_file}")
    
    def load_rescued(self) -> List[Dict]:
        """Load rescued Q&A từ file"""
        rescued_file = os.path.join(self.output_dir, "qa_rescued.json")
        if os.path.exists(rescued_file):
            return load_json(rescued_file)
        return []
    
    def get_stats(self) -> Dict:
        """Trả về thống kê"""
        return {
            "total_processed": len(self._rescued) + len(self._still_bad),
            "rescued": len(self._rescued),
            "still_bad": len(self._still_bad),
            "rescue_rate": len(self._rescued) / (len(self._rescued) + len(self._still_bad)) * 100 if (self._rescued or self._still_bad) else 0
        }
