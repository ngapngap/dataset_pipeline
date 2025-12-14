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
        self.extracted_docs_path = config.get("extracted_docs_path", "./output/extracted/extracted_documents.json")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load documents để trích số hiệu từ nội dung
        self._documents = self._load_documents()
        
        # Metadata cache (được tạo trong _load_documents)
        if not hasattr(self, '_documents_metadata'):
            self._documents_metadata = {}
        
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
    
    def _load_documents(self) -> Dict[str, Dict]:
        """
        Load documents từ extracted_documents.json để trích số hiệu từ nội dung
        Returns: Dict {file_name: document}
        """
        if not os.path.exists(self.extracted_docs_path):
            logger.warning(f"Không tìm thấy extracted documents: {self.extracted_docs_path}")
            return {}
        
        try:
            documents = load_json(self.extracted_docs_path)
            doc_dict = {doc.get("file_name", ""): doc for doc in documents}
            
            # Cache metadata riêng để truy cập nhanh
            self._documents_metadata = {
                file_name: doc.get("metadata", {})
                for file_name, doc in doc_dict.items()
            }
            
            return doc_dict
        except Exception as e:
            logger.error(f"Lỗi load documents: {e}")
            return {}
    
    def _extract_doc_number(self, source_doc: str) -> Optional[Dict]:
        """
        Trích xuất số hiệu văn bản - ƯU TIÊN dùng metadata đã extract sẵn.
        Fallback về parse từ content hoặc filename nếu không có metadata.
        """
        # ƯU TIÊN 1: Dùng metadata đã extract sẵn từ Extractor
        metadata = getattr(self, '_documents_metadata', {}).get(source_doc, {})
        if metadata and metadata.get("doc_number"):
            return {
                "number": metadata.get("doc_number_short", ""),
                "type": metadata.get("doc_type", ""),
                "full_info": metadata  # Lưu toàn bộ metadata để dùng sau
            }
        
        # FALLBACK 1: Trích từ nội dung văn bản
        doc = self._documents.get(source_doc)
        if doc and doc.get("content"):
            content = doc.get("content", "")
            doc_number = self._extract_doc_number_from_content(content)
            if doc_number and doc_number != "(Không xác định số hiệu)":
                # Parse doc_number để tạo doc_info
                return self._parse_doc_number_string(doc_number)
        
        # FALLBACK 2: Trích từ tên file (giữ lại logic cũ)
        return self._extract_doc_number_from_filename(source_doc)
    
    def _extract_doc_number_from_content(self, content: str) -> str:
        """
        Trích xuất số hiệu văn bản từ nội dung (giống generator)
        """
        # Chỉ tìm trong 3000 ký tự đầu (phần header)
        header = content[:3000] if len(content) > 3000 else content
        
        # Normalize: loại bỏ khoảng trắng thừa
        header_normalized = re.sub(r'\s+', ' ', header)
        
        # Patterns để tìm số hiệu văn bản (theo thứ tự ưu tiên)
        patterns = [
            # Luật: Luật số: 41/2024/QH15
            (r'Luật\s+số:\s*(\d+/\d{4}/QH\d+)', 'Luật số'),
            # Nghị định: Số: 143/2018/NĐ-CP
            (r'Số:\s*(\d+/\d{4}/NĐ\s*-?\s*CP)', 'Nghị định số'),
            # Thông tư: Số: 12/2025/TT-BNV
            (r'Số:\s*(\d+/\d{4}/TT\s*-?\s*[A-Z]+)', 'Thông tư số'),
            # Quyết định: Số: 2222/QĐ-BHXH
            (r'Số:\s*(\d+/QĐ\s*-?\s*[A-Z]+)', 'Quyết định số'),
            # Nghị quyết: Số: 190/2025/QH15
            (r'Số:\s*(\d+/\d{4}/QH\d+)', 'Nghị quyết số'),
            # Công văn: Số: 1234/BHXH-CSXH
            (r'Số:\s*(\d+/[A-Z]+\s*-?\s*[A-Z]+)', 'Công văn số'),
        ]
        
        for pattern, doc_type in patterns:
            match = re.search(pattern, header_normalized, re.IGNORECASE)
            if match:
                doc_num = match.group(1)
                # Normalize số hiệu: loại bỏ khoảng trắng
                doc_num = re.sub(r'\s+', '', doc_num)
                return f"{doc_type} {doc_num}"
        
        return "(Không xác định số hiệu)"
    
    def _parse_doc_number_string(self, doc_number_str: str) -> Optional[Dict]:
        """
        Parse chuỗi doc_number thành dict {"number": "...", "type": "..."}
        Ví dụ: "Nghị định số 143/2018/NĐ-CP" -> {"number": "143/2018/NĐ-CP", "type": "Nghị định số"}
        """
        if not doc_number_str or doc_number_str == "(Không xác định số hiệu)":
            return None
        
        # Extract number và type
        match = re.match(r'^(.+?)\s+(\d+/.+)$', doc_number_str)
        if match:
            doc_type = match.group(1).strip()
            number = match.group(2).strip()
            return {
                "number": number,
                "type": doc_type
            }
        
        return None
    
    def _extract_doc_number_from_filename(self, source_doc: str) -> Optional[Dict]:
        """
        Fallback: Trích xuất số hiệu văn bản từ tên file (logic cũ)
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
        # Nếu có "văn bản không có số hiệu" thì coi như chưa có
        if "văn bản không có số hiệu" in answer.lower():
            return False
        
        patterns = [
            r'\d+/\d{4}/[A-Za-zĐ]+-?[A-Za-z]*',  # 11/2025/TT-BNV
            r'\d+/\d{4}/NĐ-CP',                   # Nghị định
            r'\d+/\d{4}/TT-[A-Z]+',               # Thông tư  
            r'\d+/\d{4}/QH\d+',                   # Luật
            r'\d+/QĐ-[A-Z]+',                     # Quyết định (2222/QĐ-BHXH)
        ]
        return any(re.search(p, answer) for p in patterns)
    
    def _add_doc_number_to_answer(self, answer: str, doc_info: Dict) -> str:
        """
        Thêm số hiệu văn bản vào câu trả lời.
        Nếu có metadata đầy đủ (ngày, cơ quan), sẽ thêm vào để answer phong phú hơn.
        """
        doc_number = doc_info["number"]
        doc_type = doc_info["type"]

        # Lấy metadata đầy đủ nếu có
        metadata = doc_info.get("full_info", {})
        issued_date = metadata.get("issued_date")
        issued_by = metadata.get("issued_by")

        # Tạo chuỗi số hiệu đầy đủ (có thể thêm ngày, cơ quan)
        doc_number_full = f"{doc_type} {doc_number}"
        if issued_date:
            doc_number_full += f" ngày {issued_date}"
        if issued_by:
            doc_number_full += f" của {issued_by}"

        # Patterns để tìm vị trí cần chèn (theo thứ tự ưu tiên)
        patterns_to_fix = [
            # ========== THAY THẾ "văn bản không có số hiệu" ==========
            # Theo điểm X khoản Y Điều Z của văn bản không có số hiệu -> ... của Quyết định số 166/QĐ-BHXH
            (r'(Theo\s+điểm\s+[a-zđ]\s+khoản\s+\d+[a-zđ]?\s*Điều\s+\d+[a-zđ]?\s*của)\s+văn\s+bản\s+không\s+có\s+số\s+hiệu',
             rf'\1 {doc_number_full}'),
            # Theo khoản X Điều Y của văn bản không có số hiệu
            (r'(Theo\s+khoản\s+\d+[a-zđ]?\s*Điều\s+\d+[a-zđ]?\s*của)\s+văn\s+bản\s+không\s+có\s+số\s+hiệu',
             rf'\1 {doc_number_full}'),
            # Theo Điều X, Khoản Y của văn bản không có số hiệu
            (r'(Theo\s+Điều\s+\d+[a-zđ]?\s*,?\s*Khoản\s+\d+[a-zđ]?\s*của)\s+văn\s+bản\s+không\s+có\s+số\s+hiệu',
             rf'\1 {doc_number_full}'),
            # Theo Điều X của văn bản không có số hiệu
            (r'(Theo\s+Điều\s+\d+[a-zđ]?\s*của)\s+văn\s+bản\s+không\s+có\s+số\s+hiệu',
             rf'\1 {doc_number_full}'),
            # Căn cứ điểm X khoản Y Điều Z của văn bản không có số hiệu
            (r'(Căn cứ\s+điểm\s+[a-zđ]\s+khoản\s+\d+[a-zđ]?\s*Điều\s+\d+[a-zđ]?\s*của)\s+văn\s+bản\s+không\s+có\s+số\s+hiệu',
             rf'\1 {doc_number_full}'),
            # Căn cứ khoản X Điều Y của văn bản không có số hiệu
            (r'(Căn cứ\s+khoản\s+\d+[a-zđ]?\s*Điều\s+\d+[a-zđ]?\s*của)\s+văn\s+bản\s+không\s+có\s+số\s+hiệu',
             rf'\1 {doc_number_full}'),
            # Căn cứ Điều X, Khoản Y của văn bản không có số hiệu
            (r'(Căn cứ\s+Điều\s+\d+[a-zđ]?\s*,?\s*Khoản\s+\d+[a-zđ]?\s*của)\s+văn\s+bản\s+không\s+có\s+số\s+hiệu',
             rf'\1 {doc_number_full}'),
            # Căn cứ Điều X của văn bản không có số hiệu
            (r'(Căn cứ\s+Điều\s+\d+[a-zđ]?\s*của)\s+văn\s+bản\s+không\s+có\s+số\s+hiệu',
             rf'\1 {doc_number_full}'),

            # ========== MỚI: THAY THẾ "của văn bản không có số hiệu" (rộng hơn) ==========
            # Theo hướng dẫn tại mục X của văn bản không có số hiệu
            (r'(Theo\s+hướng\s+dẫn\s+tại\s+mục\s+[\d.]+\s*của)\s+văn\s+bản\s+không\s+có\s+số\s+hiệu',
             rf'\1 {doc_number_full}'),
            # Theo Mục X, Điểm Y của văn bản không có số hiệu
            (r'(Theo\s+[Mm]ục\s+[\d.]+\s*,?\s*Điểm\s+[a-zđ]\s*của)\s+văn\s+bản\s+không\s+có\s+số\s+hiệu',
             rf'\1 {doc_number_full}'),
            # Theo Điểm X.Y Điều Z của văn bản không có số hiệu
            (r'(Theo\s+Điểm\s+[\d.]+\s*Điều\s+\d+\s*của)\s+văn\s+bản\s+không\s+có\s+số\s+hiệu',
             rf'\1 {doc_number_full}'),
            # Căn cứ Khoản X của văn bản không có số hiệu
            (r'(Căn\s+cứ\s+[Kk]hoản\s+\d+\s*của)\s+văn\s+bản\s+không\s+có\s+số\s+hiệu',
             rf'\1 {doc_number_full}'),
            # ban hành kèm theo văn bản không có số hiệu
            (r'(ban\s+hành\s+kèm\s+theo)\s+văn\s+bản\s+không\s+có\s+số\s+hiệu',
             rf'\1 {doc_number_full}'),
            # của văn bản không có số hiệu (general catch-all)
            (r'của\s+văn\s+bản\s+không\s+có\s+số\s+hiệu',
             rf'của {doc_number_full}'),

            # ========== MỚI: THAY THẾ "Nghị định chưa rõ số hiệu" ==========
            (r'(của)\s+(Nghị định|Thông tư|Luật|Quyết định)\s+chưa\s+rõ\s+số\s+hiệu',
             rf'\1 {doc_type} {doc_number}'),
            (r'(Nghị định|Thông tư|Luật|Quyết định)\s+chưa\s+rõ\s+số\s+hiệu',
             rf'{doc_type} {doc_number}'),

            # ========== MỚI: THAY THẾ "của Nghị định," không có số ==========
            # Căn cứ Khoản X Điều Y của Nghị định,
            (r'(Căn\s+cứ\s+[Kk]hoản\s+\d+[a-zđ]?\s*Điều\s+\d+[a-zđ]?\s*của)\s+(Nghị định|Thông tư|Luật|Quyết định)\s*,',
             rf'\1 {doc_type} {doc_number},'),
            # Căn cứ Điều X của Nghị định,
            (r'(Căn\s+cứ\s+Điều\s+\d+[a-zđ]?\s*của)\s+(Nghị định|Thông tư|Luật|Quyết định)\s*,',
             rf'\1 {doc_type} {doc_number},'),
            # Theo Khoản X Điều Y của Nghị định,
            (r'(Theo\s+[Kk]hoản\s+\d+[a-zđ]?\s*Điều\s+\d+[a-zđ]?\s*của)\s+(Nghị định|Thông tư|Luật|Quyết định)\s*,',
             rf'\1 {doc_type} {doc_number},'),
            # Theo Điều X của Nghị định, (NEW)
            (r'(Theo\s+Điều\s+\d+[a-zđ]?\s*của)\s+(Nghị định|Thông tư|Luật|Quyết định)\s*,',
             rf'\1 {doc_type} {doc_number},'),
            # Theo Điểm X Khoản Y Điều Z của Nghị định, (NEW)
            (r'(Theo\s+Điểm\s+[a-zđ]\s+[Kk]hoản\s+\d+[a-zđ]?\s*Điều\s+\d+[a-zđ]?\s*của)\s+(Nghị định|Thông tư|Luật|Quyết định)\s*,',
             rf'\1 {doc_type} {doc_number},'),
            # Căn cứ theo Khoản X Điều Y của Nghị định, (NEW - có "theo" ở giữa)
            (r'(Căn\s+cứ\s+theo\s+[Kk]hoản\s+\d+[a-zđ]?\s*Điều\s+\d+[a-zđ]?\s*của)\s+(Nghị định|Thông tư|Luật|Quyết định)\s*,',
             rf'\1 {doc_type} {doc_number},'),

            # ========== MỚI: THAY THẾ "Điều X của Nghị định này" ==========
            (r'(Điều\s+\d+[a-zđ]?\s*của)\s+(Nghị định|Thông tư|Luật|Quyết định)\s+này',
             rf'\1 {doc_type} {doc_number}'),
            # Phụ lục X kèm theo Thông tư này (NEW)
            (r'([Pp]hụ\s+lục\s+[IVX\d]+\s+kèm\s+theo)\s+(Nghị định|Thông tư|Luật|Quyết định)\s+này',
             rf'\1 {doc_type} {doc_number}'),
            # quy định tại Điều X của Nghị định này (NEW)
            (r'(quy\s+định\s+tại\s+Điều\s+\d+[a-zđ]?\s*của)\s+(Nghị định|Thông tư|Luật|Quyết định)\s+này',
             rf'\1 {doc_type} {doc_number}'),
            # quy định tại Phụ lục X kèm theo Thông tư này (NEW)
            (r'(quy\s+định\s+tại\s+[Pp]hụ\s+lục\s+[IVX\d]+\s+kèm\s+theo)\s+(Nghị định|Thông tư|Luật|Quyết định)\s+này',
             rf'\1 {doc_type} {doc_number}'),

            # ========== MỚI: THAY THẾ "Theo khoản X của văn bản" ==========
            (r'(Theo\s+[Kk]hoản\s+\d+\s*của)\s+văn\s+bản',
             rf'\1 {doc_number_full}'),

            # ========== MỚI: THAY THẾ "Theo hướng dẫn tại mục X," ==========
            (r'(Theo\s+hướng\s+dẫn\s+tại\s+mục\s+[\d.]+)\s*,',
             rf'\1 {doc_type} {doc_number},'),

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

            # ========== MỚI: Theo ... Thông tư này/Nghị định này ==========
            (r'(Theo\s+[Kk]hoản\s+\d+[a-zđ]?\s*Điều\s+\d+[a-zđ]?)\s+(Nghị định|Thông tư|Luật)\s+này',
             rf'\1 {doc_type} {doc_number}'),
            (r'(Theo\s+Điều\s+\d+[a-zđ]?)\s+(Nghị định|Thông tư|Luật)\s+này',
             rf'\1 {doc_type} {doc_number}'),

            # ========== MỚI: Thông tư hướng dẫn (không có số) ==========
            (r'(Theo\s+[Kk]hoản\s+\d+[a-zđ]?\s*Điều\s+\d+[a-zđ]?)\s+(Thông tư|Nghị định)\s+hướng\s+dẫn',
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
