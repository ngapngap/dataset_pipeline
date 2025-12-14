# -*- coding: utf-8 -*-
"""
Text Extractor - Trích xuất text từ các file tài liệu
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from core.logger import get_logger
from core.utils import save_json, load_json


logger = get_logger(__name__)


class TextExtractor:
    """
    Trích xuất text từ các file tài liệu.
    Hỗ trợ: .txt, .doc, .docx, .pdf
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Config cho text extraction
        """
        self.input_dir = config.get("input_dir", "data/raw")
        self.output_dir = config.get("output_dir", "data/extracted") 
        self.supported_extensions = config.get("extensions", [".txt", ".doc", ".docx", ".pdf"])
        self.max_workers = config.get("max_workers", 4)
        self.encoding = config.get("encoding", "utf-8")
        
        # Tạo output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def extract_all(self) -> List[Dict[str, Any]]:
        """
        Trích xuất text từ tất cả file trong input_dir
        
        Returns:
            List các document đã extract
        """
        # Tìm tất cả file
        files = self._find_files()
        logger.info(f"Tìm thấy {len(files)} file để trích xuất")
        
        if not files:
            logger.warning("Không tìm thấy file nào!")
            return []
        
        documents = []
        processed = 0
        total_files = len(files)
        
        # Ghi live status ban đầu
        self._save_live_status("extract", 0, total_files, 0)
        
        # Xử lý tuần tự nếu số file ít
        if len(files) <= 5:
            for file_path in tqdm(files, desc="Extracting"):
                doc = self._extract_file(file_path)
                if doc:
                    documents.append(doc)
                processed += 1
                # Cập nhật live status mỗi file
                self._save_live_status("extract", processed, total_files, len(documents))
        else:
            # Xử lý song song với nhiều file
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._extract_file, f): f for f in files}
                
                for future in tqdm(as_completed(futures), total=len(files), desc="Extracting"):
                    doc = future.result()
                    if doc:
                        documents.append(doc)
                    processed += 1
                    # Cập nhật live status mỗi 2 files
                    if processed % 2 == 0:
                        self._save_live_status("extract", processed, total_files, len(documents))
        
        # Lưu kết quả
        output_file = os.path.join(self.output_dir, "extracted_documents.json")
        save_json(documents, output_file)
        logger.info(f"Đã trích xuất {len(documents)} documents -> {output_file}")
        
        return documents
    
    def _find_files(self) -> List[Path]:
        """Tìm tất cả file được hỗ trợ trong input_dir"""
        files = []
        input_path = Path(self.input_dir)
        
        if not input_path.exists():
            logger.error(f"Thư mục không tồn tại: {self.input_dir}")
            return files
        
        for ext in self.supported_extensions:
            files.extend(input_path.glob(f"*{ext}"))
            files.extend(input_path.glob(f"**/*{ext}"))  # Recursive
        
        # Loại bỏ duplicate
        files = list(set(files))
        return files
    
    def _extract_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Trích xuất text từ 1 file
        
        Args:
            file_path: Đường dẫn file
        
        Returns:
            Document dict hoặc None nếu lỗi
        """
        ext = file_path.suffix.lower()
        
        try:
            if ext == ".txt":
                content = self._extract_txt(file_path)
            elif ext == ".docx":
                content = self._extract_docx(file_path)
            elif ext == ".doc":
                content = self._extract_doc(file_path)
            elif ext == ".pdf":
                content = self._extract_pdf(file_path)
            else:
                logger.warning(f"Không hỗ trợ định dạng: {ext}")
                return None
            
            if content and content.strip():
                # Trích xuất metadata từ 3000 ký tự đầu
                metadata = self._extract_metadata(content)
                
                return {
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "content": content.strip(),
                    "char_count": len(content),
                    "word_count": len(content.split()),
                    "metadata": metadata
                }
            else:
                logger.warning(f"File rỗng: {file_path.name}")
                return None
                
        except Exception as e:
            logger.error(f"Lỗi extract {file_path.name}: {e}")
            return None
    
    def _extract_txt(self, file_path: Path) -> str:
        """Extract từ file .txt"""
        encodings = [self.encoding, 'utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Không thể decode file: {file_path}")
    
    def _extract_docx(self, file_path: Path) -> str:
        """Extract từ file .docx"""
        try:
            from docx import Document
            doc = Document(str(file_path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)
        except ImportError:
            raise ImportError("Cần cài đặt: pip install python-docx")
    
    def _extract_doc(self, file_path: Path) -> str:
        """Extract từ file .doc (legacy Word)"""
        try:
            # Thử dùng antiword (nếu có)
            import subprocess
            result = subprocess.run(
                ['antiword', str(file_path)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout
        except:
            pass
        
        # Fallback: dùng textract
        try:
            import textract
            text = textract.process(str(file_path)).decode('utf-8')
            return text
        except ImportError:
            raise ImportError("Cần cài đặt: pip install textract")
    
    def _extract_pdf(self, file_path: Path) -> str:
        """Extract từ file .pdf"""
        try:
            import PyPDF2
            text_parts = []
            
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            
            return "\n".join(text_parts)
        except ImportError:
            try:
                # Fallback: dùng pdfplumber
                import pdfplumber
                text_parts = []
                
                with pdfplumber.open(str(file_path)) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                
                return "\n".join(text_parts)
            except ImportError:
                raise ImportError("Cần cài đặt: pip install PyPDF2 hoặc pip install pdfplumber")
    
    def load_extracted(self) -> List[Dict[str, Any]]:
        """Load documents đã extract từ file"""
        output_file = os.path.join(self.output_dir, "extracted_documents.json")
        if os.path.exists(output_file):
            return load_json(output_file)
        return []
    
    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """
        Trích xuất metadata từ 3000 ký tự đầu của văn bản:
        - Số hiệu văn bản
        - Ngày ban hành
        - Cơ quan ban hành
        - Loại văn bản
        - Tên văn bản
        """
        import re
        from datetime import datetime
        
        # Chỉ tìm trong 3000 ký tự đầu (phần header)
        header = content[:3000] if len(content) > 3000 else content
        header_normalized = re.sub(r'\s+', ' ', header)
        
        metadata = {
            "doc_number": None,
            "doc_number_short": None,
            "doc_type": None,
            "issued_date": None,
            "issued_by": None,
            "title": None
        }
        
        # ========== TRÍCH XUẤT SỐ HIỆU VĂN BẢN ==========
        patterns = [
            (r'Luật\s+số:\s*(\d+/\d{4}/QH\d+)', 'Luật số'),
            (r'Số:\s*(\d+/\d{4}/NĐ\s*-?\s*CP)', 'Nghị định số'),
            (r'Số:\s*(\d+/\d{4}/NĐ)', 'Nghị định số'),
            (r'Số:\s*(\d+/\d{4}/TT\s*-?\s*[A-Z]+)', 'Thông tư số'),
            (r'Số:\s*(\d+/QĐ\s*-?\s*[A-Z]+)', 'Quyết định số'),
            (r'Số:\s*(\d+/\d{4}/QH\d+)', 'Nghị quyết số'),
            (r'Số:\s*(\d+/[A-Z]+\s*-?\s*[A-Z]+)', 'Công văn số'),
        ]
        
        for pattern, doc_type in patterns:
            match = re.search(pattern, header_normalized, re.IGNORECASE)
            if match:
                doc_num = match.group(1)
                doc_num = re.sub(r'\s+', '', doc_num)
                # Fix: Nếu là NĐ nhưng thiếu -CP thì thêm vào
                if doc_type == 'Nghị định số' and 'NĐ' in doc_num and '-CP' not in doc_num:
                    doc_num = doc_num.replace('NĐ', 'NĐ-CP')
                
                metadata["doc_number"] = f"{doc_type} {doc_num}"
                metadata["doc_number_short"] = doc_num
                metadata["doc_type"] = doc_type
                break
        
        # ========== TRÍCH XUẤT NGÀY BAN HÀNH ==========
        date_patterns = [
            r'Hà Nội,?\s*ngày\s+(\d+)\s+tháng\s+(\d+)\s+năm\s+(\d{4})',
            r'ngày\s+(\d+)\s+tháng\s+(\d+)\s+năm\s+(\d{4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, header_normalized, re.IGNORECASE)
            if match:
                day, month, year = match.groups()
                try:
                    # Validate date
                    datetime(int(year), int(month), int(day))
                    metadata["issued_date"] = f"{day.zfill(2)}/{month.zfill(2)}/{year}"
                    break
                except ValueError:
                    continue
        
        # ========== TRÍCH XUẤT CƠ QUAN BAN HÀNH ==========
        issued_by_patterns = [
            (r'Chính phủ\s+ban hành', 'Chính phủ'),
            (r'Quốc hội\s+ban hành', 'Quốc hội'),
            (r'Bộ trưởng\s+Bộ\s+([A-Za-z\s-]+)\s+ban hành', lambda m: f"Bộ {m.group(1).strip()}"),
            (r'Giám đốc\s+([A-Za-z\s-]+)\s+ban hành', lambda m: f"Giám đốc {m.group(1).strip()}"),
            (r'Bộ\s+([A-Za-z\s-]+)\s+ban hành', lambda m: f"Bộ {m.group(1).strip()}"),
        ]
        
        for pattern, formatter in issued_by_patterns:
            match = re.search(pattern, header_normalized, re.IGNORECASE)
            if match:
                if callable(formatter):
                    metadata["issued_by"] = formatter(match)
                else:
                    metadata["issued_by"] = formatter
                break
        
        # ========== TRÍCH XUẤT TÊN VĂN BẢN ==========
        # Tìm dòng sau "NGHỊ ĐỊNH", "THÔNG TƯ", "LUẬT", "QUYẾT ĐỊNH"
        doc_type_keywords = ['NGHỊ ĐỊNH', 'THÔNG TƯ', 'LUẬT', 'QUYẾT ĐỊNH']
        lines = header.split('\n')
        
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()
            if any(keyword in line_upper for keyword in doc_type_keywords):
                # Lấy dòng tiếp theo làm tên văn bản
                if i + 1 < len(lines):
                    title = lines[i + 1].strip()
                    if title and len(title) > 10:  # Bỏ qua dòng quá ngắn
                        metadata["title"] = title[:200]  # Giới hạn 200 ký tự
                        break
        
        return metadata
    
    def _save_live_status(self, step: str, processed: int, total: int, docs_count: int):
        """Ghi file live status cho Dashboard real-time"""
        try:
            # Ghi vào thư mục cha của output_dir (thường là ./output/)
            parent_dir = os.path.dirname(self.output_dir)
            status_file = os.path.join(parent_dir, "live_status.json")
            
            status = {
                "step": step,
                "chunks_processed": processed,
                "chunks_total": total,
                "qa_generated": 0,
                "docs_extracted": docs_count,
                "cache_hits": 0,
                "cache_misses": 0,
                "failed_chunks": 0,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, ensure_ascii=False)
        except Exception:
            pass  # Không để lỗi ảnh hưởng pipeline
