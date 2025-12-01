# -*- coding: utf-8 -*-
"""
Text Extractor - Trích xuất text từ các file tài liệu
"""

import os
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
        
        # Xử lý tuần tự nếu số file ít
        if len(files) <= 5:
            for file_path in tqdm(files, desc="Extracting"):
                doc = self._extract_file(file_path)
                if doc:
                    documents.append(doc)
        else:
            # Xử lý song song với nhiều file
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._extract_file, f): f for f in files}
                
                for future in tqdm(as_completed(futures), total=len(files), desc="Extracting"):
                    doc = future.result()
                    if doc:
                        documents.append(doc)
        
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
                return {
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "content": content.strip(),
                    "char_count": len(content),
                    "word_count": len(content.split())
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
