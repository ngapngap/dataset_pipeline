# -*- coding: utf-8 -*-
"""
Utility Functions
"""

from __future__ import annotations

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple
from datetime import datetime, timedelta


def load_api_keys(filepath: str) -> List[str]:
    """Load API keys từ file
    
    Args:
        filepath: Đường dẫn file chứa API keys (mỗi key 1 dòng)
        
    Returns:
        List các API keys
    """
    if not os.path.exists(filepath):
        return []
    
    keys = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                keys.append(line)
    return keys


def get_document_name(filename: str, mappings: Dict[str, str]) -> str:
    """Lấy tên đầy đủ của văn bản từ filename
    
    Args:
        filename: Tên file
        mappings: Dict mapping pattern → full name
        
    Returns:
        Tên đầy đủ hoặc filename gốc
    """
    for pattern, full_name in mappings.items():
        if pattern in filename:
            return full_name
    return filename


def normalize_text(text: str) -> str:
    """Chuẩn hóa text (lowercase, remove extra spaces)"""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def compute_hash(text: str) -> str:
    """Tính hash của text"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:12]


def save_json(data: Any, filepath: str, ensure_ascii: bool = False):
    """Lưu data thành JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=2)


def load_json(filepath: str) -> Any:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_jsonl(data: List[Dict], filepath: str):
    """Lưu data thành JSONL file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_file_list(directory: str, extensions: List[str] = None, 
                  ignore_patterns: List[str] = None) -> List[str]:
    """Lấy danh sách files trong thư mục
    
    Args:
        directory: Thư mục cần scan
        extensions: List các extension cho phép (vd: ['.txt', '.pdf'])
        ignore_patterns: List patterns để bỏ qua
        
    Returns:
        List đường dẫn files
    """
    if not os.path.exists(directory):
        return []
    
    extensions = extensions or ['.txt', '.pdf', '.doc', '.docx']
    ignore_patterns = ignore_patterns or []
    
    files = []
    for filename in os.listdir(directory):
        # Check extension
        if not any(filename.lower().endswith(ext) for ext in extensions):
            continue
        
        # Check ignore patterns
        should_ignore = False
        for pattern in ignore_patterns:
            if pattern.endswith('*'):
                if filename.startswith(pattern[:-1]):
                    should_ignore = True
                    break
            elif '*' in pattern:
                # Simple wildcard matching
                regex = pattern.replace('*', '.*')
                if re.match(regex, filename):
                    should_ignore = True
                    break
            elif pattern in filename:
                should_ignore = True
                break
        
        if not should_ignore:
            files.append(os.path.join(directory, filename))
    
    return sorted(files)


def format_timestamp(dt: datetime = None) -> str:
    """Format datetime thành string"""
    dt = dt or datetime.now()
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def parse_json_from_text(text: str) -> Optional[List[Dict]]:
    """Extract và parse JSON array từ text
    
    Args:
        text: Text có thể chứa JSON
        
    Returns:
        Parsed JSON array hoặc None
    """
    if not text:
        return None
    
    # Tìm JSON array trong text
    patterns = [
        r'\[[\s\S]*\]',  # Greedy match
        r'\[\s*\{[\s\S]*?\}\s*\]',  # Match array of objects
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                continue
    
    return None


def estimate_time(total_tasks: int, tasks_per_minute: float) -> str:
    """Ước tính thời gian hoàn thành
    
    Args:
        total_tasks: Tổng số tasks
        tasks_per_minute: Tốc độ xử lý
        
    Returns:
        String mô tả thời gian
    """
    if tasks_per_minute <= 0:
        return "N/A"
    
    minutes = total_tasks / tasks_per_minute
    
    if minutes < 1:
        return f"~{int(minutes * 60)} giây"
    elif minutes < 60:
        return f"~{int(minutes)} phút"
    else:
        hours = minutes / 60
        return f"~{hours:.1f} giờ"


def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 200) -> List[str]:
    """Chia text thành các chunks với overlap
    
    Args:
        text: Text cần chia
        chunk_size: Kích thước mỗi chunk (characters)
        overlap: Số ký tự overlap giữa các chunks
        
    Returns:
        List các text chunks
    """
    if not text:
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Tìm điểm cắt hợp lý (cuối câu hoặc đoạn)
        if end < len(text):
            # Ưu tiên cắt ở cuối đoạn
            paragraph_end = text.rfind('\n\n', start, end)
            if paragraph_end > start + chunk_size // 2:
                end = paragraph_end + 2
            else:
                # Hoặc cuối câu
                sentence_end = max(
                    text.rfind('. ', start, end),
                    text.rfind('.\n', start, end),
                    text.rfind('? ', start, end),
                    text.rfind('! ', start, end)
                )
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start với overlap
        start = end - overlap if end < len(text) else end
    
    return chunks


def deduplicate_qa(qa_list: List[Dict], 
                   question_key: str = "question",
                   similarity_threshold: float = 0.9) -> List[Dict]:
    """Loại bỏ các Q&A trùng lặp
    
    Args:
        qa_list: List Q&A pairs
        question_key: Key chứa question
        similarity_threshold: Ngưỡng tương đồng để coi là trùng
        
    Returns:
        List Q&A đã loại bỏ trùng lặp
    """
    if not qa_list:
        return []
    
    seen_hashes = set()
    unique_qa = []
    
    for qa in qa_list:
        question = qa.get(question_key, "")
        # Normalize và hash
        normalized = normalize_text(question)
        q_hash = compute_hash(normalized)
        
        if q_hash not in seen_hashes:
            seen_hashes.add(q_hash)
            unique_qa.append(qa)
    
    return unique_qa


# ==============================================================================
# 📦 CACHE MANAGER - Tránh gọi lại API cho chunks đã xử lý
# ==============================================================================

class CacheManager:
    """
    Quản lý cache để tránh gọi lại API cho chunks đã xử lý.
    
    Lợi ích:
    - Pipeline crash → không tốn tiền cho chunks đã xử lý
    - Chạy lại pipeline → skip chunks đã có
    - Debug/test → miễn phí
    
    Cách hoạt động:
    - Hash nội dung chunk → cache key
    - Lưu kết quả vào file JSON
    - Tự động xóa cache quá hạn
    
    Attributes:
        cache_dir: Thư mục lưu cache
        ttl_days: Số ngày giữ cache
        enabled: Bật/tắt cache
        hits: Số lần cache hit
        misses: Số lần cache miss
    """
    
    cache_dir: str
    ttl_days: int
    enabled: bool
    hits: int
    misses: int
    
    def __init__(
        self, 
        cache_dir: str = "./cache", 
        ttl_days: int = 30, 
        enabled: bool = True
    ) -> None:
        """
        Args:
            cache_dir: Thư mục lưu cache
            ttl_days: Số ngày giữ cache (0 = không xóa)
            enabled: Bật/tắt cache
        """
        self.cache_dir = cache_dir
        self.ttl_days = ttl_days
        self.enabled = enabled
        
        # Stats
        self.hits = 0
        self.misses = 0
        
        if self.enabled:
            os.makedirs(cache_dir, exist_ok=True)
            self._cleanup_expired()
    
    def _get_cache_key(self, content: str, prompt_template: str = "", model: str = "") -> str:
        """
        Tạo cache key từ nội dung chunk + prompt + model.
        
        Thay đổi prompt hoặc model → cache key khác → không dùng cache cũ
        """
        combined = f"{content}|{prompt_template}|{model}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Đường dẫn file cache"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def get(self, content: str, prompt_template: str = "", model: str = "") -> Optional[List[Dict]]:
        """
        Lấy kết quả từ cache nếu có.
        
        Args:
            content: Nội dung chunk
            prompt_template: Template prompt (để phân biệt khi đổi prompt)
            model: Tên model (để phân biệt khi đổi model)
            
        Returns:
            Cached result hoặc None nếu không có
        """
        if not self.enabled:
            return None
        
        cache_key = self._get_cache_key(content, prompt_template, model)
        cache_path = self._get_cache_path(cache_key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check TTL
                if self.ttl_days > 0:
                    created_at = datetime.fromisoformat(data.get("created_at", "2000-01-01"))
                    if datetime.now() - created_at > timedelta(days=self.ttl_days):
                        os.remove(cache_path)
                        return None
                
                self.hits += 1
                return data.get("result")
                
            except (json.JSONDecodeError, KeyError, ValueError):
                # Cache file bị lỗi → xóa
                os.remove(cache_path)
                return None
        
        self.misses += 1
        return None
    
    def set(self, content: str, result: List[Dict], prompt_template: str = "", model: str = ""):
        """
        Lưu kết quả vào cache.
        
        Args:
            content: Nội dung chunk
            result: Kết quả Q&A pairs
            prompt_template: Template prompt
            model: Tên model
        """
        if not self.enabled:
            return
        
        cache_key = self._get_cache_key(content, prompt_template, model)
        cache_path = self._get_cache_path(cache_key)
        
        data = {
            "created_at": datetime.now().isoformat(),
            "content_hash": hashlib.md5(content.encode()).hexdigest()[:8],
            "model": model,
            "result": result
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # Không crash nếu lưu cache lỗi
            pass
    
    def exists(self, content: str, prompt_template: str = "", model: str = "") -> bool:
        """
        Kiểm tra xem cache có tồn tại không (không load data).
        Dùng để đếm nhanh số chunks đã cache.
        
        Args:
            content: Nội dung chunk
            prompt_template: Template prompt
            model: Tên model
            
        Returns:
            True nếu cache tồn tại và còn hạn
        """
        if not self.enabled:
            return False
        
        cache_key = self._get_cache_key(content, prompt_template, model)
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return False
        
        # Check TTL nếu cần
        if self.ttl_days > 0:
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
                if datetime.now() - mtime > timedelta(days=self.ttl_days):
                    return False
            except Exception:
                return False
        
        return True
    
    def _cleanup_expired(self) -> None:
        """Xóa cache files quá hạn"""
        if self.ttl_days <= 0:
            return
        
        if not os.path.exists(self.cache_dir):
            return
        
        expired_count = 0
        cutoff = datetime.now() - timedelta(days=self.ttl_days)
        
        for filename in os.listdir(self.cache_dir):
            if not filename.endswith('.json'):
                continue
            
            filepath = os.path.join(self.cache_dir, filename)
            
            try:
                # Check file modification time (nhanh hơn đọc JSON)
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                if mtime < cutoff:
                    os.remove(filepath)
                    expired_count += 1
            except Exception:
                pass
        
        if expired_count > 0:
            print(f"🗑️ Đã xóa {expired_count} cache files quá hạn")
    
    def invalidate(self, cache_key: str) -> None:
        """
        Xóa một cache entry cụ thể theo key.
        
        Args:
            cache_key: Cache key (16 ký tự hex) cần xóa
        """
        if not self.enabled:
            return
        
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except Exception:
                pass
    
    def clear(self) -> None:
        """Xóa toàn bộ cache"""
        if os.path.exists(self.cache_dir):
            import shutil
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê cache"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        # Đếm số file cache
        cache_files = 0
        cache_size = 0
        if os.path.exists(self.cache_dir):
            for f in os.listdir(self.cache_dir):
                if f.endswith('.json'):
                    cache_files += 1
                    cache_size += os.path.getsize(os.path.join(self.cache_dir, f))
        
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_files": cache_files,
            "cache_size_mb": f"{cache_size / 1024 / 1024:.2f} MB"
        }
