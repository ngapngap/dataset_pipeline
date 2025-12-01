# -*- coding: utf-8 -*-
"""
QA Generator - Sinh cặp câu hỏi-trả lời từ tài liệu
"""

import os
import json
import time
import random
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Lock
from tqdm import tqdm

from core.logger import get_logger
from core.utils import save_json, load_json, save_jsonl, chunk_text, CacheManager
from providers import create_providers_from_config


logger = get_logger(__name__)


# Prompt template mặc định cho dataset pháp luật
DEFAULT_PROMPT_TEMPLATE = """Bạn là chuyên gia tư vấn pháp luật. Dựa trên văn bản pháp luật sau, hãy tạo {num_qa} cặp câu hỏi-trả lời.

📄 VĂN BẢN PHÁP LUẬT:
{content}

⚠️ YÊU CẦU BẮT BUỘC:

1. CÂU HỎI phải thực tế như người dân hỏi:
   ✅ "Tôi đóng BHXH bao nhiêu năm thì được hưởng lương hưu?"
   ✅ "Mức đóng BHXH hàng tháng là bao nhiêu phần trăm?"
   ❌ "Văn bản này quy định về nội dung gì?" (quá chung)

2. CÂU TRẢ LỜI BẮT BUỘC có 2 phần:
   
   📌 PHẦN CĂN CỨ PHÁP LÝ (bắt buộc):
   - Trích dẫn chính xác: "Căn cứ Điều X, Khoản Y, Luật/Nghị định..."
   - Ghi rõ tên văn bản, số hiệu
   
   📌 PHẦN NỘI DUNG TRẢ LỜI:
   - Giải thích rõ ràng, dễ hiểu
   - Có số liệu cụ thể: %, số năm, số tiền (nếu có)

3. Nếu văn bản không có thông tin cụ thể → trả về []

OUTPUT FORMAT (JSON array):
[
  {{
    "question": "câu hỏi thực tế của người dân",
    "answer": "Căn cứ Điều X, Khoản Y của [Tên văn bản]:\\n\\n[Nội dung trả lời chi tiết với số liệu cụ thể]"
  }}
]

Chỉ trả về JSON array, không giải thích thêm."""


class QAGenerator:
    """
    Sinh cặp Q&A từ documents sử dụng LLM.
    Hỗ trợ đa luồng với mỗi API key 1 thread riêng.
    Có cache để tránh gọi lại API cho chunks đã xử lý.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Config cho QA generation
        """
        # Provider config
        self.provider_name = config.get("provider", "gemini")
        self.model = config.get("model", "gemini-2.0-flash")
        self.api_keys = config.get("api_keys", [])
        self.base_url = config.get("base_url", None)  # Cho custom providers
        self.generation_config = config.get("generation", {})
        
        # QA config
        self.num_qa_per_chunk = config.get("num_qa_per_chunk", 5)
        self.chunk_size = config.get("chunk_size", 3000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.prompt_template = config.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
        
        # Output config
        self.output_dir = config.get("output_dir", "data/generated")
        self.save_interval = config.get("save_interval", 50)  # Lưu sau mỗi N items
        
        # Delay config (để tránh rate limit)
        self.request_delay = config.get("request_delay", 1.0)
        
        # Cache config
        cache_config = config.get("cache", {})
        self.cache = CacheManager(
            cache_dir=cache_config.get("cache_dir", "./cache"),
            ttl_days=cache_config.get("ttl_days", 30),
            enabled=cache_config.get("enabled", True)
        )
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Tạo providers (1 provider / 1 API key)
        self.providers = []
        if self.api_keys:
            self.providers = create_providers_from_config(
                provider_name=self.provider_name,
                api_keys=self.api_keys,
                model=self.model,
                generation_config=self.generation_config,
                base_url=self.base_url
            )
            logger.info(f"Đã tạo {len(self.providers)} providers ({self.provider_name})")
        
        # Log cache status
        if self.cache.enabled:
            logger.info(f"📦 Cache ENABLED - Dir: {self.cache.cache_dir}, TTL: {self.cache.ttl_days} days")
        else:
            logger.info("📦 Cache DISABLED")
        
        # Tracking
        self._results_lock = Lock()
        self._all_qa_pairs = []
        self._failed_chunks = []
        self._cache_hits = 0
    
    def generate(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sinh Q&A từ danh sách documents
        
        Args:
            documents: List documents đã extract
        
        Returns:
            List tất cả Q&A pairs
        """
        if not self.providers:
            logger.error("Không có provider nào được cấu hình!")
            return []
        
        # Chuẩn bị chunks từ documents
        all_chunks = self._prepare_chunks(documents)
        logger.info(f"Tổng số chunks cần xử lý: {len(all_chunks)}")
        
        if not all_chunks:
            return []
        
        # Reset tracking
        self._all_qa_pairs = []
        self._failed_chunks = []
        self._cache_hits = 0
        
        # Chia chunks cho các providers (round-robin)
        chunk_queues = self._distribute_chunks(all_chunks)
        
        # Chạy đa luồng - mỗi provider 1 thread
        num_workers = len(self.providers)
        logger.info(f"Bắt đầu generate với {num_workers} threads")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i, provider in enumerate(self.providers):
                chunks = chunk_queues[i]
                future = executor.submit(
                    self._worker_generate,
                    provider=provider,
                    chunks=chunks,
                    worker_id=i + 1
                )
                futures.append(future)
            
            # Progress bar tổng
            pbar = tqdm(total=len(all_chunks), desc="Generating Q&A")
            
            # Đợi kết quả
            completed = 0
            while completed < len(all_chunks):
                time.sleep(1)
                new_completed = len(self._all_qa_pairs) + len(self._failed_chunks)
                if new_completed > completed:
                    pbar.update(new_completed - completed)
                    completed = new_completed
            
            pbar.close()
            
            # Đợi tất cả hoàn thành
            for future in futures:
                future.result()
        
        # Lưu kết quả
        self._save_results()
        
        # Log cache stats
        cache_stats = self.cache.get_stats()
        logger.info(f"📦 Cache stats: {cache_stats['hits']} hits, {cache_stats['misses']} misses ({cache_stats['hit_rate']} saved)")
        
        logger.info(f"Hoàn thành! Tổng: {len(self._all_qa_pairs)} Q&A pairs, {len(self._failed_chunks)} chunks thất bại")
        
        return self._all_qa_pairs
    
    def _prepare_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chuẩn bị chunks từ documents"""
        all_chunks = []
        
        for doc in documents:
            content = doc.get("content", "")
            if not content:
                continue
            
            # Chunk text
            text_chunks = chunk_text(
                content,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap
            )
            
            for i, chunk_text_content in enumerate(text_chunks):
                all_chunks.append({
                    "doc_name": doc.get("file_name", "unknown"),
                    "chunk_id": i,
                    "content": chunk_text_content
                })
        
        return all_chunks
    
    def _distribute_chunks(self, chunks: List[Dict]) -> List[List[Dict]]:
        """Phân phối chunks cho các providers (round-robin)"""
        num_providers = len(self.providers)
        queues = [[] for _ in range(num_providers)]
        
        for i, chunk in enumerate(chunks):
            provider_idx = i % num_providers
            queues[provider_idx].append(chunk)
        
        return queues
    
    def _worker_generate(self, provider, chunks: List[Dict], worker_id: int):
        """
        Worker function - xử lý chunks với 1 provider
        
        Args:
            provider: LLM provider
            chunks: Danh sách chunks cần xử lý
            worker_id: ID của worker
        """
        logger.debug(f"Worker {worker_id} bắt đầu với {len(chunks)} chunks")
        
        for chunk in chunks:
            try:
                qa_pairs = self._generate_qa_for_chunk(provider, chunk)
                
                if qa_pairs:
                    with self._results_lock:
                        self._all_qa_pairs.extend(qa_pairs)
                        
                        # Auto-save
                        if len(self._all_qa_pairs) % self.save_interval == 0:
                            self._save_intermediate()
                else:
                    with self._results_lock:
                        self._failed_chunks.append(chunk)
                
                # Delay để tránh rate limit
                time.sleep(self.request_delay)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} lỗi: {e}")
                with self._results_lock:
                    self._failed_chunks.append(chunk)
        
        logger.debug(f"Worker {worker_id} hoàn thành")
    
    def _generate_qa_for_chunk(self, provider, chunk: Dict) -> List[Dict[str, Any]]:
        """
        Sinh Q&A cho 1 chunk (có cache)
        
        Returns:
            List Q&A pairs hoặc None nếu lỗi
        """
        content = chunk["content"]
        
        # === CHECK CACHE FIRST ===
        cached_result = self.cache.get(
            content=content,
            prompt_template=self.prompt_template[:100],  # Hash phần đầu prompt
            model=self.model
        )
        
        if cached_result is not None:
            logger.debug(f"📦 Cache HIT for chunk {chunk['doc_name']}:{chunk['chunk_id']}")
            # Cập nhật metadata cho cached result
            for qa in cached_result:
                qa["source_doc"] = chunk["doc_name"]
                qa["chunk_id"] = chunk["chunk_id"]
                qa["from_cache"] = True
            return cached_result
        
        # === CACHE MISS → CALL API ===
        logger.debug(f"📦 Cache MISS for chunk {chunk['doc_name']}:{chunk['chunk_id']} → calling API")
        
        # Tạo prompt
        prompt = self.prompt_template.format(
            num_qa=self.num_qa_per_chunk,
            content=content
        )
        
        # Gọi LLM
        response = provider.generate(prompt)
        
        if not response:
            return None
        
        # Parse JSON từ response
        qa_pairs = self._parse_qa_response(response)
        
        if qa_pairs:
            # Thêm metadata
            for qa in qa_pairs:
                qa["source_doc"] = chunk["doc_name"]
                qa["chunk_id"] = chunk["chunk_id"]
            
            # === SAVE TO CACHE ===
            self.cache.set(
                content=content,
                result=qa_pairs,
                prompt_template=self.prompt_template[:100],
                model=self.model
            )
        
        return qa_pairs
    
    def _parse_qa_response(self, response: str) -> Optional[List[Dict]]:
        """Parse JSON response từ LLM"""
        try:
            # Tìm JSON array trong response
            response = response.strip()
            
            # Xử lý markdown code block
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()
            
            # Tìm [ và ]
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                qa_list = json.loads(json_str)
                
                # Validate
                valid_pairs = []
                for item in qa_list:
                    if isinstance(item, dict) and "question" in item and "answer" in item:
                        valid_pairs.append({
                            "question": item["question"].strip(),
                            "answer": item["answer"].strip()
                        })
                
                return valid_pairs if valid_pairs else None
            
            return None
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return None
    
    def _save_intermediate(self):
        """Lưu kết quả trung gian"""
        output_file = os.path.join(self.output_dir, "qa_intermediate.json")
        save_json(self._all_qa_pairs, output_file)
    
    def _save_results(self):
        """Lưu kết quả cuối cùng"""
        # Lưu JSON
        output_json = os.path.join(self.output_dir, "qa_generated.json")
        save_json(self._all_qa_pairs, output_json)
        
        # Lưu JSONL
        output_jsonl = os.path.join(self.output_dir, "qa_generated.jsonl")
        save_jsonl(self._all_qa_pairs, output_jsonl)
        
        # Lưu failed chunks
        if self._failed_chunks:
            failed_file = os.path.join(self.output_dir, "failed_chunks.json")
            save_json(self._failed_chunks, failed_file)
            logger.warning(f"Đã lưu {len(self._failed_chunks)} chunks thất bại -> {failed_file}")
        
        logger.info(f"Đã lưu {len(self._all_qa_pairs)} Q&A pairs -> {output_json}")
    
    def retry_failed(self) -> List[Dict[str, Any]]:
        """Thử lại các chunks thất bại"""
        failed_file = os.path.join(self.output_dir, "failed_chunks.json")
        
        if not os.path.exists(failed_file):
            logger.info("Không có chunks thất bại để retry")
            return []
        
        failed_chunks = load_json(failed_file)
        logger.info(f"Retry {len(failed_chunks)} chunks thất bại")
        
        # Reset và chạy lại
        self._failed_chunks = []
        
        # Tạo documents giả từ failed chunks
        fake_docs = [
            {"file_name": c["doc_name"], "content": c["content"]}
            for c in failed_chunks
        ]
        
        new_qa = self.generate(fake_docs)
        
        # Merge với kết quả cũ
        existing = load_json(os.path.join(self.output_dir, "qa_generated.json"))
        all_qa = existing + new_qa
        
        save_json(all_qa, os.path.join(self.output_dir, "qa_generated.json"))
        
        return new_qa
