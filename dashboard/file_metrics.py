# -*- coding: utf-8 -*-
"""
File-based Metrics Reader - Đọc metrics từ files mà Pipeline ghi ra

Giải quyết vấn đề: Dashboard và Pipeline chạy trong 2 process riêng biệt,
không thể share in-memory MetricsCollector.
"""

import os
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime


class FileMetricsReader:
    """
    Đọc metrics từ các file output của pipeline.
    
    Files được theo dõi:
    - output/pipeline_state.json - Trạng thái tổng quan
    - output/generated/qa_intermediate.json - Q&A đang generate
    - output/evaluated/qa_good.json - Q&A đã đánh giá tốt
    - output/evaluated/qa_bad.json - Q&A đã đánh giá xấu
    """
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self._last_check = 0
        self._cached_stats = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Đọc stats từ files"""
        now = time.time()

        # Cache 0.2 giây để real-time hơn (giảm từ 0.5s)
        if now - self._last_check < 0.2 and self._cached_stats:
            return self._cached_stats
        
        self._last_check = now
        
        stats = {
            "current_step": "idle",
            "elapsed_seconds": 0,
            "elapsed_formatted": "0s",
            "chunks": {
                "total": 0,
                "processed": 0,
                "success": 0,
                "failed": 0,
                "progress_percent": 0
            },
            "qa": {
                "generated": 0,
                "good": 0,
                "bad": 0,
                "rescued": 0,
                "good_rate": 0
            },
            "cache": {
                "hits": 0,
                "misses": 0,
                "hit_rate": 0
            },
            "performance": {
                "processing_rate": 0,
                "estimated_remaining": "N/A"
            },
            "errors": {
                "count": 0,
                "recent": []
            },
            "source": "file"  # Đánh dấu đọc từ file
        }
        
        # ĐỌC LIVE STATUS TRƯỚC (cập nhật real-time từ generator/extractor)
        live = self._read_json("live_status.json")
        if live:
            step = live.get("step", "idle")
            stats["current_step"] = step
            stats["chunks"]["total"] = live.get("chunks_total", 0)
            stats["chunks"]["processed"] = live.get("chunks_processed", 0)
            stats["qa"]["generated"] = live.get("qa_generated", 0)
            stats["qa"]["good"] = live.get("qa_good", 0)
            stats["qa"]["bad"] = live.get("qa_bad", 0)
            stats["qa"]["rescued"] = live.get("qa_rescued", 0)
            stats["cache"]["hits"] = live.get("cache_hits", 0)
            stats["cache"]["misses"] = live.get("cache_misses", 0)
            stats["chunks"]["failed"] = live.get("failed_chunks", 0)

            # Thêm docs_extracted
            stats["docs_extracted"] = live.get("docs_extracted", 0)

            # Thời gian
            elapsed = live.get("elapsed_seconds", 0)
            stats["elapsed_seconds"] = elapsed
            stats["elapsed_formatted"] = self._format_duration(elapsed)

            # Progress
            if stats["chunks"]["total"] > 0:
                stats["chunks"]["progress_percent"] = (
                    stats["chunks"]["processed"] / stats["chunks"]["total"] * 100
                )

            # Cache hit rate
            total_cache = stats["cache"]["hits"] + stats["cache"]["misses"]
            if total_cache > 0:
                stats["cache"]["hit_rate"] = stats["cache"]["hits"] / total_cache * 100

            # QA good rate
            total_qa_eval = stats["qa"]["good"] + stats["qa"]["bad"]
            if total_qa_eval > 0:
                stats["qa"]["good_rate"] = stats["qa"]["good"] / total_qa_eval * 100

            # Processing rate và ETA từ live_status hoặc tính từ elapsed
            processing_rate = live.get("processing_rate", 0)
            if processing_rate == 0 and elapsed > 0 and stats["chunks"]["processed"] > 0:
                processing_rate = (stats["chunks"]["processed"] / elapsed) * 60
            stats["performance"]["processing_rate"] = processing_rate

            # ETA
            remaining_chunks = stats["chunks"]["total"] - stats["chunks"]["processed"]
            if processing_rate > 0 and remaining_chunks > 0:
                eta_seconds = (remaining_chunks / processing_rate) * 60
                stats["performance"]["estimated_remaining"] = self._format_duration(eta_seconds)
            elif stats["chunks"]["processed"] >= stats["chunks"]["total"] and stats["chunks"]["total"] > 0:
                stats["performance"]["estimated_remaining"] = "Done"

            stats["source"] = "live"
            self._cached_stats = stats
            return stats
        
        # Fallback: Đọc pipeline_state.json
        state = self._read_json("pipeline_state.json")
        if state:
            stats["current_step"] = state.get("current_step", "idle")
            stats["chunks"]["total"] = state.get("documents_count", 0) * 5  # Ước tính
            stats["qa"]["generated"] = state.get("qa_pairs_count", 0)
            stats["qa"]["good"] = state.get("good_qa_count", 0)
            stats["qa"]["bad"] = state.get("bad_qa_count", 0)
            
            # Tính started time
            started_at = state.get("started_at")
            if started_at:
                try:
                    start_time = datetime.fromisoformat(started_at)
                    elapsed = (datetime.now() - start_time).total_seconds()
                    stats["elapsed_seconds"] = elapsed
                    stats["elapsed_formatted"] = self._format_duration(elapsed)
                except:
                    pass
        
        # Đọc qa_intermediate.json để biết số Q&A đang generate
        intermediate = self._read_json("generated/qa_intermediate.json")
        if intermediate and isinstance(intermediate, list):
            stats["qa"]["generated"] = len(intermediate)
            stats["chunks"]["processed"] = len(set(
                (q.get("source_doc"), q.get("chunk_id")) 
                for q in intermediate
            ))
        
        # Đọc qa_generated.json
        generated = self._read_json("generated/qa_generated.json")
        if generated and isinstance(generated, list):
            stats["qa"]["generated"] = len(generated)
            
            # Đếm từ cache
            from_cache = sum(1 for q in generated if q.get("from_cache"))
            stats["cache"]["hits"] = from_cache
            stats["cache"]["misses"] = len(generated) - from_cache
            if generated:
                stats["cache"]["hit_rate"] = from_cache / len(generated) * 100
        
        # Đọc qa_good.json và qa_bad.json
        good = self._read_json("evaluated/qa_good.json")
        if good and isinstance(good, list):
            stats["qa"]["good"] = len(good)
        
        bad = self._read_json("evaluated/qa_bad.json")
        if bad and isinstance(bad, list):
            stats["qa"]["bad"] = len(bad)
        
        # Tính good_rate
        total_eval = stats["qa"]["good"] + stats["qa"]["bad"]
        if total_eval > 0:
            stats["qa"]["good_rate"] = stats["qa"]["good"] / total_eval * 100
        
        # Đọc failed_chunks.json
        failed = self._read_json("generated/failed_chunks.json")
        if failed and isinstance(failed, list):
            stats["chunks"]["failed"] = len(failed)
            stats["errors"]["count"] = len(failed)
        
        # Đọc extracted_documents.json để biết số docs
        extracted = self._read_json("extracted/extracted_documents.json")
        if extracted and isinstance(extracted, list):
            # Tính tổng chunks
            total_chunks = 0
            for doc in extracted:
                content = doc.get("content", "")
                # Ước tính số chunks (chunk_size=4000)
                total_chunks += max(1, len(content) // 3800)
            stats["chunks"]["total"] = total_chunks
            
            if stats["chunks"]["total"] > 0:
                stats["chunks"]["progress_percent"] = (
                    stats["chunks"]["processed"] / stats["chunks"]["total"] * 100
                )
        
        self._cached_stats = stats
        return stats
    
    def _read_json(self, relative_path: str) -> Optional[Any]:
        """Đọc file JSON"""
        file_path = self.output_dir / relative_path
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    # Compatibility methods với MetricsCollector interface
    def reset(self):
        self._cached_stats = {}
    
    def add_update_callback(self, callback):
        pass  # File-based không cần callback
    
    def remove_update_callback(self, callback):
        pass


# Global instance
_file_metrics: Optional[FileMetricsReader] = None


def get_file_metrics(output_dir: str = "./output") -> FileMetricsReader:
    """Get global FileMetricsReader instance"""
    global _file_metrics
    if _file_metrics is None:
        _file_metrics = FileMetricsReader(output_dir)
    return _file_metrics

