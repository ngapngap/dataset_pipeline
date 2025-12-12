# -*- coding: utf-8 -*-
"""
Metrics Collector - Thu thập và lưu trữ metrics từ pipeline

Thread-safe metrics storage với callback hooks cho pipeline events.
"""

from __future__ import annotations

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque


@dataclass
class ErrorRecord:
    """Record một lỗi"""
    timestamp: str
    error_type: str
    message: str
    chunk_info: Optional[str] = None


@dataclass 
class ChunkRecord:
    """Record một chunk đã xử lý"""
    timestamp: float
    success: bool
    duration: float
    qa_count: int = 0


class MetricsCollector:
    """
    Thu thập và lưu trữ metrics từ pipeline.
    
    Thread-safe để có thể được gọi từ nhiều workers.
    
    Features:
    - Track chunks processed
    - Track Q&A generated
    - Track errors
    - Calculate processing rate
    - Estimate remaining time
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Args:
            max_history: Số records tối đa lưu trong history
        """
        self._lock = threading.RLock()
        self.max_history = max_history
        
        # Current state
        self.current_step: str = "idle"
        self.started_at: Optional[float] = None
        
        # Chunk processing
        self.chunks_total: int = 0
        self.chunks_processed: int = 0
        self.chunks_success: int = 0
        self.chunks_failed: int = 0
        
        # Q&A stats
        self.qa_generated: int = 0
        self.qa_good: int = 0
        self.qa_bad: int = 0
        
        # Cache stats
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        
        # History (for rate calculation)
        self._chunk_history: deque = deque(maxlen=max_history)
        self._error_history: deque = deque(maxlen=100)
        
        # Callbacks
        self._update_callbacks: List[Callable] = []
    
    def reset(self) -> None:
        """Reset all metrics"""
        with self._lock:
            self.current_step = "idle"
            self.started_at = None
            self.chunks_total = 0
            self.chunks_processed = 0
            self.chunks_success = 0
            self.chunks_failed = 0
            self.qa_generated = 0
            self.qa_good = 0
            self.qa_bad = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self._chunk_history.clear()
            self._error_history.clear()
    
    def start_step(self, step: str, total_chunks: int = 0) -> None:
        """Bắt đầu một step mới"""
        with self._lock:
            self.current_step = step
            self.started_at = time.time()
            self.chunks_total = total_chunks
            self.chunks_processed = 0
            self._notify_update()
    
    def end_step(self) -> None:
        """Kết thúc step hiện tại"""
        with self._lock:
            self.current_step = "completed"
            self._notify_update()
    
    def record_chunk(
        self, 
        success: bool, 
        duration: float,
        qa_count: int = 0,
        from_cache: bool = False
    ) -> None:
        """Ghi nhận một chunk đã xử lý"""
        with self._lock:
            self.chunks_processed += 1
            
            if success:
                self.chunks_success += 1
                self.qa_generated += qa_count
            else:
                self.chunks_failed += 1
            
            if from_cache:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            
            # Add to history
            self._chunk_history.append(ChunkRecord(
                timestamp=time.time(),
                success=success,
                duration=duration,
                qa_count=qa_count
            ))
            
            self._notify_update()
    
    def record_qa_evaluated(self, good: int, bad: int) -> None:
        """Ghi nhận kết quả evaluate"""
        with self._lock:
            self.qa_good += good
            self.qa_bad += bad
            self._notify_update()
    
    def record_error(
        self, 
        error_type: str, 
        message: str,
        chunk_info: Optional[str] = None
    ) -> None:
        """Ghi nhận một lỗi"""
        with self._lock:
            self._error_history.append(ErrorRecord(
                timestamp=datetime.now().isoformat(),
                error_type=error_type,
                message=message[:200],  # Truncate
                chunk_info=chunk_info
            ))
            self._notify_update()
    
    def get_processing_rate(self) -> float:
        """Tính tốc độ xử lý (chunks/minute)"""
        with self._lock:
            if len(self._chunk_history) < 2:
                return 0.0
            
            # Lấy chunks trong 60 giây gần nhất
            now = time.time()
            recent = [c for c in self._chunk_history if now - c.timestamp < 60]
            
            if not recent:
                return 0.0
            
            # Tính rate
            time_span = now - recent[0].timestamp
            if time_span <= 0:
                return 0.0
            
            return len(recent) / (time_span / 60)
    
    def get_estimated_remaining(self) -> str:
        """Ước tính thời gian còn lại"""
        with self._lock:
            rate = self.get_processing_rate()
            remaining_chunks = self.chunks_total - self.chunks_processed
            
            if rate <= 0 or remaining_chunks <= 0:
                return "N/A"
            
            minutes = remaining_chunks / rate
            
            if minutes < 1:
                return f"~{int(minutes * 60)} giây"
            elif minutes < 60:
                return f"~{int(minutes)} phút"
            else:
                hours = minutes / 60
                return f"~{hours:.1f} giờ"
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy tất cả stats"""
        with self._lock:
            elapsed = 0.0
            if self.started_at:
                elapsed = time.time() - self.started_at
            
            return {
                "current_step": self.current_step,
                "elapsed_seconds": elapsed,
                "elapsed_formatted": self._format_duration(elapsed),
                
                "chunks": {
                    "total": self.chunks_total,
                    "processed": self.chunks_processed,
                    "success": self.chunks_success,
                    "failed": self.chunks_failed,
                    "progress_percent": (
                        self.chunks_processed / self.chunks_total * 100
                        if self.chunks_total > 0 else 0
                    )
                },
                
                "qa": {
                    "generated": self.qa_generated,
                    "good": self.qa_good,
                    "bad": self.qa_bad,
                    "good_rate": (
                        self.qa_good / (self.qa_good + self.qa_bad) * 100
                        if (self.qa_good + self.qa_bad) > 0 else 0
                    )
                },
                
                "cache": {
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "hit_rate": (
                        self.cache_hits / (self.cache_hits + self.cache_misses) * 100
                        if (self.cache_hits + self.cache_misses) > 0 else 0
                    )
                },
                
                "performance": {
                    "processing_rate": self.get_processing_rate(),
                    "estimated_remaining": self.get_estimated_remaining()
                },
                
                "errors": {
                    "count": len(self._error_history),
                    "recent": [
                        {
                            "timestamp": e.timestamp,
                            "type": e.error_type,
                            "message": e.message
                        }
                        for e in list(self._error_history)[-10:]
                    ]
                }
            }
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration thành readable string"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def add_update_callback(self, callback: Callable) -> None:
        """Thêm callback khi có update"""
        self._update_callbacks.append(callback)
    
    def remove_update_callback(self, callback: Callable) -> None:
        """Xóa callback"""
        if callback in self._update_callbacks:
            self._update_callbacks.remove(callback)
    
    def _notify_update(self) -> None:
        """Notify all callbacks"""
        stats = self.get_stats()
        for callback in self._update_callbacks:
            try:
                callback(stats)
            except Exception:
                pass  # Don't let callback errors break metrics


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def reset_metrics_collector() -> None:
    """Reset global metrics collector"""
    global _metrics_collector
    if _metrics_collector:
        _metrics_collector.reset()

