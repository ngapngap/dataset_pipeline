# -*- coding: utf-8 -*-
"""
Custom Exceptions - Định nghĩa các exception cho pipeline

Phân loại lỗi theo mức độ nghiêm trọng và khả năng recovery.
"""

from __future__ import annotations

from typing import Optional, Any, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ErrorSeverity(str, Enum):
    """Mức độ nghiêm trọng của lỗi"""
    WARNING = "warning"    # Có thể tiếp tục
    ERROR = "error"        # Cần xử lý nhưng có thể skip
    CRITICAL = "critical"  # Phải dừng pipeline


class ErrorCategory(str, Enum):
    """Phân loại lỗi theo nguồn gốc"""
    CONFIG = "config"           # Lỗi cấu hình
    VALIDATION = "validation"   # Lỗi validation
    API = "api"                 # Lỗi từ API providers
    RATE_LIMIT = "rate_limit"   # Bị giới hạn rate
    AUTH = "auth"               # Lỗi xác thực
    NETWORK = "network"         # Lỗi mạng
    IO = "io"                   # Lỗi đọc/ghi file
    PARSE = "parse"             # Lỗi parse data
    INTERNAL = "internal"       # Lỗi nội bộ


# ==============================================================================
# BASE EXCEPTION
# ==============================================================================

class PipelineError(Exception):
    """Base exception cho tất cả lỗi pipeline
    
    Attributes:
        message: Thông điệp lỗi
        category: Phân loại lỗi
        severity: Mức độ nghiêm trọng
        details: Chi tiết bổ sung
        recoverable: Có thể recovery không
        timestamp: Thời điểm xảy ra lỗi
    """
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ) -> None:
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.recoverable = recoverable
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception thành dict để logging/serialization"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "details": self.details,
            "timestamp": self.timestamp
        }
    
    def __str__(self) -> str:
        return f"[{self.category.value.upper()}] {self.message}"


# ==============================================================================
# CONFIG & VALIDATION ERRORS
# ==============================================================================

class ConfigurationError(PipelineError):
    """Lỗi cấu hình - config không hợp lệ
    
    Ví dụ:
    - Thiếu required fields
    - Giá trị không hợp lệ
    - File config không tìm thấy
    """
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        suggestion: Optional[str] = None
    ) -> None:
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["invalid_value"] = str(value)
        if suggestion:
            details["suggestion"] = suggestion
        
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIG,
            severity=ErrorSeverity.CRITICAL,
            details=details,
            recoverable=False
        )
        self.field = field
        self.value = value
        self.suggestion = suggestion


class ValidationError(PipelineError):
    """Lỗi validation dữ liệu
    
    Ví dụ:
    - Input directory không tồn tại
    - API key không hợp lệ
    - Split ratios không đúng
    """
    
    def __init__(
        self,
        message: str,
        field: str,
        value: Any = None,
        expected: Optional[str] = None
    ) -> None:
        details = {
            "field": field,
        }
        if value is not None:
            details["actual_value"] = str(value)
        if expected:
            details["expected"] = expected
        
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            details=details,
            recoverable=False
        )
        self.field = field
        self.value = value
        self.expected = expected


# ==============================================================================
# API & PROVIDER ERRORS
# ==============================================================================

class APIError(PipelineError):
    """Lỗi từ LLM API providers
    
    Ví dụ:
    - Server trả về lỗi 500
    - Response không parse được
    - Timeout
    """
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None
    ) -> None:
        details = {}
        if provider:
            details["provider"] = provider
        if status_code:
            details["status_code"] = status_code
        if response_body:
            details["response_body"] = response_body[:500]  # Truncate
        
        super().__init__(
            message=message,
            category=ErrorCategory.API,
            severity=ErrorSeverity.ERROR,
            details=details,
            recoverable=True
        )
        self.provider = provider
        self.status_code = status_code


class RateLimitError(PipelineError):
    """Lỗi rate limit từ API
    
    Có thể retry sau khi chờ một khoảng thời gian.
    """
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        retry_after: Optional[float] = None
    ) -> None:
        details = {}
        if provider:
            details["provider"] = provider
        if retry_after:
            details["retry_after_seconds"] = retry_after
        
        super().__init__(
            message=message,
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.WARNING,
            details=details,
            recoverable=True
        )
        self.provider = provider
        self.retry_after = retry_after


class AuthenticationError(PipelineError):
    """Lỗi xác thực API key
    
    Không thể retry - cần kiểm tra lại API key.
    """
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None
    ) -> None:
        details = {}
        if provider:
            details["provider"] = provider
        
        super().__init__(
            message=message,
            category=ErrorCategory.AUTH,
            severity=ErrorSeverity.CRITICAL,
            details=details,
            recoverable=False
        )
        self.provider = provider


# ==============================================================================
# IO & NETWORK ERRORS
# ==============================================================================

class NetworkError(PipelineError):
    """Lỗi kết nối mạng
    
    Có thể retry.
    """
    
    def __init__(
        self,
        message: str,
        url: Optional[str] = None
    ) -> None:
        details = {}
        if url:
            details["url"] = url
        
        super().__init__(
            message=message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            details=details,
            recoverable=True
        )
        self.url = url


class FileIOError(PipelineError):
    """Lỗi đọc/ghi file
    
    Ví dụ:
    - File không tồn tại
    - Không có quyền ghi
    - Disk full
    """
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: str = "read"  # read/write
    ) -> None:
        details = {
            "operation": operation
        }
        if file_path:
            details["file_path"] = file_path
        
        super().__init__(
            message=message,
            category=ErrorCategory.IO,
            severity=ErrorSeverity.ERROR,
            details=details,
            recoverable=False
        )
        self.file_path = file_path
        self.operation = operation


class ParseError(PipelineError):
    """Lỗi parse data
    
    Ví dụ:
    - JSON không hợp lệ
    - YAML syntax error
    - LLM response không parse được
    """
    
    def __init__(
        self,
        message: str,
        content: Optional[str] = None,
        expected_format: str = "JSON"
    ) -> None:
        details = {
            "expected_format": expected_format
        }
        if content:
            details["content_preview"] = content[:200]  # Truncate
        
        super().__init__(
            message=message,
            category=ErrorCategory.PARSE,
            severity=ErrorSeverity.WARNING,
            details=details,
            recoverable=True
        )
        self.content = content
        self.expected_format = expected_format


# ==============================================================================
# ERROR SUMMARY
# ==============================================================================

@dataclass
class ErrorSummary:
    """Tổng hợp các lỗi trong một session
    
    Dùng để báo cáo cuối pipeline.
    """
    
    total_errors: int = 0
    errors_by_category: Dict[str, int] = field(default_factory=dict)
    errors_by_severity: Dict[str, int] = field(default_factory=dict)
    recent_errors: List[Dict[str, Any]] = field(default_factory=list)
    max_recent: int = 50
    
    def add_error(self, error: PipelineError) -> None:
        """Thêm một lỗi vào summary"""
        self.total_errors += 1
        
        # Count by category
        cat = error.category.value
        self.errors_by_category[cat] = self.errors_by_category.get(cat, 0) + 1
        
        # Count by severity
        sev = error.severity.value
        self.errors_by_severity[sev] = self.errors_by_severity.get(sev, 0) + 1
        
        # Keep recent errors
        self.recent_errors.append(error.to_dict())
        if len(self.recent_errors) > self.max_recent:
            self.recent_errors.pop(0)
    
    def has_critical_errors(self) -> bool:
        """Kiểm tra có lỗi critical không"""
        return self.errors_by_severity.get(ErrorSeverity.CRITICAL.value, 0) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert summary thành dict"""
        return {
            "total_errors": self.total_errors,
            "by_category": self.errors_by_category,
            "by_severity": self.errors_by_severity,
            "recent_errors": self.recent_errors
        }
    
    def get_report(self) -> str:
        """Tạo báo cáo text"""
        lines = [
            "=" * 50,
            "ERROR SUMMARY",
            "=" * 50,
            f"Total errors: {self.total_errors}",
            "",
            "By category:"
        ]
        
        for cat, count in sorted(self.errors_by_category.items()):
            lines.append(f"  {cat}: {count}")
        
        lines.append("")
        lines.append("By severity:")
        for sev, count in sorted(self.errors_by_severity.items()):
            lines.append(f"  {sev}: {count}")
        
        if self.recent_errors:
            lines.append("")
            lines.append("Recent errors:")
            for err in self.recent_errors[-5:]:
                lines.append(f"  [{err['category']}] {err['message'][:80]}")
        
        lines.append("=" * 50)
        return "\n".join(lines)

