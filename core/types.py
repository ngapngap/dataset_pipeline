# -*- coding: utf-8 -*-
"""
Type Definitions - Định nghĩa các types cho pipeline

Sử dụng TypedDict cho config structures và Protocol cho interfaces.
"""

from typing import (
    TypedDict, List, Dict, Any, Optional, 
    Protocol, Tuple, Union, Callable
)
from dataclasses import dataclass
from enum import Enum


# ==============================================================================
# ENUMS
# ==============================================================================

class EvaluationMode(str, Enum):
    """Chế độ đánh giá chất lượng"""
    RULE = "rule"
    LLM = "llm"
    HYBRID = "hybrid"


class PipelineStep(str, Enum):
    """Các bước trong pipeline"""
    EXTRACT = "extract"
    GENERATE = "generate"
    EVALUATE = "evaluate"
    SPLIT = "split"
    EXPORT = "export"


class ErrorSeverity(str, Enum):
    """Mức độ nghiêm trọng của lỗi"""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ==============================================================================
# CONFIG TYPED DICTS
# ==============================================================================

class GeneralConfig(TypedDict, total=False):
    """Config cho phần general"""
    project_name: str
    description: str
    language: str
    input_dir: str
    output_dir: str
    log_level: str
    log_file: str
    seed: int


class ProviderConfig(TypedDict, total=False):
    """Config cho một LLM provider"""
    model: str
    api_keys_file: str
    api_key: str
    base_url: str
    max_tokens: int
    max_output_tokens: int
    temperature: float
    top_p: float
    rate_limit_per_minute: int


class LLMConfig(TypedDict, total=False):
    """Config cho LLM section"""
    provider: str
    providers: Dict[str, ProviderConfig]


class ProcessingConfig(TypedDict, total=False):
    """Config cho processing"""
    threads_per_key: int
    max_threads: int
    chunk_size: int
    chunk_overlap: int
    max_retries: int
    retry_delay: int
    requests_per_minute: int
    checkpoint_enabled: bool
    checkpoint_interval: int


class CacheConfig(TypedDict, total=False):
    """Config cho cache"""
    enabled: bool
    cache_dir: str
    ttl_days: int


class QAGenerationConfig(TypedDict, total=False):
    """Config cho QA generation"""
    qa_per_chunk: int
    perspectives: List[Dict[str, Any]]
    prompt_template: str


class QualityCriterion(TypedDict, total=False):
    """Một tiêu chí đánh giá chất lượng"""
    name: str
    weight: float
    description: str
    patterns: List[str]
    min_chars: int


class QualityConfig(TypedDict, total=False):
    """Config cho quality evaluation"""
    enabled: bool
    min_score: int
    criteria: List[QualityCriterion]
    use_llm_evaluation: bool
    max_regenerate_iterations: int


class OutputConfig(TypedDict, total=False):
    """Config cho output"""
    formats: List[str]
    structure: Dict[str, Any]
    train_ratio: float
    val_ratio: float
    test_ratio: float


class ExportConfig(TypedDict, total=False):
    """Config cho export"""
    format: str
    system_prompt: str


class PipelineConfigDict(TypedDict, total=False):
    """Full pipeline config structure"""
    general: GeneralConfig
    llm: LLMConfig
    processing: ProcessingConfig
    qa_generation: QAGenerationConfig
    quality: QualityConfig
    output: OutputConfig
    export: ExportConfig
    documents: Dict[str, Any]


# ==============================================================================
# DATA TYPED DICTS
# ==============================================================================

class DocumentChunk(TypedDict):
    """Một chunk từ document"""
    doc_name: str
    chunk_id: int
    content: str


class QAPair(TypedDict, total=False):
    """Một cặp Q&A"""
    question: str
    answer: str
    source_doc: str
    chunk_id: int
    perspective: str
    eval_score: float
    eval_reason: str
    eval_method: str
    from_cache: bool


class EvaluationResult(TypedDict):
    """Kết quả đánh giá một Q&A"""
    score: float
    reason: str
    has_legal_citation: bool
    keep: bool


class SplitInfo(TypedDict):
    """Thông tin về một split"""
    samples: int
    documents: int
    doc_ids: List[str]


class DatasetSplits(TypedDict):
    """Các splits của dataset"""
    train: List[QAPair]
    validation: List[QAPair]
    test: List[QAPair]


class ExtractedDocument(TypedDict, total=False):
    """Document đã extract"""
    file_name: str
    file_path: str
    content: str
    num_chunks: int
    encoding: str
    size_bytes: int


# ==============================================================================
# VALIDATION TYPES
# ==============================================================================

@dataclass
class ValidationError:
    """Một lỗi validation"""
    field: str
    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    value: Any = None


@dataclass
class ValidationResult:
    """Kết quả validation"""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    
    def add_error(self, field: str, message: str, value: Any = None) -> None:
        """Thêm một lỗi"""
        self.errors.append(ValidationError(
            field=field,
            message=message,
            severity=ErrorSeverity.ERROR,
            value=value
        ))
        self.is_valid = False
    
    def add_warning(self, field: str, message: str, value: Any = None) -> None:
        """Thêm một cảnh báo"""
        self.warnings.append(ValidationError(
            field=field,
            message=message,
            severity=ErrorSeverity.WARNING,
            value=value
        ))


# ==============================================================================
# PROTOCOLS (Interfaces)
# ==============================================================================

class LLMProviderProtocol(Protocol):
    """Protocol cho LLM Provider"""
    
    api_key: str
    model: str
    max_tokens: int
    temperature: float
    
    def generate(self, prompt: str) -> Optional[str]:
        """Generate text từ prompt"""
        ...
    
    @property
    def provider_name(self) -> str:
        """Tên provider"""
        ...


class MetricsCollectorProtocol(Protocol):
    """Protocol cho Metrics Collector"""
    
    def record_chunk_processed(self, success: bool, duration: float) -> None:
        """Ghi nhận một chunk đã xử lý"""
        ...
    
    def record_qa_generated(self, count: int) -> None:
        """Ghi nhận số Q&A đã generate"""
        ...
    
    def record_error(self, error_type: str, message: str) -> None:
        """Ghi nhận một lỗi"""
        ...
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê"""
        ...


# ==============================================================================
# CALLBACK TYPES
# ==============================================================================

# Callback khi hoàn thành một chunk
ChunkCallback = Callable[[DocumentChunk, Optional[List[QAPair]]], None]

# Callback khi có lỗi
ErrorCallback = Callable[[Exception, Optional[DocumentChunk]], None]

# Callback để cập nhật progress
ProgressCallback = Callable[[int, int, str], None]  # current, total, message


# ==============================================================================
# PIPELINE STATE
# ==============================================================================

class PipelineState(TypedDict, total=False):
    """State của pipeline"""
    started_at: Optional[str]
    steps_completed: List[str]
    current_step: Optional[str]
    documents: List[ExtractedDocument]
    qa_pairs: List[QAPair]
    good_qa: List[QAPair]
    bad_qa: List[QAPair]
    rescued_qa: List[QAPair]
    splits: DatasetSplits
    tokenized_splits: Dict[str, List[Dict[str, Any]]]


# ==============================================================================
# METRICS TYPES
# ==============================================================================

class ErrorRecord(TypedDict):
    """Record một lỗi"""
    timestamp: str
    error_type: str
    message: str
    chunk_info: Optional[str]


class MetricsSnapshot(TypedDict):
    """Snapshot của metrics tại một thời điểm"""
    timestamp: str
    current_step: str
    chunks_processed: int
    chunks_total: int
    qa_generated: int
    qa_good: int
    qa_bad: int
    cache_hits: int
    cache_misses: int
    errors: List[ErrorRecord]
    processing_rate: float  # chunks per minute
    estimated_remaining: str  # time remaining


# Type aliases for convenience
QAList = List[QAPair]
DocumentList = List[ExtractedDocument]
ChunkList = List[DocumentChunk]

