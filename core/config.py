# -*- coding: utf-8 -*-
"""
Configuration Manager - Load và quản lý config từ YAML
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union


@dataclass
class LLMProviderConfig:
    """Config cho một LLM provider"""
    model: str
    api_keys_file: str = ""
    api_key: str = ""                     # Single API key inline
    base_url: str = ""                    # Custom base URL
    max_tokens: int = 3000
    temperature: float = 0.7
    top_p: float = 0.9
    rate_limit_per_minute: int = 60
    
    # Runtime
    api_keys: List[str] = field(default_factory=list)
    
    def load_keys(self, base_dir: str) -> List[str]:
        """
        Load API keys từ file hoặc inline api_key.
        Ưu tiên: api_keys_file > api_key
        """
        # Nếu đã có keys thì không load lại
        if self.api_keys:
            return self.api_keys
        
        # Thử load từ file trước
        if self.api_keys_file:
            keys_path = os.path.join(base_dir, self.api_keys_file)
            if os.path.exists(keys_path):
                with open(keys_path, 'r', encoding='utf-8') as f:
                    self.api_keys = [
                        line.strip() for line in f 
                        if line.strip() and not line.startswith('#')
                    ]
        
        # Nếu không có file hoặc file rỗng, dùng inline api_key
        if not self.api_keys and self.api_key:
            self.api_keys = [self.api_key]
        
        return self.api_keys


@dataclass  
class ProcessingConfig:
    """Config cho xử lý"""
    threads_per_key: int = 1
    max_threads: int = 20
    chunk_size: int = 4000
    chunk_overlap: int = 200
    max_retries: int = 5
    retry_delay: int = 2
    requests_per_minute: int = 60
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 50


@dataclass
class QAGenerationConfig:
    """Config cho sinh Q&A"""
    qa_per_chunk: int = 5
    perspectives: List[Dict] = field(default_factory=list)
    prompt_template: str = ""
    
    def get_enabled_perspectives(self) -> List[Dict]:
        """Lấy danh sách perspectives được bật"""
        return [p for p in self.perspectives if p.get('enabled', True)]


@dataclass
class QualityConfig:
    """Config cho đánh giá chất lượng"""
    enabled: bool = True
    min_score: int = 3
    criteria: List[Dict] = field(default_factory=list)
    use_llm_evaluation: bool = False


@dataclass
class OutputConfig:
    """Config cho output"""
    formats: List[str] = field(default_factory=lambda: ["json", "jsonl"])
    structure: Dict = field(default_factory=dict)
    deduplicate_enabled: bool = True
    similarity_threshold: float = 0.9


class PipelineConfig:
    """Main configuration class"""
    
    config_path: str
    base_dir: str
    _raw_config: Dict[str, Any]
    
    # Parsed config attributes
    project_name: str
    description: str
    language: str
    input_dir: str
    output_dir: str
    log_level: str
    log_file: str
    llm_provider: str
    llm_providers: Dict[str, LLMProviderConfig]
    processing: ProcessingConfig
    qa_generation: QAGenerationConfig
    quality: QualityConfig
    output: OutputConfig
    document_mappings: Dict[str, str]
    ignore_patterns: List[str]
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = config_path or self._find_config()
        self.base_dir = os.path.dirname(os.path.abspath(self.config_path))
        
        # Load YAML
        self._raw_config = self._load_yaml()
        
        # Parse sections
        self._parse_config()
        
        # Create directories
        self._setup_directories()
    
    def _find_config(self) -> str:
        """Tìm file config.yaml
        
        Returns:
            Đường dẫn tuyệt đối tới config.yaml
            
        Raises:
            FileNotFoundError: Nếu không tìm thấy config.yaml
        """
        search_paths: List[str] = [
            "./config.yaml",
            "../config.yaml",
            os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return os.path.abspath(path)
        
        raise FileNotFoundError("Không tìm thấy config.yaml")
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML file
        
        Returns:
            Dict chứa config đã parse từ YAML
        """
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _parse_config(self) -> None:
        """Parse config thành các objects"""
        # General
        general = self._raw_config.get('general', {})
        self.project_name = general.get('project_name', 'dataset')
        self.description = general.get('description', '')
        self.language = general.get('language', 'vi')
        self.input_dir = self._resolve_path(general.get('input_dir', './input'))
        self.output_dir = self._resolve_path(general.get('output_dir', './output'))
        self.log_level = general.get('log_level', 'INFO')
        self.log_file = self._resolve_path(general.get('log_file', './logs/pipeline.log'))
        
        # LLM
        llm = self._raw_config.get('llm', {})
        self.llm_provider = llm.get('provider', 'gemini')
        self.llm_providers = {}
        
        for provider_name, provider_config in llm.get('providers', {}).items():
            self.llm_providers[provider_name] = LLMProviderConfig(
                model=provider_config.get('model', ''),
                api_keys_file=provider_config.get('api_keys_file', ''),
                api_key=provider_config.get('api_key', ''),
                base_url=provider_config.get('base_url', ''),
                max_tokens=provider_config.get('max_tokens', provider_config.get('max_output_tokens', 3000)),
                temperature=provider_config.get('temperature', 0.7),
                top_p=provider_config.get('top_p', 0.9),
                rate_limit_per_minute=provider_config.get('rate_limit_per_minute', 60)
            )
        
        # Processing
        proc = self._raw_config.get('processing', {})
        self.processing = ProcessingConfig(
            threads_per_key=proc.get('threads_per_key', 1),
            max_threads=proc.get('max_threads', 20),
            chunk_size=proc.get('chunk_size', 4000),
            chunk_overlap=proc.get('chunk_overlap', 200),
            max_retries=proc.get('max_retries', 5),
            retry_delay=proc.get('retry_delay', 2),
            requests_per_minute=proc.get('requests_per_minute', 60),
            checkpoint_enabled=proc.get('checkpoint_enabled', True),
            checkpoint_interval=proc.get('checkpoint_interval', 50)
        )
        
        # QA Generation
        qa = self._raw_config.get('qa_generation', {})
        self.qa_generation = QAGenerationConfig(
            qa_per_chunk=qa.get('qa_per_chunk', 5),
            perspectives=qa.get('perspectives', []),
            prompt_template=qa.get('prompt_template', '')
        )
        
        # Quality
        qual = self._raw_config.get('quality', {})
        self.quality = QualityConfig(
            enabled=qual.get('enabled', True),
            min_score=qual.get('min_score', 3),
            criteria=qual.get('criteria', []),
            use_llm_evaluation=qual.get('use_llm_evaluation', False)
        )
        
        # Output
        out = self._raw_config.get('output', {})
        dedup = out.get('deduplicate', {})
        self.output = OutputConfig(
            formats=out.get('formats', ['json', 'jsonl']),
            structure=out.get('structure', {}),
            deduplicate_enabled=dedup.get('enabled', True),
            similarity_threshold=dedup.get('similarity_threshold', 0.9)
        )
        
        # Documents
        docs = self._raw_config.get('documents', {})
        self.document_mappings = docs.get('mappings', {})
        self.ignore_patterns = docs.get('ignore_patterns', [])
    
    def _resolve_path(self, path: str) -> str:
        """Resolve đường dẫn tương đối
        
        Args:
            path: Đường dẫn có thể là tương đối hoặc tuyệt đối
            
        Returns:
            Đường dẫn tuyệt đối đã chuẩn hóa
        """
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.base_dir, path))
    
    def _setup_directories(self) -> None:
        """Tạo các thư mục cần thiết"""
        dirs = [
            self.output_dir,
            os.path.join(self.output_dir, "extracted"),
            os.path.join(self.output_dir, "raw_qa"),
            os.path.join(self.output_dir, "evaluated"),
            os.path.join(self.output_dir, "final"),
            os.path.dirname(self.log_file)
        ]
        
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def get_current_provider(self) -> LLMProviderConfig:
        """Lấy config của provider hiện tại"""
        if self.llm_provider not in self.llm_providers:
            raise ValueError(f"Provider '{self.llm_provider}' không được cấu hình")
        
        provider = self.llm_providers[self.llm_provider]
        
        # Load API keys nếu chưa có
        if not provider.api_keys:
            provider.load_keys(self.base_dir)
        
        return provider
    
    def get_document_name(self, filename: str) -> str:
        """Lấy tên đầy đủ của văn bản"""
        for pattern, full_name in self.document_mappings.items():
            if pattern in filename:
                return full_name
        return filename
    
    def to_dict(self) -> Dict:
        """Export config thành dict"""
        return self._raw_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Lấy giá trị config theo key path (dot notation).
        
        Args:
            key: Key path, vd: "llm.provider" hoặc "processing.chunk_size"
            default: Giá trị mặc định nếu key không tồn tại
        
        Returns:
            Giá trị config hoặc default
        
        Example:
            config.get("llm.providers.gemini.model")
            config.get("processing.chunk_size", 4000)
        """
        keys = key.split(".")
        value = self._raw_config
        
        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value[k]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default
    
    def __repr__(self):
        return f"PipelineConfig(project={self.project_name}, provider={self.llm_provider})"
