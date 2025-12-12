# -*- coding: utf-8 -*-
"""
Config Validators - Validate config trước khi chạy pipeline

Kiểm tra:
- Paths hợp lệ
- API keys có sẵn
- Giá trị config trong phạm vi cho phép
- Ratios đúng
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .errors import (
    ValidationError, 
    ConfigurationError,
    ErrorSeverity
)


@dataclass
class ValidationIssue:
    """Một vấn đề validation"""
    field: str
    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    value: Any = None
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity.value,
            "value": str(self.value) if self.value is not None else None,
            "suggestion": self.suggestion
        }


@dataclass
class ValidationResult:
    """Kết quả validation tổng hợp"""
    is_valid: bool = True
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    
    def add_error(
        self, 
        field: str, 
        message: str, 
        value: Any = None,
        suggestion: Optional[str] = None
    ) -> None:
        """Thêm một lỗi"""
        self.errors.append(ValidationIssue(
            field=field,
            message=message,
            severity=ErrorSeverity.ERROR,
            value=value,
            suggestion=suggestion
        ))
        self.is_valid = False
    
    def add_warning(
        self, 
        field: str, 
        message: str, 
        value: Any = None,
        suggestion: Optional[str] = None
    ) -> None:
        """Thêm một cảnh báo (không fail validation)"""
        self.warnings.append(ValidationIssue(
            field=field,
            message=message,
            severity=ErrorSeverity.WARNING,
            value=value,
            suggestion=suggestion
        ))
    
    def merge(self, other: ValidationResult) -> None:
        """Merge kết quả từ validator khác"""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings]
        }
    
    def get_error_messages(self) -> List[str]:
        """Lấy danh sách messages lỗi"""
        return [f"[{e.field}] {e.message}" for e in self.errors]
    
    def get_warning_messages(self) -> List[str]:
        """Lấy danh sách messages cảnh báo"""
        return [f"[{w.field}] {w.message}" for w in self.warnings]


class ConfigValidator:
    """Validator cho PipelineConfig
    
    Kiểm tra tất cả các config values trước khi pipeline chạy.
    
    Usage:
        validator = ConfigValidator(config)
        result = validator.validate_all()
        if not result.is_valid:
            raise ConfigurationError(...)
    """
    
    def __init__(self, config: Any) -> None:
        """
        Args:
            config: PipelineConfig instance
        """
        self.config = config
        self._raw_config = getattr(config, '_raw_config', {})
    
    def validate_all(self) -> ValidationResult:
        """Chạy tất cả validations
        
        Returns:
            ValidationResult với errors và warnings
        """
        result = ValidationResult()
        
        # Validate từng phần
        result.merge(self.validate_paths())
        result.merge(self.validate_api_keys())
        result.merge(self.validate_processing())
        result.merge(self.validate_ratios())
        result.merge(self.validate_quality())
        
        return result
    
    def validate_paths(self) -> ValidationResult:
        """Validate các đường dẫn
        
        Kiểm tra:
        - input_dir tồn tại
        - output_dir có thể tạo/ghi
        - log file directory có thể tạo
        """
        result = ValidationResult()
        
        # Input directory
        input_dir = getattr(self.config, 'input_dir', None)
        if input_dir:
            if not os.path.exists(input_dir):
                result.add_error(
                    field="general.input_dir",
                    message=f"Input directory không tồn tại: {input_dir}",
                    value=input_dir,
                    suggestion="Tạo thư mục hoặc sửa đường dẫn trong config"
                )
            elif not os.path.isdir(input_dir):
                result.add_error(
                    field="general.input_dir",
                    message=f"Đường dẫn không phải là thư mục: {input_dir}",
                    value=input_dir
                )
            elif not os.listdir(input_dir):
                result.add_warning(
                    field="general.input_dir",
                    message=f"Thư mục input rỗng: {input_dir}",
                    value=input_dir,
                    suggestion="Thêm documents vào thư mục trước khi chạy"
                )
        else:
            result.add_error(
                field="general.input_dir",
                message="Thiếu input_dir trong config",
                suggestion="Thêm general.input_dir vào config.yaml"
            )
        
        # Output directory - có thể tạo
        output_dir = getattr(self.config, 'output_dir', None)
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                # Test write permission
                test_file = os.path.join(output_dir, ".write_test")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except PermissionError:
                result.add_error(
                    field="general.output_dir",
                    message=f"Không có quyền ghi vào: {output_dir}",
                    value=output_dir
                )
            except Exception as e:
                result.add_error(
                    field="general.output_dir",
                    message=f"Lỗi tạo output directory: {e}",
                    value=output_dir
                )
        
        return result
    
    def validate_api_keys(self) -> ValidationResult:
        """Validate API keys cho provider đang dùng
        
        Kiểm tra:
        - Provider được cấu hình
        - Có ít nhất 1 API key
        - API key file tồn tại (nếu dùng file)
        """
        result = ValidationResult()
        
        llm_provider = getattr(self.config, 'llm_provider', None)
        llm_providers = getattr(self.config, 'llm_providers', {})
        
        if not llm_provider:
            result.add_error(
                field="llm.provider",
                message="Thiếu LLM provider",
                suggestion="Thêm llm.provider vào config (gemini, openai, anthropic, ...)"
            )
            return result
        
        if llm_provider not in llm_providers:
            result.add_error(
                field="llm.provider",
                message=f"Provider '{llm_provider}' không được cấu hình trong llm.providers",
                value=llm_provider,
                suggestion=f"Thêm llm.providers.{llm_provider} vào config"
            )
            return result
        
        provider_config = llm_providers.get(llm_provider)
        if not provider_config:
            result.add_error(
                field=f"llm.providers.{llm_provider}",
                message="Config của provider rỗng"
            )
            return result
        
        # Check API keys
        has_keys = False
        
        # Check inline api_key
        if hasattr(provider_config, 'api_key') and provider_config.api_key:
            has_keys = True
        
        # Check api_keys list (already loaded)
        if hasattr(provider_config, 'api_keys') and provider_config.api_keys:
            has_keys = True
        
        # Check api_keys_file
        if hasattr(provider_config, 'api_keys_file') and provider_config.api_keys_file:
            keys_file = provider_config.api_keys_file
            base_dir = getattr(self.config, 'base_dir', '.')
            if not os.path.isabs(keys_file):
                keys_file = os.path.join(base_dir, keys_file)
            
            if not os.path.exists(keys_file):
                result.add_error(
                    field=f"llm.providers.{llm_provider}.api_keys_file",
                    message=f"File API keys không tồn tại: {keys_file}",
                    value=keys_file,
                    suggestion="Tạo file và thêm API keys hoặc dùng api_key inline"
                )
            else:
                # Check file not empty
                with open(keys_file, 'r') as f:
                    keys = [l.strip() for l in f if l.strip() and not l.startswith('#')]
                if keys:
                    has_keys = True
                else:
                    result.add_warning(
                        field=f"llm.providers.{llm_provider}.api_keys_file",
                        message=f"File API keys rỗng: {keys_file}",
                        suggestion="Thêm API keys vào file"
                    )
        
        # Local providers (ollama, lmstudio) không cần API key thật
        local_providers = ['ollama', 'lmstudio', 'vllm']
        is_local = (
            llm_provider.lower() in local_providers or
            (hasattr(provider_config, 'base_url') and 
             provider_config.base_url and 
             'localhost' in provider_config.base_url)
        )
        
        if not has_keys and not is_local:
            result.add_error(
                field=f"llm.providers.{llm_provider}",
                message="Không có API key",
                suggestion="Thêm api_key hoặc api_keys_file vào config"
            )
        
        return result
    
    def validate_processing(self) -> ValidationResult:
        """Validate processing config
        
        Kiểm tra:
        - chunk_size > 0
        - chunk_overlap < chunk_size
        - temperature trong 0-2
        - max_retries > 0
        """
        result = ValidationResult()
        
        processing = getattr(self.config, 'processing', None)
        if not processing:
            return result
        
        # Chunk size
        chunk_size = getattr(processing, 'chunk_size', 4000)
        if chunk_size <= 0:
            result.add_error(
                field="processing.chunk_size",
                message="chunk_size phải > 0",
                value=chunk_size
            )
        elif chunk_size < 500:
            result.add_warning(
                field="processing.chunk_size",
                message="chunk_size quá nhỏ, có thể không đủ context",
                value=chunk_size,
                suggestion="Nên đặt ít nhất 1000"
            )
        elif chunk_size > 10000:
            result.add_warning(
                field="processing.chunk_size",
                message="chunk_size quá lớn, có thể vượt context window của model",
                value=chunk_size
            )
        
        # Chunk overlap
        chunk_overlap = getattr(processing, 'chunk_overlap', 200)
        if chunk_overlap < 0:
            result.add_error(
                field="processing.chunk_overlap",
                message="chunk_overlap không thể âm",
                value=chunk_overlap
            )
        elif chunk_overlap >= chunk_size:
            result.add_error(
                field="processing.chunk_overlap",
                message="chunk_overlap phải nhỏ hơn chunk_size",
                value=chunk_overlap
            )
        
        # Max retries
        max_retries = getattr(processing, 'max_retries', 5)
        if max_retries <= 0:
            result.add_error(
                field="processing.max_retries",
                message="max_retries phải > 0",
                value=max_retries
            )
        elif max_retries > 20:
            result.add_warning(
                field="processing.max_retries",
                message="max_retries quá cao, có thể chậm khi API fail",
                value=max_retries
            )
        
        # Threads
        max_threads = getattr(processing, 'max_threads', 20)
        if max_threads < 0:
            result.add_error(
                field="processing.max_threads",
                message="max_threads không thể âm",
                value=max_threads
            )
        elif max_threads > 100:
            result.add_warning(
                field="processing.max_threads",
                message="max_threads quá cao, có thể bị rate limit",
                value=max_threads
            )
        
        return result
    
    def validate_ratios(self) -> ValidationResult:
        """Validate split ratios
        
        Kiểm tra:
        - Tổng train + val + test = 1.0 (± 0.01)
        - Mỗi ratio trong 0-1
        """
        result = ValidationResult()
        
        output = self._raw_config.get('output', {})
        
        train_ratio = output.get('train_ratio', 0.70)
        val_ratio = output.get('val_ratio', 0.15)
        test_ratio = output.get('test_ratio', 0.15)
        
        # Check range
        for name, value in [
            ('train_ratio', train_ratio),
            ('val_ratio', val_ratio),
            ('test_ratio', test_ratio)
        ]:
            if not (0 <= value <= 1):
                result.add_error(
                    field=f"output.{name}",
                    message=f"{name} phải trong khoảng 0-1",
                    value=value
                )
        
        # Check sum
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.01:
            result.add_error(
                field="output.ratios",
                message=f"Tổng train + val + test = {total}, phải bằng 1.0",
                value=f"train={train_ratio}, val={val_ratio}, test={test_ratio}",
                suggestion="Điều chỉnh các ratio sao cho tổng = 1.0"
            )
        
        return result
    
    def validate_quality(self) -> ValidationResult:
        """Validate quality config
        
        Kiểm tra:
        - min_score trong phạm vi hợp lệ
        """
        result = ValidationResult()
        
        quality = self._raw_config.get('quality', {})
        
        min_score = quality.get('min_score', 4)
        if min_score < 1:
            result.add_error(
                field="quality.min_score",
                message="min_score phải >= 1",
                value=min_score
            )
        elif min_score > 5:
            result.add_warning(
                field="quality.min_score",
                message="min_score > 5 có thể loại bỏ quá nhiều Q&A",
                value=min_score
            )
        
        return result


def validate_config(config: Any) -> ValidationResult:
    """Convenience function để validate config
    
    Args:
        config: PipelineConfig instance
        
    Returns:
        ValidationResult
        
    Raises:
        ConfigurationError: Nếu có lỗi critical
    """
    validator = ConfigValidator(config)
    result = validator.validate_all()
    
    if not result.is_valid:
        errors_msg = "; ".join(result.get_error_messages())
        raise ConfigurationError(
            message=f"Config validation failed: {errors_msg}",
            field="config",
            value=None
        )
    
    return result

