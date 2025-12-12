# -*- coding: utf-8 -*-
"""
Base LLM Provider - Abstract class cho các LLM providers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
import time
import random
import logging

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class cho LLM providers
    
    Attributes:
        api_key: API key cho provider
        model: Tên model
        max_tokens: Số tokens tối đa cho response
        temperature: Nhiệt độ sampling
        top_p: Top-p sampling
        max_retries: Số lần retry tối đa
        retry_delay: Delay giữa các retry (seconds)
        consecutive_failures: Số lần fail liên tiếp
    """
    
    api_key: str
    model: str
    max_tokens: int
    temperature: float
    top_p: float
    max_retries: int
    retry_delay: float
    consecutive_failures: int
    
    def __init__(
        self, 
        api_key: str, 
        model: str, 
        max_tokens: int = 3000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_retries: int = 5,
        retry_delay: float = 2.0,
        **kwargs: Any
    ) -> None:
        """Initialize provider
        
        Args:
            api_key: API key
            model: Tên model
            max_tokens: Số tokens tối đa
            temperature: Nhiệt độ sampling (0.0 - 1.0)
            top_p: Top-p sampling (0.0 - 1.0)
            max_retries: Số lần retry tối đa
            retry_delay: Delay giữa các retry (seconds)
            **kwargs: Additional arguments
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.consecutive_failures = 0
        
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize provider-specific client"""
        pass
    
    @abstractmethod
    def _call_api(self, prompt: str) -> Optional[str]:
        """Call API và trả về response text
        
        Args:
            prompt: Input prompt
            
        Returns:
            Response text hoặc None nếu fail
        """
        pass
    
    def _categorize_error(self, error: Exception) -> Tuple[str, bool]:
        """Phân loại lỗi để xử lý phù hợp
        
        Args:
            error: Exception cần phân loại
            
        Returns:
            Tuple of (error_type, should_retry)
        """
        error_str = str(error).lower()
        
        # Rate limit errors
        if 'rate' in error_str or '429' in error_str or 'quota' in error_str:
            return 'rate_limit', True
        
        # Auth errors - không retry
        if '401' in error_str or '403' in error_str or 'auth' in error_str or 'key' in error_str:
            return 'auth_error', False
        
        # Timeout errors
        if 'timeout' in error_str or 'timed out' in error_str:
            return 'timeout', True
        
        # Server errors - có thể retry
        if '500' in error_str or '502' in error_str or '503' in error_str:
            return 'server_error', True
        
        # Connection errors
        if 'connection' in error_str or 'network' in error_str:
            return 'connection_error', True
        
        # Default: unknown error, try to retry
        return 'unknown', True
    
    def generate(self, prompt: str) -> Optional[str]:
        """Generate text với retry logic và error categorization
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text hoặc None nếu fail
        """
        last_error: Optional[Exception] = None
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                result = self._call_api(prompt)
                duration = time.time() - start_time
                
                if result:
                    # Reset consecutive failures on success
                    self.consecutive_failures = 0
                    logger.debug(
                        f"{self.provider_name}: Success in {duration:.2f}s "
                        f"(attempt {attempt + 1})"
                    )
                    return result
                
            except Exception as e:
                last_error = e
                self.consecutive_failures += 1
                error_type, should_retry = self._categorize_error(e)
                
                # Log với chi tiết
                logger.warning(
                    f"{self.provider_name}: {error_type} - {str(e)[:100]} "
                    f"(attempt {attempt + 1}/{self.max_retries}, "
                    f"consecutive failures: {self.consecutive_failures})"
                )
                
                # Không retry cho auth errors
                if not should_retry:
                    logger.error(
                        f"{self.provider_name}: Non-retryable error: {error_type}"
                    )
                    break
                
                # Tính wait time
                if error_type == 'rate_limit':
                    # Exponential backoff cho rate limit
                    wait_time = self.retry_delay * (2 ** attempt) + random.uniform(1, 3)
                else:
                    # Linear backoff cho các lỗi khác
                    wait_time = self.retry_delay * (attempt + 1) + random.uniform(0, 1)
                
                if attempt < self.max_retries - 1:
                    logger.info(f"{self.provider_name}: Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                    continue
        
        # Log final failure
        if last_error:
            logger.error(
                f"{self.provider_name}: Failed after {self.max_retries} attempts. "
                f"Last error: {str(last_error)[:200]}"
            )
        
        return None
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Tên provider"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.provider_name}(model={self.model})"
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê của provider
        
        Returns:
            Dict chứa thống kê
        """
        return {
            "provider": self.provider_name,
            "model": self.model,
            "consecutive_failures": self.consecutive_failures,
            "max_retries": self.max_retries
        }
