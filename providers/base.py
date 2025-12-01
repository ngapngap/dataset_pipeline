# -*- coding: utf-8 -*-
"""
Base LLM Provider - Abstract class cho các LLM providers
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import time
import random


class BaseLLMProvider(ABC):
    """Abstract base class cho LLM providers"""
    
    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.max_tokens = kwargs.get('max_tokens', 3000)
        self.temperature = kwargs.get('temperature', 0.7)
        self.top_p = kwargs.get('top_p', 0.9)
        self.max_retries = kwargs.get('max_retries', 5)
        self.retry_delay = kwargs.get('retry_delay', 2)
        
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """Initialize provider-specific client"""
        pass
    
    @abstractmethod
    def _call_api(self, prompt: str) -> Optional[str]:
        """Call API và trả về response text"""
        pass
    
    def generate(self, prompt: str) -> Optional[str]:
        """Generate text với retry logic
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text hoặc None nếu fail
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                result = self._call_api(prompt)
                if result:
                    return result
            
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Rate limit - wait longer
                if 'rate' in error_str or '429' in error_str:
                    wait_time = self.retry_delay * (attempt + 1) * 2 + random.uniform(1, 3)
                    time.sleep(wait_time)
                    continue
                
                # Other errors - normal retry
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay + random.uniform(0, 1))
                    continue
        
        return None
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Tên provider"""
        pass
    
    def __repr__(self):
        return f"{self.provider_name}(model={self.model})"
