# -*- coding: utf-8 -*-
"""
Custom LLM Provider - Hỗ trợ bất kỳ OpenAI-compatible API endpoint
"""

from typing import Optional
from .base import BaseLLMProvider


class CustomLLMProvider(BaseLLMProvider):
    """
    Custom LLM Provider cho các API tương thích OpenAI.
    Hỗ trợ: Ollama, LM Studio, vLLM, Together, Groq, Fireworks, etc.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "http://localhost:11434/v1",  # Default Ollama
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        **kwargs
    ):
        """
        Args:
            api_key: API key (có thể là "ollama" cho local)
            model: Tên model
            base_url: Base URL của API endpoint
            max_tokens: Max tokens
            temperature: Temperature
            top_p: Top-p sampling
            max_retries: Số lần retry
            retry_delay: Delay giữa các retry
            **kwargs: Các tham số bổ sung
        """
        self.base_url = base_url
        self.extra_params = kwargs
        super().__init__(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    
    def _initialize(self):
        """Initialize OpenAI client với custom base_url"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=120.0  # 2 phút timeout cho API chậm như megallm
            )
        except ImportError:
            raise ImportError("Cần cài đặt: pip install openai")
    
    def _call_api(self, prompt: str) -> Optional[str]:
        """Call custom API endpoint"""
        import time
        import logging
        logger = logging.getLogger(__name__)
        
        start_time = time.time()
        try:
            # Build request params
            request_params = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p
            }
            
            # Add extra params if any
            request_params.update(self.extra_params)
            
            response = self.client.chat.completions.create(**request_params)
            elapsed = time.time() - start_time
            
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                logger.info(f"✅ API response received in {elapsed:.1f}s ({len(result)} chars)")
                return result
            
            logger.warning(f"⚠️ API returned empty response after {elapsed:.1f}s")
            return None
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ API error after {elapsed:.1f}s: {type(e).__name__}: {str(e)[:100]}")
            raise e
    
    @property
    def provider_name(self) -> str:
        return f"Custom ({self.base_url})"
