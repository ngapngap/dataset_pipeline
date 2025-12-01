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
                base_url=self.base_url
            )
        except ImportError:
            raise ImportError("Cần cài đặt: pip install openai")
    
    def _call_api(self, prompt: str) -> Optional[str]:
        """Call custom API endpoint"""
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
            
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            return None
            
        except Exception as e:
            raise e
    
    @property
    def provider_name(self) -> str:
        return f"Custom ({self.base_url})"
