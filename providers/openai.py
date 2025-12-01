# -*- coding: utf-8 -*-
"""
OpenAI Provider
"""

from typing import Optional
from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API Provider"""
    
    def _initialize(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Cần cài đặt: pip install openai")
    
    def _call_api(self, prompt: str) -> Optional[str]:
        """Call OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            return None
            
        except Exception as e:
            raise e
    
    @property
    def provider_name(self) -> str:
        return "OpenAI"
