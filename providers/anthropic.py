# -*- coding: utf-8 -*-
"""
Anthropic Provider (Claude)
"""

from typing import Optional
from .base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API Provider"""
    
    def _initialize(self):
        """Initialize Anthropic client"""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Cần cài đặt: pip install anthropic")
    
    def _call_api(self, prompt: str) -> Optional[str]:
        """Call Anthropic API"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            if response.content and len(response.content) > 0:
                return response.content[0].text.strip()
            return None
            
        except Exception as e:
            raise e
    
    @property
    def provider_name(self) -> str:
        return "Anthropic"
