# -*- coding: utf-8 -*-
"""
Google Gemini Provider
"""

from typing import Optional
from .base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API Provider"""
    
    def _initialize(self):
        """Initialize Gemini client"""
        try:
            import google.generativeai as genai
            self.genai = genai
            self.genai.configure(api_key=self.api_key)
            self.client = self.genai.GenerativeModel(self.model)
        except ImportError:
            raise ImportError("Cần cài đặt: pip install google-generativeai")
    
    def _call_api(self, prompt: str) -> Optional[str]:
        """Call Gemini API"""
        try:
            response = self.client.generate_content(
                prompt,
                generation_config=self.genai.types.GenerationConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
            )
            
            if response.text:
                return response.text.strip()
            return None
            
        except Exception as e:
            raise e
    
    @property
    def provider_name(self) -> str:
        return "Gemini"
