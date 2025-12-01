# -*- coding: utf-8 -*-
"""
Provider Factory - Tạo provider theo config
"""

from typing import Dict, Any, List
from .base import BaseLLMProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .custom import CustomLLMProvider


# Registry các provider chuẩn
PROVIDER_REGISTRY = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "custom": CustomLLMProvider,
}

# Preset cho các provider phổ biến
PROVIDER_PRESETS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "default_key": "ollama"
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "default_key": "lm-studio"
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "default_key": "vllm"
    },
    "megallm": {
        "base_url": "https://ai.megallm.io/v1",
        "default_key": None
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "default_key": None  # Cần API key thật
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_key": None
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "default_key": None
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_key": None
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_key": None
    },
}


def create_provider(
    provider_name: str,
    api_key: str,
    model: str,
    generation_config: Dict[str, Any] = None,
    base_url: str = None
) -> BaseLLMProvider:
    """
    Factory function để tạo LLM provider
    
    Args:
        provider_name: Tên provider (gemini, openai, anthropic, custom, ollama, together, groq, etc.)
        api_key: API key
        model: Tên model
        generation_config: Config cho generation (max_tokens, temperature, etc.)
        base_url: Custom base URL (cho custom provider)
    
    Returns:
        BaseLLMProvider instance
    """
    provider_name_lower = provider_name.lower()
    config = generation_config or {}
    
    # Check nếu là preset (ollama, together, groq, etc.)
    if provider_name_lower in PROVIDER_PRESETS:
        preset = PROVIDER_PRESETS[provider_name_lower]
        return CustomLLMProvider(
            api_key=api_key or preset.get("default_key", ""),
            model=model,
            base_url=base_url or preset["base_url"],
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.9),
            max_retries=config.get("max_retries", 3),
            retry_delay=config.get("retry_delay", 5.0)
        )
    
    # Check nếu là custom với base_url
    if provider_name_lower == "custom" or base_url:
        if not base_url:
            raise ValueError("Custom provider cần base_url")
        return CustomLLMProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.9),
            max_retries=config.get("max_retries", 3),
            retry_delay=config.get("retry_delay", 5.0)
        )
    
    # Standard providers
    if provider_name_lower not in PROVIDER_REGISTRY:
        available = list(PROVIDER_REGISTRY.keys()) + list(PROVIDER_PRESETS.keys())
        raise ValueError(f"Provider không hỗ trợ: {provider_name}. Các provider có sẵn: {available}")
    
    provider_class = PROVIDER_REGISTRY[provider_name_lower]
    return provider_class(
        api_key=api_key,
        model=model,
        max_tokens=config.get("max_tokens", 4096),
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 0.9),
        max_retries=config.get("max_retries", 3),
        retry_delay=config.get("retry_delay", 5.0)
    )


def create_providers_from_config(
    provider_name: str,
    api_keys: list,
    model: str,
    generation_config: Dict[str, Any] = None,
    base_url: str = None
) -> List[BaseLLMProvider]:
    """
    Tạo danh sách provider từ danh sách API keys
    Mỗi API key sẽ có 1 provider riêng (1 thread / 1 key)
    
    Args:
        provider_name: Tên provider
        api_keys: Danh sách API keys
        model: Tên model
        generation_config: Config cho generation
        base_url: Custom base URL (cho custom providers)
    
    Returns:
        List[BaseLLMProvider]
    """
    providers = []
    for api_key in api_keys:
        provider = create_provider(
            provider_name=provider_name,
            api_key=api_key,
            model=model,
            generation_config=generation_config,
            base_url=base_url
        )
        providers.append(provider)
    
    return providers


def get_available_providers() -> List[str]:
    """Lấy danh sách các provider có sẵn"""
    standard = list(PROVIDER_REGISTRY.keys())
    presets = list(PROVIDER_PRESETS.keys())
    return standard + presets
