# -*- coding: utf-8 -*-
"""
Dashboard Module - Real-time monitoring cho Pipeline

Cung cấp:
- Web dashboard để theo dõi tiến độ pipeline
- WebSocket updates real-time
- Metrics collection và export
"""

from .metrics import MetricsCollector, get_metrics_collector
from .app import create_app

__all__ = [
    'MetricsCollector',
    'get_metrics_collector',
    'create_app'
]

