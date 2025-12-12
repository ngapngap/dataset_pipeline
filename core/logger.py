# -*- coding: utf-8 -*-
"""
Logger Module - Logging với màu sắc và file output
"""

from __future__ import annotations

import os
import sys
import logging
from datetime import datetime
from typing import Optional, Dict, ClassVar


class ColorFormatter(logging.Formatter):
    """Formatter với màu sắc cho console"""
    
    COLORS: ClassVar[Dict[str, str]] = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Windows không hỗ trợ ANSI colors tốt
        if sys.platform == 'win32':
            return super().format(record)
        
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class Logger:
    """Logger singleton cho pipeline
    
    Attributes:
        _instance: Singleton instance
        _logger: Underlying logging.Logger instance
    """
    
    _instance: ClassVar[Optional[Logger]] = None
    _logger: ClassVar[Optional[logging.Logger]] = None
    
    def __new__(cls, *args: str, **kwargs: str) -> Logger:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self, 
        name: str = "pipeline", 
        level: str = "INFO", 
        log_file: Optional[str] = None
    ) -> None:
        if Logger._logger is not None:
            return
        
        Logger._logger = logging.getLogger(name)
        Logger._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        Logger._logger.handlers = []  # Clear existing handlers
        
        # Console handler với màu
        console = logging.StreamHandler()
        console.setFormatter(ColorFormatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        ))
        Logger._logger.addHandler(console)
        
        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            Logger._logger.addHandler(file_handler)
    
    @classmethod
    def get_logger(cls) -> logging.Logger:
        if cls._logger is None:
            cls()
        return cls._logger
    
    @staticmethod
    def debug(msg: str):
        Logger.get_logger().debug(msg)
    
    @staticmethod
    def info(msg: str):
        Logger.get_logger().info(msg)
    
    @staticmethod
    def warning(msg: str):
        Logger.get_logger().warning(msg)
    
    @staticmethod
    def error(msg: str):
        Logger.get_logger().error(msg)
    
    @staticmethod
    def critical(msg: str):
        Logger.get_logger().critical(msg)
    
    @staticmethod
    def step(step_num: int, total: int, msg: str):
        """Log một bước trong pipeline"""
        border = "=" * 60
        Logger.get_logger().info(f"\n{border}")
        Logger.get_logger().info(f"BƯỚC {step_num}/{total}: {msg}")
        Logger.get_logger().info(border)
    
    @staticmethod
    def success(msg: str):
        """Log success message"""
        Logger.get_logger().info(f"✅ {msg}")
    
    @staticmethod
    def progress(current: int, total: int, extra: str = ""):
        """Log progress"""
        pct = 100 * current / total if total > 0 else 0
        msg = f"[{current}/{total}] {pct:.1f}%"
        if extra:
            msg += f" - {extra}"
        Logger.get_logger().info(msg)


# ============ Helper functions ============

def get_logger(name: str = None) -> logging.Logger:
    """
    Lấy logger instance.
    Tương thích với cách gọi: logger = get_logger(__name__)
    """
    if Logger._logger is None:
        Logger()
    return Logger._logger


def setup_logger(log_file: Optional[str] = None, level: str = "INFO") -> None:
    """
    Setup logger với file output.
    
    Args:
        log_file: Đường dẫn file log
        level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    # Reset singleton
    Logger._instance = None
    Logger._logger = None
    
    # Khởi tạo mới
    Logger(name="pipeline", level=level, log_file=log_file)
