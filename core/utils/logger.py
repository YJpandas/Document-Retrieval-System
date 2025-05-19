"""
日志工具模块，提供统一的日志记录功能
"""
import os
import logging
import colorlog
from datetime import datetime
from typing import Optional

from core.utils.config_loader import config_loader


class Logger:
    """日志记录器，提供统一的日志记录接口"""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str = "app") -> logging.Logger:
        """
        获取或创建一个日志记录器
        
        Args:
            name: 日志记录器名称
            
        Returns:
            logging.Logger: 日志记录器实例
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        # 从配置中获取日志级别和路径
        config = config_loader.get_config("app_config") or {}
        log_config = config.get("logging", {})
        log_level = log_config.get("level", "INFO")
        log_file = log_config.get("file", "logs/app.log")
        log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # 创建日志记录器
        logger = logging.getLogger(name)
        
        # 设置日志级别
        level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(level)
        
        # 避免重复添加处理器
        if logger.handlers:
            return logger
        
        # 创建控制台处理器（带颜色）
        console_handler = colorlog.StreamHandler()
        color_formatter = colorlog.ColoredFormatter(
            "%(log_color)s" + log_format,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)
        
        # 创建文件处理器
        if log_file:
            # 确保日志目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_formatter = logging.Formatter(log_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        cls._loggers[name] = logger
        return logger


# 提供通用日志记录工具函数
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取指定名称的日志记录器
    
    Args:
        name: 日志记录器名称，默认为调用模块名
        
    Returns:
        logging.Logger: 日志记录器实例
    """
    if name is None:
        # 尝试从调用栈获取模块名
        import inspect
        frame = inspect.currentframe()
        if frame:
            frame = frame.f_back
            if frame:
                module = inspect.getmodule(frame)
                if module:
                    name = module.__name__
    
    name = name or "app"
    return Logger.get_logger(name)