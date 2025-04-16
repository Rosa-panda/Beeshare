"""
日志工具模块

提供增强的日志记录功能，包括格式化、过滤和上下文记录
"""

import logging
import os
import sys
import traceback
from datetime import datetime
from functools import wraps
import inspect
import json

# 日志级别映射
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# 颜色格式（控制台输出）
COLORS = {
    'DEBUG': '\033[36m',     # 青色
    'INFO': '\033[32m',      # 绿色
    'WARNING': '\033[33m',   # 黄色
    'ERROR': '\033[31m',     # 红色
    'CRITICAL': '\033[35m',  # 紫色
    'RESET': '\033[0m'       # 重置
}

class LogFormatter(logging.Formatter):
    """自定义日志格式化器，支持颜色和详细信息"""
    
    def format(self, record):
        """格式化日志记录"""
        # 保存原始的格式化信息
        format_orig = self._style._fmt
        
        # 添加详细的上下文信息（仅在DEBUG级别）
        if record.levelno == logging.DEBUG:
            self._style._fmt = "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
        
        # 获取调用信息（文件名和行号）
        if not hasattr(record, 'module_path') and hasattr(record, 'pathname'):
            module_path = record.pathname.split(os.sep)
            record.module_path = '.'.join(module_path[-3:]) if len(module_path) > 2 else record.pathname
        
        # 控制台添加颜色
        if hasattr(record, 'color'):
            levelname = record.levelname
            if levelname in COLORS:
                record.levelname = f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"
        
        # 使用父类方法格式化
        result = super().format(record)
        
        # 恢复原始格式
        self._style._fmt = format_orig
        
        return result

def setup_logger(name, log_file=None, level='INFO', console=True, file=True):
    """
    设置日志记录器
    
    Args:
        name (str): 日志记录器名称
        log_file (str, optional): 日志文件路径. 默认为None.
        level (str, optional): 日志级别. 默认为'INFO'.
        console (bool, optional): 是否输出到控制台. 默认为True.
        file (bool, optional): 是否输出到文件. 默认为True.
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 获取日志级别
    log_level = LOG_LEVELS.get(level.upper(), logging.INFO)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 日志格式
    console_formatter = LogFormatter('%(asctime)s [%(levelname)s] [%(name)s] - %(message)s')
    file_formatter = LogFormatter('%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] [%(process)d:%(thread)d] - %(message)s')
    
    # 添加控制台处理器
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        # 为控制台输出添加颜色标记
        console_handler.addFilter(lambda record: setattr(record, 'color', True) or True)
        logger.addHandler(console_handler)
    
    # 添加文件处理器
    if file and log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_function_call(logger=None, level='DEBUG'):
    """
    装饰器：记录函数的调用和返回
    
    Args:
        logger (logging.Logger, optional): 日志记录器. 默认为None.
        level (str, optional): 日志级别. 默认为'DEBUG'.
    
    Returns:
        function: 装饰后的函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 如果未提供记录器，使用函数所属模块的记录器
            nonlocal logger
            if logger is None:
                module_name = func.__module__
                logger = logging.getLogger(module_name)
            
            # 获取日志级别
            log_level = LOG_LEVELS.get(level.upper(), logging.DEBUG)
            
            # 记录函数调用
            func_name = func.__qualname__
            arg_names = inspect.getfullargspec(func).args
            args_str = []
            
            # 处理位置参数
            all_args = list(args)  # 转换为列表以便索引
            has_self = arg_names and arg_names[0] in ('self', 'cls')
            
            # 组装参数字符串
            for i, arg_name in enumerate(arg_names):
                if i < len(all_args):
                    # 跳过对self或cls的打印，但不移除参数
                    if i == 0 and has_self:
                        continue
                    args_str.append(f"{arg_name}={repr(all_args[i])}")
                elif arg_name in kwargs:
                    args_str.append(f"{arg_name}={repr(kwargs[arg_name])}")
            
            # 处理未命名的位置参数
            if len(all_args) > len(arg_names):
                for i in range(len(arg_names), len(all_args)):
                    args_str.append(repr(all_args[i]))
            
            # 处理没有在arg_names中的关键字参数
            for key, value in kwargs.items():
                if key not in arg_names:
                    args_str.append(f"{key}={repr(value)}")
            
            # 记录函数调用日志
            logger.log(log_level, f"调用 {func_name}({', '.join(args_str)})")
            
            try:
                # 执行函数 - 不修改原始参数
                result = func(*args, **kwargs)
                
                # 记录函数返回
                result_str = repr(result)
                if len(result_str) > 100:
                    result_str = result_str[:97] + "..."
                logger.log(log_level, f"{func_name} 返回: {result_str}")
                
                return result
            except Exception as e:
                # 记录函数异常
                logger.error(f"{func_name} 发生异常: {str(e)}")
                logger.debug(f"异常详情: {traceback.format_exc()}")
                raise
                
        return wrapper
    return decorator

def log_exception(logger=None, reraise=True, level='ERROR'):
    """
    上下文管理器：记录代码块中的异常
    
    Args:
        logger (logging.Logger, optional): 日志记录器. 默认为None.
        reraise (bool, optional): 是否重新抛出异常. 默认为True.
        level (str, optional): 日志级别. 默认为'ERROR'.
    
    Returns:
        object: 上下文管理器
    """
    class ExceptionLogger:
        def __init__(self, logger, reraise, level):
            self.logger = logger
            self.reraise = reraise
            self.level = LOG_LEVELS.get(level.upper(), logging.ERROR)
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                # 获取异常详情
                tb_str = ''.join(traceback.format_exception(exc_type, exc_val, exc_tb))
                
                # 如果未提供记录器，使用根记录器
                if self.logger is None:
                    self.logger = logging.getLogger()
                
                # 记录异常
                self.logger.log(self.level, f"捕获异常: {exc_val}")
                self.logger.debug(f"异常详情:\n{tb_str}")
                
                # 是否重新抛出异常
                return not self.reraise
            return False
    
    return ExceptionLogger(logger, reraise, level)

# 测试代码
if __name__ == "__main__":
    # 设置测试日志记录器
    test_logger = setup_logger('test_logger', 'test.log', 'DEBUG')
    
    # 测试日志记录
    test_logger.debug("这是一条调试信息")
    test_logger.info("这是一条信息")
    test_logger.warning("这是一条警告")
    test_logger.error("这是一条错误")
    test_logger.critical("这是一条严重错误")
    
    # 测试函数调用装饰器
    @log_function_call()
    def test_function(a, b, c=None):
        return a + b
    
    test_function(1, 2, c=3)
    
    # 测试异常记录上下文管理器
    try:
        with log_exception(test_logger):
            1/0
    except:
        pass 