"""
日志配置模块
"""
import os
import logging

def setup_logging(log_level='INFO', log_file='logs/stock_data.log'):
    """
    设置日志记录
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
    """
    # 创建日志目录
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    return logging.getLogger(__name__) 