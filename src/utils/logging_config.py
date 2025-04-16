"""
日志配置模块。

该模块提供了日志系统的配置功能，支持同时输出到控制台和文件，
以及不同级别的日志记录。日志系统是项目调试和问题排查的重要工具。

创建日期: 2024-04-11
最后修改: 2024-04-14
作者: BeeShare开发团队
"""
import os
import logging

def setup_logging(log_level='INFO', log_file='logs/stock_data.log'):
    """
    设置全局日志记录器的配置。
    
    此函数配置全局日志系统，包括日志级别、格式化方式和输出目标。
    日志会同时输出到控制台和指定的日志文件中。
    
    Args:
        log_level (str): 日志记录级别，可选值包括'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'，默认为'INFO'
        log_file (str): 日志文件路径，默认为'logs/stock_data.log'
        
    Returns:
        logging.Logger: 配置好的日志记录器实例
        
    Raises:
        OSError: 当日志目录创建失败时可能抛出
        
    Examples:
        >>> logger = setup_logging('DEBUG', 'logs/debug.log')
        >>> logger.debug('这是一条调试日志')
        >>> logger.info('这是一条信息日志')
        >>> logger.error('这是一条错误日志')
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