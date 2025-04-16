"""
日期处理工具模块。

该模块提供了一系列用于日期处理的辅助函数，包括日期格式转换、
日期计算等功能，用于简化项目中的日期操作。

创建日期: 2024-04-12
最后修改: 2024-04-15
作者: BeeShare开发团队
"""
from datetime import timedelta
from src.utils.time_utils import get_network_time

def get_date_n_days_ago(n=90, date_format='%Y-%m-%d'):
    """
    获取n天前的日期并格式化为字符串。
    
    这个函数主要用于获取历史数据的默认开始日期，默认为90天前。
    使用网络时间作为基准，确保时间准确性。
    
    Args:
        n (int): 指定多少天前，默认为90天
        date_format (str): 返回日期的格式字符串，默认为'%Y-%m-%d'
        
    Returns:
        str: 格式化的日期字符串
        
    Raises:
        ValueError: 当参数n为负数时抛出
        
    Examples:
        >>> get_date_n_days_ago(10)
        '2023-07-05'  # 假设今天是2023-07-15
        
        >>> get_date_n_days_ago(30, '%Y%m%d')
        '20230615'  # 假设今天是2023-07-15
    """
    if n < 0:
        raise ValueError("参数n必须为非负整数")
        
    today = get_network_time()
    n_days_ago = today - timedelta(days=n)
    return n_days_ago.strftime(date_format) 