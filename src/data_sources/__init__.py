"""
数据源模块，包含各种股票数据来源的接口实现
"""

from .base import DataSource
from .akshare import AKShareDS

# 导出接口
__all__ = ['DataSource', 'AKShareDS']

# 后续会从各个具体数据源模块导入相应的类 