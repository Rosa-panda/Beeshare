"""
数据源模块，提供股票数据获取功能。

该模块提供了各种数据源的实现，用于获取股票历史数据、实时行情、
股票基本信息等。目前主要支持AKShare数据源，未来可扩展其他数据源。

创建日期: 2024-04-01
最后修改: 2024-04-12
作者: BeeShare开发团队
"""

from .base import DataSource
from .akshare import AKShareDS

# 导出接口
__all__ = ['DataSource', 'AKShareDS']

# 后续会从各个具体数据源模块导入相应的类 