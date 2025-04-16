"""
存储模块，提供股票数据存储功能。

该模块提供了各种存储方案的实现，用于存储股票历史数据、实时行情、
股票基本信息等。目前主要支持SQLite存储，未来可扩展其他存储方式。

创建日期: 2024-04-01
最后修改: 2024-04-10
作者: BeeShare开发团队
"""

from .base import Storage, DataType
from .sqlite_storage import SQLiteStorage

# 导出接口
__all__ = ['Storage', 'DataType', 'SQLiteStorage']

# 系统现在只使用SQLite存储 