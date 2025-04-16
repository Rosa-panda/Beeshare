"""
存储模块，包含不同的数据存储方案
"""

from .base import Storage
from .sqlite_storage import SQLiteStorage

# 导出接口
__all__ = ['Storage', 'SQLiteStorage']

# 系统现在只使用SQLite存储 