"""
枚举值定义
"""
from enum import Enum, auto

class DataType(Enum):
    """数据类型枚举"""
    HISTORICAL = "historical"
    REALTIME = "realtime"
    INDEX = "index"
    
class DataSource(Enum):
    """数据源类型枚举"""
    AKSHARE = "akshare"
    
class StorageType(Enum):
    """存储类型枚举"""
    SQLITE = "sqlite" 