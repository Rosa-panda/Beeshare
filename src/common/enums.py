"""
项目枚举值定义模块。

该模块包含整个项目中使用的各种枚举类型，用于统一管理有限集合的常量值。
使用枚举可以增强代码的类型安全性和可读性，减少错误发生的可能性。

创建日期: 2024-04-10
最后修改: 2024-04-15
作者: BeeShare开发团队
"""
from enum import Enum, auto

class DataType(Enum):
    """
    数据类型枚举。
    
    用于标识不同类型的股票数据，如历史数据、实时数据等。
    在存储、检索和处理数据时用于区分不同的数据类别。
    """
    HISTORICAL = "historical"  # 历史行情数据
    REALTIME = "realtime"      # 实时行情数据
    INDEX = "index"            # 指数数据
    
class DataSource(Enum):
    """
    数据源类型枚举。
    
    定义系统支持的各种数据源类型，每个值对应一个特定的数据提供者。
    使用枚举可以在代码中统一引用数据源，而不依赖于字符串字面量。
    """
    AKSHARE = "akshare"        # AKShare数据源
    
class StorageType(Enum):
    """
    存储类型枚举。
    
    定义系统支持的各种数据存储方式，如SQLite、CSV等。
    用于在存储配置和数据访问层统一管理不同的存储介质。
    """
    SQLITE = "sqlite"          # SQLite数据库存储 