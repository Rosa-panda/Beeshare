"""
存储基类模块，定义所有存储方案应该实现的接口。

该模块提供了数据存储抽象接口的定义，所有具体的存储实现（如SQLite、CSV等）
都需要继承该基类并实现定义的抽象方法。基类定义了统一的数据存储和检索接口，
使上层应用能够以统一的方式使用不同的存储方案。

主要功能：
1. 数据保存
2. 数据加载
3. 数据删除
4. 数据存在性检查
5. 日期验证与转换

创建日期: 2024-04-05
最后修改: 2024-04-18
作者: BeeShare开发团队

流程图:
```mermaid
classDiagram
    class Storage {
        +str name
        +dict config
        +__init__(name, config)
        +save_data(data, data_type, symbol) bool
        +load_data(data_type, symbol, start_date, end_date) DataFrame
        +delete_data(data_type, symbol) bool
        +exists(data_type, symbol) bool
    }
    Storage <|-- CSVStorage
    Storage <|-- SQLiteStorage
```
"""

from abc import ABC, abstractmethod
import pandas as pd
import logging
from datetime import datetime
from enum import Enum, auto

logger = logging.getLogger(__name__)

class DataType(Enum):
    """
    数据类型枚举。
    
    用于区分不同类型的股票数据，在存储和检索数据时用作标识符。
    通过使用枚举而非字符串，可以确保类型安全和代码统一性。
    
    Attributes:
        HISTORICAL: 历史行情数据
        REALTIME: 实时行情数据
        SYMBOL: 股票代码和基本信息
        INDEX: 指数数据
    """
    HISTORICAL = "historical"  # 历史行情数据
    REALTIME = "realtime"      # 实时行情数据
    SYMBOL = "symbol"          # 股票代码和基本信息
    INDEX = "index"            # 指数数据

class Storage(ABC):
    """
    存储的抽象基类，定义了所有存储方式都应该实现的方法。
    
    该类定义了统一的数据存储和检索接口，所有具体的存储实现都需要继承该类
    并实现相应的抽象方法。通过统一接口，上层应用可以方便地切换不同的
    存储方案而不需要修改代码。
    
    Attributes:
        name (str): 存储方式名称
        config (dict): 存储配置信息
    """
    
    def __init__(self, name, config=None):
        """
        初始化存储方式。
        
        Args:
            name (str): 存储方式名称
            config (dict, optional): 存储配置字典。默认为None。
        """
        self.name = name
        self.config = config or {}
        logger.info(f"初始化存储方式: {name}")
    
    @abstractmethod
    def save_data(self, data, data_type, symbol=None, **kwargs):
        """
        保存数据
        
        Args:
            data (pandas.DataFrame): 要保存的数据
            data_type (str): 数据类型，如'historical', 'realtime'
            symbol (str, optional): 股票代码. Defaults to None.
        
        Returns:
            bool: 是否保存成功
        """
        pass
    
    @abstractmethod
    def load_data(self, data_type, symbol=None, start_date=None, end_date=None, **kwargs):
        """
        加载数据
        
        Args:
            data_type (str): 数据类型，如'historical', 'realtime'
            symbol (str, optional): 股票代码. Defaults to None.
            start_date (str, optional): 开始日期，格式YYYY-MM-DD. Defaults to None.
            end_date (str, optional): 结束日期，格式YYYY-MM-DD. Defaults to None.
        
        Returns:
            pandas.DataFrame: 加载的数据
        """
        pass
    
    @abstractmethod
    def delete_data(self, data_type, symbol=None, **kwargs):
        """
        删除数据
        
        Args:
            data_type (str): 数据类型，如'historical', 'realtime'
            symbol (str, optional): 股票代码. Defaults to None.
        
        Returns:
            bool: 是否删除成功
        """
        pass
    
    @abstractmethod
    def exists(self, data_type, symbol=None, **kwargs):
        """
        检查数据是否存在
        
        Args:
            data_type (str): 数据类型，如'historical', 'realtime'
            symbol (str, optional): 股票代码. Defaults to None.
        
        Returns:
            bool: 数据是否存在
        """
        pass
    
    def _validate_dates(self, start_date, end_date):
        """
        验证日期格式并转换为标准格式
        
        Args:
            start_date (str): 开始日期
            end_date (str): 结束日期
        
        Returns:
            tuple: 标准格式的开始日期和结束日期
        """
        if start_date is None:
            # 默认为最早日期
            start_date = '1900-01-01'
        
        if end_date is None:
            # 默认为今天
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # 如果日期是字符串格式，转换为datetime对象
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
        return start_date, end_date 