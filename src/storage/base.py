"""
存储基类模块，定义所有存储方案应该实现的接口

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
    数据类型枚举
    """
    HISTORICAL = "historical"
    REALTIME = "realtime"
    SYMBOL = "symbol"
    INDEX = "index"

class Storage(ABC):
    """
    存储的抽象基类，定义了所有存储方式都应该实现的方法
    
    Args:
        name (str): 存储方式名称
        config (dict): 存储配置信息
    """
    
    def __init__(self, name, config=None):
        """
        初始化存储方式
        
        Args:
            name (str): 存储方式名称
            config (dict, optional): 存储配置. Defaults to None.
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