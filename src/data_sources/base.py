"""
数据源基类模块，定义所有数据源应该实现的接口。

该模块提供了数据源抽象接口的定义，所有具体的数据源实现（如AKShare、
东方财富等）都需要继承该基类并实现定义的抽象方法。基类定义了统一的
数据访问接口，使上层应用能够以统一的方式使用不同的数据源。

主要功能：
1. 历史数据获取
2. 实时行情获取
3. 股票代码搜索
4. 股票详细信息获取
5. 数据源连接检查

创建日期: 2024-04-08
最后修改: 2024-04-20
作者: BeeShare开发团队

流程图:
```mermaid
classDiagram
    class DataSource {
        +str name
        +dict config
        +bool is_ready
        +__init__(name, config)
        +check_connection() bool
        +get_historical_data(symbol, start_date, end_date, interval) DataFrame
        +get_realtime_data(symbols) DataFrame
        +search_symbols(keyword) list
        +get_symbol_info(symbol) dict
    }
    DataSource <|-- YahooFinance
    DataSource <|-- EastMoney
    DataSource <|-- AKShare
```
"""

from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataSource(ABC):
    """
    数据源的抽象基类，定义了所有数据源都应该实现的方法。
    
    该类定义了统一的数据访问接口，所有具体的数据源实现都需要继承该类
    并实现相应的抽象方法。通过统一接口，上层应用可以方便地切换不同的
    数据源而不需要修改代码。
    
    Attributes:
        name (str): 数据源名称
        config (dict): 数据源配置信息
        is_ready (bool): 数据源是否就绪可用
        timeout (int): 请求超时时间（秒）
    """
    
    def __init__(self, name, config=None):
        """
        初始化数据源。
        
        Args:
            name (str): 数据源名称
            config (dict, optional): 数据源配置字典。默认为None。
        """
        self.name = name
        self.config = config or {}
        self.is_ready = False
        self.timeout = self.config.get('timeout', 10)
        logger.info(f"初始化数据源: {name}")
    
    @abstractmethod
    def check_connection(self):
        """
        检查与数据源的连接是否正常
        
        Returns:
            bool: 连接是否成功
        """
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol, start_date=None, end_date=None, interval='1d'):
        """
        获取历史股票数据
        
        Args:
            symbol (str): 股票代码
            start_date (str, optional): 开始日期，格式YYYY-MM-DD. Defaults to None.
            end_date (str, optional): 结束日期，格式YYYY-MM-DD. Defaults to None.
            interval (str, optional): 数据间隔，如'1d'表示日线数据. Defaults to '1d'.
        
        Returns:
            pandas.DataFrame: 包含历史数据的DataFrame
        """
        pass
    
    @abstractmethod
    def get_realtime_data(self, symbols):
        """
        获取实时股票数据
        
        Args:
            symbols (list): 股票代码列表
        
        Returns:
            pandas.DataFrame: 包含实时数据的DataFrame
        """
        pass
    
    @abstractmethod
    def search_symbols(self, keyword):
        """
        搜索股票代码
        
        Args:
            keyword (str): 搜索关键词
        
        Returns:
            list: 匹配的股票代码和名称列表
        """
        pass
    
    @abstractmethod
    def get_symbol_info(self, symbol):
        """
        获取股票的详细信息
        
        Args:
            symbol (str): 股票代码
        
        Returns:
            dict: 股票详细信息
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
            # 默认为1年前
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        if end_date is None:
            # 默认为今天
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # 如果日期是字符串格式，转换为datetime对象
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
        return start_date, end_date 