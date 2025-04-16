"""
数据源接口模块

定义了所有数据源需要实现的抽象接口。
任何新的数据源都应该继承此接口并实现相应的方法。
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Union, List, Dict, Any, Optional


class DataSourceInterface(ABC):
    """数据源抽象接口类
    
    所有具体数据源实现必须继承此类并实现其中的抽象方法。
    """
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """初始化数据源
        
        Args:
            config: 数据源配置参数，可选
            
        Returns:
            bool: 初始化是否成功
        """
        pass
        
    @abstractmethod
    def is_available(self) -> bool:
        """检查数据源是否可用
        
        Returns:
            bool: 数据源是否可用
        """
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, 
                           interval: str = '1d') -> pd.DataFrame:
        """获取历史数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期，格式YYYY-MM-DD
            end_date: 结束日期，格式YYYY-MM-DD
            interval: 数据间隔，默认日频'1d'
            
        Returns:
            DataFrame: 包含历史数据的DataFrame对象
            
        Raises:
            DataSourceError: 获取数据失败时抛出
        """
        pass
    
    @abstractmethod
    def get_realtime_data(self, symbols: Union[str, List[str]]) -> pd.DataFrame:
        """获取实时数据
        
        Args:
            symbols: 股票代码或股票代码列表
            
        Returns:
            DataFrame: 包含实时数据的DataFrame对象
            
        Raises:
            DataSourceError: 获取数据失败时抛出
        """
        pass
    
    @abstractmethod
    def search_symbols(self, keyword: str) -> pd.DataFrame:
        """搜索股票代码
        
        Args:
            keyword: 搜索关键词
            
        Returns:
            DataFrame: 包含搜索结果的DataFrame对象
            
        Raises:
            DataSourceError: 搜索失败时抛出
        """
        pass
    
    @abstractmethod
    def get_index_data(self, symbol: str, start_date: str, end_date: str,
                     interval: str = '1d') -> pd.DataFrame:
        """获取指数数据
        
        Args:
            symbol: 指数代码
            start_date: 开始日期，格式YYYY-MM-DD
            end_date: 结束日期，格式YYYY-MM-DD
            interval: 数据间隔，默认日频'1d'
            
        Returns:
            DataFrame: 包含指数数据的DataFrame对象
            
        Raises:
            DataSourceError: 获取数据失败时抛出
        """
        pass
    
    @abstractmethod
    def identify_stock_type(self, symbol: str) -> Dict[str, Any]:
        """识别股票类型
        
        Args:
            symbol: 股票代码
            
        Returns:
            Dict: 包含股票类型信息的字典
            
        Raises:
            DataSourceError: 识别失败时抛出
        """
        pass
    
    @abstractmethod
    def get_all_stock_list(self) -> pd.DataFrame:
        """获取所有股票列表
        
        Returns:
            DataFrame: 包含所有股票信息的DataFrame对象
            
        Raises:
            DataSourceError: 获取失败时抛出
        """
        pass


class DataSourceError(Exception):
    """数据源异常类
    
    当数据源操作发生错误时抛出此异常。
    """
    
    def __init__(self, message: str, source: str = "", error_code: int = 0, 
                original_error: Optional[Exception] = None):
        """初始化数据源异常
        
        Args:
            message: 错误消息
            source: 错误来源
            error_code: 错误代码
            original_error: 原始异常对象
        """
        self.message = message
        self.source = source
        self.error_code = error_code
        self.original_error = original_error
        super().__init__(f"[{source}] {message} (Code: {error_code})") 