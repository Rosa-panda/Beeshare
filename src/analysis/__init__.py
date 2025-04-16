"""
股票数据分析模块。

该模块提供各种分析工具，用于分析股票数据，包括技术分析、基本面分析、
统计分析和聚类分析，支持数据预处理、分析计算和结果可视化。

创建日期: 2024-04-10
最后修改: 2024-04-20
作者: BeeShare开发团队
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Analyzer:
    """
    分析器基类。
    
    定义所有分析器的共同接口和基本功能，作为各种具体分析器的父类。
    负责处理数据输入、预处理和基本验证等通用功能。
    
    Attributes:
        data (pd.DataFrame): 待分析的数据
    """
    
    def __init__(self, data=None):
        """
        初始化分析器。
        
        Args:
            data (pandas.DataFrame, optional): 分析数据。默认为None。
        """
        self.data = data
    
    def set_data(self, data):
        """
        设置分析数据。
        
        Args:
            data (pandas.DataFrame): 待分析的数据框。
            
        Returns:
            Analyzer: 返回self以支持链式调用。
        """
        self.data = data
        return self
        
    def analyze(self):
        """
        执行分析。
        
        需要由子类实现具体分析逻辑。
        
        Returns:
            pandas.DataFrame: 分析结果。
            
        Raises:
            NotImplementedError: 如果子类没有实现此方法。
        """
        raise NotImplementedError("子类必须实现analyze方法")
        
    def visualize(self):
        """
        可视化分析结果。
        
        需要由子类实现具体可视化逻辑。
        
        Returns:
            matplotlib.Figure: 可视化结果图表。
            
        Raises:
            NotImplementedError: 如果子类没有实现此方法。
        """
        raise NotImplementedError("子类必须实现visualize方法")

# 导出子模块中的类
from .technical import TechnicalAnalyzer
from .fundamental import FundamentalAnalyzer
from .statistical import StatisticalAnalyzer
from .visualization import StockVisualizer
# 启用聚类分析
from .clustering import ClusteringAnalyzer

__all__ = ['Analyzer', 'TechnicalAnalyzer', 'FundamentalAnalyzer', 'StatisticalAnalyzer', 'StockVisualizer', 'ClusteringAnalyzer']
