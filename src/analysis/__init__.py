"""
股票数据分析模块

提供各种分析工具，用于分析股票数据，包括技术分析、基本面分析、统计分析和聚类分析
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Analyzer:
    """分析器基类"""
    
    def __init__(self, data=None):
        """
        初始化分析器
        
        Args:
            data (pandas.DataFrame, optional): 分析数据. Defaults to None.
        """
        self.data = data
    
    def set_data(self, data):
        """
        设置分析数据
        
        Args:
            data (pandas.DataFrame): 分析数据
        """
        self.data = data
        
    def analyze(self):
        """
        执行分析
        
        需要由子类实现具体分析逻辑
        
        Returns:
            pandas.DataFrame: 分析结果
        """
        raise NotImplementedError("子类必须实现analyze方法")
        
    def visualize(self):
        """
        可视化分析结果
        
        需要由子类实现具体可视化逻辑
        
        Returns:
            matplotlib.Figure: 可视化结果
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
