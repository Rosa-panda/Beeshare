"""
数据源模块

提供各种数据源的实现和管理。
"""

from src.data.data_source.data_source_interface import DataSourceInterface, DataSourceError
from src.data.data_source.akshare_data_source import AKShareDataSource
from src.data.data_source.data_source_factory import DataSourceFactory

__all__ = ['DataSourceInterface', 'DataSourceError', 'AKShareDataSource', 'DataSourceFactory'] 