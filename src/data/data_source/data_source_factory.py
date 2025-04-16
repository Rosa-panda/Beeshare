"""
数据源工厂模块

提供创建和管理不同数据源的工厂类。
"""

import logging
from typing import Dict, Any, List, Optional, Type
from src.data.data_source.data_source_interface import DataSourceInterface, DataSourceError
from src.data.data_source.akshare_data_source import AKShareDataSource


class DataSourceFactory:
    """数据源工厂类
    
    负责创建和管理不同的数据源实例。
    """
    
    # 数据源类映射
    _source_mapping: Dict[str, Type[DataSourceInterface]] = {
        'akshare': AKShareDataSource
    }
    
    # 数据源实例缓存
    _instances: Dict[str, DataSourceInterface] = {}
    
    @classmethod
    def register_data_source(cls, name: str, source_class: Type[DataSourceInterface]) -> None:
        """注册新的数据源类
        
        Args:
            name: 数据源名称
            source_class: 数据源类，必须是DataSourceInterface的子类
            
        Raises:
            ValueError: 当数据源名称已存在或类型不正确时抛出
        """
        if name in cls._source_mapping:
            raise ValueError(f"数据源名称 '{name}' 已经存在")
        
        if not issubclass(source_class, DataSourceInterface):
            raise ValueError(f"数据源类必须继承自DataSourceInterface")
        
        cls._source_mapping[name] = source_class
        logging.getLogger('BeeShare.DataSourceFactory').info(f"注册数据源: {name}")
    
    @classmethod
    def create_data_source(cls, name: str, config: Dict[str, Any] = None) -> DataSourceInterface:
        """创建指定名称的数据源实例
        
        Args:
            name: 数据源名称
            config: 数据源配置，可选
            
        Returns:
            DataSourceInterface: 数据源实例
            
        Raises:
            DataSourceError: 当创建数据源失败时抛出
        """
        logger = logging.getLogger('BeeShare.DataSourceFactory')
        
        if name not in cls._source_mapping:
            error_msg = f"未知的数据源名称: {name}"
            logger.error(error_msg)
            raise DataSourceError(error_msg, source="DataSourceFactory", error_code=3001)
        
        try:
            source_class = cls._source_mapping[name]
            instance = source_class(config)
            logger.info(f"创建数据源实例: {name}")
            return instance
        except Exception as e:
            error_msg = f"创建数据源 {name} 实例失败: {str(e)}"
            logger.error(error_msg)
            raise DataSourceError(error_msg, source="DataSourceFactory", 
                                 error_code=3002, original_error=e)
    
    @classmethod
    def get_data_source(cls, name: str, config: Dict[str, Any] = None, 
                       reuse: bool = True) -> DataSourceInterface:
        """获取指定名称的数据源实例，支持实例复用
        
        Args:
            name: 数据源名称
            config: 数据源配置，可选
            reuse: 是否复用已有实例，默认为True
            
        Returns:
            DataSourceInterface: 数据源实例
        """
        logger = logging.getLogger('BeeShare.DataSourceFactory')
        
        # 如果要求复用且实例已存在，直接返回
        if reuse and name in cls._instances:
            instance = cls._instances[name]
            # 如果提供了新配置，则更新实例配置
            if config:
                instance.initialize(config)
            return instance
        
        # 创建新实例
        instance = cls.create_data_source(name, config)
        
        # 如果要求复用，则缓存实例
        if reuse:
            cls._instances[name] = instance
            logger.info(f"缓存数据源实例: {name}")
        
        return instance
    
    @classmethod
    def get_available_sources(cls) -> List[str]:
        """获取所有可用的数据源名称
        
        Returns:
            List[str]: 数据源名称列表
        """
        return list(cls._source_mapping.keys())
    
    @classmethod
    def clear_instances(cls, name: Optional[str] = None) -> None:
        """清除数据源实例缓存
        
        Args:
            name: 数据源名称，如果为None则清除所有缓存
        """
        logger = logging.getLogger('BeeShare.DataSourceFactory')
        
        if name is None:
            cls._instances.clear()
            logger.info("清除所有数据源实例缓存")
        elif name in cls._instances:
            del cls._instances[name]
            logger.info(f"清除数据源实例缓存: {name}") 