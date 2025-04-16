"""
数据源配置模块

管理数据源的配置信息，提供默认配置和配置加载功能。
"""

import os
import logging
import json
from typing import Dict, Any, Optional


class DataSourceConfig:
    """数据源配置类
    
    管理数据源配置信息，提供默认配置和配置加载功能。
    """
    
    # 默认配置
    DEFAULT_CONFIG = {
        'akshare': {
            'stock_list_cache_time': 86400,  # 股票列表缓存时间，默认1天
            'retry_count': 3,               # 重试次数
            'retry_delay': 1,               # 重试延迟（秒）
            'timeout': 30,                  # 请求超时时间（秒）
            'auto_initialize': True         # 自动初始化
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置类
        
        Args:
            config_path: 配置文件路径，可选。如果为None，则使用默认配置。
        """
        self.logger = logging.getLogger('BeeShare.DataSourceConfig')
        self.config = self.DEFAULT_CONFIG.copy()
        self.config_path = config_path
        
        # 如果提供了配置文件路径，则尝试加载配置
        if config_path and os.path.exists(config_path):
            try:
                self.load_config(config_path)
            except Exception as e:
                self.logger.error(f"加载配置文件失败: {str(e)}，使用默认配置")
    
    def load_config(self, config_path: str) -> None:
        """从文件加载配置
        
        Args:
            config_path: 配置文件路径
            
        Raises:
            FileNotFoundError: 文件不存在时抛出
            json.JSONDecodeError: JSON解析错误时抛出
        """
        self.logger.info(f"从文件加载配置: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                
            # 更新默认配置
            for source_name, source_config in loaded_config.items():
                if source_name in self.config:
                    self.config[source_name].update(source_config)
                else:
                    self.config[source_name] = source_config
                    
            self.logger.info("配置加载成功")
            
        except FileNotFoundError:
            self.logger.error(f"配置文件不存在: {config_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"配置文件格式错误: {str(e)}")
            raise
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """保存配置到文件
        
        Args:
            config_path: 配置文件路径，如果为None则使用初始化时的路径
            
        Raises:
            ValueError: 未指定配置路径时抛出
            IOError: 写入文件失败时抛出
        """
        path = config_path or self.config_path
        if not path:
            raise ValueError("未指定配置文件路径")
        
        self.logger.info(f"保存配置到文件: {path}")
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
                
            self.logger.info("配置保存成功")
            
        except IOError as e:
            self.logger.error(f"保存配置文件失败: {str(e)}")
            raise
    
    def get_source_config(self, source_name: str) -> Dict[str, Any]:
        """获取指定数据源的配置
        
        Args:
            source_name: 数据源名称
            
        Returns:
            Dict[str, Any]: 数据源配置，如果不存在则返回空字典
        """
        return self.config.get(source_name, {}).copy()
    
    def set_source_config(self, source_name: str, config: Dict[str, Any]) -> None:
        """设置指定数据源的配置
        
        Args:
            source_name: 数据源名称
            config: 数据源配置
        """
        self.config[source_name] = config
        self.logger.info(f"更新数据源配置: {source_name}")
    
    def update_source_config(self, source_name: str, config: Dict[str, Any]) -> None:
        """更新指定数据源的配置（合并而非替换）
        
        Args:
            source_name: 数据源名称
            config: 要更新的配置项
        """
        if source_name not in self.config:
            self.config[source_name] = {}
            
        self.config[source_name].update(config)
        self.logger.info(f"更新数据源配置项: {source_name}")
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """获取所有数据源配置
        
        Returns:
            Dict[str, Dict[str, Any]]: 所有数据源配置的副本
        """
        return self.config.copy()


# 创建全局配置实例
default_config_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'data_source_config.json')
data_source_config = DataSourceConfig(default_config_path) 