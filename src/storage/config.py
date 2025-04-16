"""
存储配置管理模块，用于管理不同存储引擎的配置和迁移

流程图:
```mermaid
classDiagram
    class StorageConfig {
        -_config: dict
        +load_config()
        +save_config()
        +get_storage_type() 
        +set_storage_type(storage_type)
        +get_storage_config(storage_type)
        +update_storage_config(storage_type, config)
    }
    
    class StorageMigration {
        +migrate(source_storage, target_storage)
        -_migrate_data(source, target, data_type, symbols)
    }
    
    StorageConfig --> StorageMigration: 使用
```
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd

from .base import Storage, DataType
from .factory import StorageFactory

logger = logging.getLogger(__name__)

# 获取项目根目录
def get_project_root() -> Path:
    """获取项目根目录的绝对路径"""
    # 当前文件所在目录
    current_dir = Path(__file__).parent.absolute()
    # 向上两级即为项目根目录
    return current_dir.parent.parent

# 默认配置文件路径
DEFAULT_CONFIG_PATH = os.path.join(get_project_root(), 'config', 'storage.json')

# 默认存储配置
DEFAULT_CONFIG = {
    "active_storage": "sqlite",
    "storage_configs": {
        "sqlite": {
            "db_path": os.path.join(get_project_root(), 'data', 'stock_data.db')
        }
    }
}

class StorageConfig:
    """
    存储配置管理类，用于管理不同存储引擎的配置
    """
    
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        """
        初始化存储配置管理器
        
        Args:
            config_path (str, optional): 配置文件路径. Defaults to DEFAULT_CONFIG_PATH.
        """
        self._config_path = config_path
        self._config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        从配置文件加载配置，如果文件不存在则使用默认配置
        
        Returns:
            dict: 加载的配置
        """
        try:
            # 确保配置目录存在
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
            
            if os.path.exists(self._config_path):
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"从 {self._config_path} 加载存储配置")
                return config
            else:
                # 如果配置文件不存在，使用默认配置并保存
                logger.warning(f"配置文件 {self._config_path} 不存在，使用默认配置")
                self._config = DEFAULT_CONFIG.copy()
                self.save_config()
                return self._config
        except Exception as e:
            logger.error(f"加载存储配置失败: {e}")
            return DEFAULT_CONFIG.copy()
    
    def save_config(self) -> None:
        """
        保存配置到配置文件
        """
        try:
            # 确保配置目录存在
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
            
            with open(self._config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4)
            logger.info(f"存储配置已保存到 {self._config_path}")
        except Exception as e:
            logger.error(f"保存存储配置失败: {e}")
    
    def get_storage_type(self) -> str:
        """
        获取当前激活的存储类型，系统现在只支持SQLite存储
        
        Returns:
            str: 始终返回'sqlite'
        """
        return "sqlite"
    
    def set_storage_type(self, storage_type: str, migrate: bool = False) -> None:
        """
        设置激活的存储类型。
        注意：系统现在只支持SQLite存储，所以此方法仅为了兼容性保留。
        
        Args:
            storage_type (str): 存储类型，将被忽略并强制设置为'sqlite'
            migrate (bool, optional): 是否迁移数据，此参数将被忽略. Defaults to False.
        """
        if storage_type != 'sqlite':
            logger.warning(f"系统现在只支持SQLite存储，忽略设置存储类型为{storage_type}的请求")
        
        # 确保配置中的active_storage为sqlite
        if self._config.get("active_storage") != "sqlite":
            self._config["active_storage"] = "sqlite"
            self.save_config()
            logger.info("存储类型已设置为SQLite")
    
    def get_storage_config(self, storage_type: Optional[str] = None) -> Dict[str, Any]:
        """
        获取指定存储类型的配置
        
        Args:
            storage_type (str, optional): 存储类型. 如果为None，则使用当前激活的存储类型.
        
        Returns:
            dict: 存储配置
        """
        if storage_type is None:
            storage_type = self.get_storage_type()
        
        storage_configs = self._config.get("storage_configs", {})
        config = storage_configs.get(storage_type, {})
        
        # 确保返回的是配置的副本，避免修改原配置
        return config.copy()
    
    def update_storage_config(self, storage_type: str, config: Dict[str, Any]) -> None:
        """
        更新指定存储类型的配置
        
        Args:
            storage_type (str): 存储类型
            config (dict): 新的配置
        """
        if "storage_configs" not in self._config:
            self._config["storage_configs"] = {}
        
        # 更新配置（合并而不是替换）
        current_config = self._config["storage_configs"].get(storage_type, {})
        current_config.update(config)
        self._config["storage_configs"][storage_type] = current_config
        
        self.save_config()
        logger.info(f"{storage_type} 的存储配置已更新")
        
        # 如果更新的是当前激活的存储，刷新存储实例
        if storage_type == self.get_storage_type():
            # 清除存储工厂中的缓存实例
            StorageFactory.clear_instances()


class StorageMigration:
    """
    存储迁移工具，用于在不同存储引擎之间迁移数据
    """
    
    def migrate(self, source_storage: Storage, target_storage: Storage) -> bool:
        """
        执行数据迁移，从源存储迁移到目标存储
        
        Args:
            source_storage (Storage): 源存储
            target_storage (Storage): 目标存储
        
        Returns:
            bool: 迁移是否成功
        """
        try:
            logger.info(f"开始数据迁移: {source_storage.__class__.__name__} -> {target_storage.__class__.__name__}")
            
            # 迁移所有类型的数据
            data_types = [DataType.HISTORICAL, DataType.REALTIME, DataType.SYMBOL, DataType.INDEX]
            
            for data_type in data_types:
                # 首先迁移股票符号数据，因为其他数据依赖于它
                if data_type == DataType.SYMBOL:
                    try:
                        symbol_data = source_storage.load_data(data_type=data_type)
                        if not symbol_data.empty:
                            target_storage.save_data(data=symbol_data, data_type=data_type)
                            logger.info(f"已迁移股票符号数据: {len(symbol_data)} 条记录")
                    except Exception as e:
                        logger.error(f"迁移股票符号数据失败: {e}")
                
                # 迁移其他类型数据
                else:
                    try:
                        # 获取所有股票符号
                        symbol_data = source_storage.load_data(data_type=DataType.SYMBOL)
                        if symbol_data.empty:
                            logger.warning(f"没有找到股票符号数据，跳过 {data_type} 数据迁移")
                            continue
                        
                        symbols = symbol_data['symbol'].unique().tolist()
                        self._migrate_data(source_storage, target_storage, data_type, symbols)
                    except Exception as e:
                        logger.error(f"迁移 {data_type} 数据失败: {e}")
            
            logger.info("数据迁移完成")
            return True
            
        except Exception as e:
            logger.error(f"数据迁移失败: {e}")
            return False
    
    def _migrate_data(self, source: Storage, target: Storage, data_type: DataType, symbols: List[str]) -> None:
        """
        迁移特定类型的数据
        
        Args:
            source (Storage): 源存储
            target (Storage): 目标存储
            data_type (DataType): 数据类型
            symbols (List[str]): 股票符号列表
        """
        total_migrated = 0
        
        for symbol in symbols:
            try:
                # 加载源数据
                data = source.load_data(data_type=data_type, symbol=symbol)
                if data.empty:
                    continue
                
                # 保存到目标存储
                target.save_data(data=data, data_type=data_type)
                total_migrated += len(data)
                logger.debug(f"已迁移 {symbol} 的 {data_type} 数据: {len(data)} 条记录")
            except Exception as e:
                logger.error(f"迁移 {symbol} 的 {data_type} 数据失败: {e}")
        
        logger.info(f"已迁移 {data_type} 数据: 共 {total_migrated} 条记录") 