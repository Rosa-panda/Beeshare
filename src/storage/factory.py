from .postgresql_storage import PostgreSQLStorage
"""
存储工厂模块，用于创建和管理存储实例

流程图:
```mermaid
classDiagram
    class StorageFactory {
        -_instances: dict
        +get_storage(storage_type: str, config: dict): Storage
        +get_default_storage(config: dict): Storage
        +clear_instances()
    }
    class Storage {
        <<abstract>>
        +save_data()
        +load_data()
        +delete_data()
        +exists()
    }
    class SQLiteStorage {
        +save_data()
        +load_data()
        +delete_data()
        +exists()
    }
    class PostgreSQLStorage {
        +save_data()
        +load_data()
        +delete_data()
        +exists()
    }
    
    StorageFactory --> Storage: 创建
    Storage <|-- SQLiteStorage: 继承
    Storage <|-- PostgreSQLStorage: 继承
```
"""

import logging
from typing import Dict, Optional, Any
from .base import Storage
from .sqlite_storage import SQLiteStorage

logger = logging.getLogger(__name__)

def create_storage(storage_type=None, config=None):
    """创建存储实例
    
    Args:
        storage_type: 存储类型，如'sqlite'或'postgresql'
        config: 存储配置
    
    Returns:
        Storage的子类实例
    """
    if storage_type is None:
        # 从配置文件加载存储类型和配置
        from .config import StorageConfig
        storage_config = StorageConfig()
        storage_type = storage_config.get_storage_type()
        config = storage_config.get_storage_config(storage_type)
    
    # 根据存储类型创建实例
    if storage_type.lower() == 'sqlite':
        storage = SQLiteStorage(config.get('db_path', 'data/beeshare.db'))
    elif storage_type.lower() in ['postgresql', 'postgres', 'timescaledb']:
        storage = PostgreSQLStorage(config)
    else:
        raise ValueError(f"不支持的存储类型: {storage_type}")
    
    return storage

class StorageFactory:
    """
    存储工厂类，用于创建和管理存储实例
    """
    
    # 存储实例缓存，键为存储类型，值为存储实例
    _instances: Dict[str, Storage] = {}
    
    @classmethod
    def get_storage(cls, storage_type: str, config: Optional[Dict[str, Any]] = None) -> Storage:
        """
        获取指定类型的存储实例
        
        Args:
            storage_type (str): 存储类型，支持'sqlite'和'postgresql'
            config (dict, optional): 存储配置. Defaults to None.
        
        Returns:
            Storage: 存储实例
        
        Raises:
            ValueError: 如果指定的存储类型不支持
        """
        # 如果实例已存在且无新配置，则返回现有实例
        if storage_type in cls._instances and config is None:
            logger.debug(f"返回已有的 {storage_type} 存储实例")
            return cls._instances[storage_type]
        
        # 创建新的存储实例
        try:
            if storage_type.lower() == 'sqlite':
                instance = SQLiteStorage(config.get('db_path', 'data/beeshare.db'))
            elif storage_type.lower() in ['postgresql', 'postgres', 'timescaledb']:
                instance = PostgreSQLStorage(config)
            else:
                raise ValueError(f"不支持的存储类型: {storage_type}")
            
            # 缓存实例
            cls._instances[storage_type] = instance
            logger.info(f"创建新的 {storage_type} 存储实例")
            
            return instance
            
        except Exception as e:
            logger.error(f"创建 {storage_type} 存储实例失败: {e}")
            raise ValueError(f"无法创建存储实例: {e}")
    
    @classmethod
    def get_default_storage(cls, config: Optional[Dict[str, Any]] = None) -> Storage:
        """
        获取默认的存储实例
        
        Args:
            config (dict, optional): 存储配置. Defaults to None.
        
        Returns:
            Storage: 默认存储实例
        """
        # 从配置文件读取默认存储类型
        from .config import StorageConfig
        storage_config = StorageConfig()
        default_type = storage_config.get_storage_type()
        
        return cls.get_storage(default_type, config or storage_config.get_storage_config(default_type))
    
    @classmethod
    def clear_instances(cls) -> None:
        """
        清除所有缓存的存储实例
        """
        cls._instances.clear()
        logger.info("已清除所有存储实例缓存") 