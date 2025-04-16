"""
存储工厂模块，用于创建和管理SQLite存储实例

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
    
    StorageFactory --> Storage: 创建
    Storage <|-- SQLiteStorage: 继承
```
"""

import logging
from typing import Dict, Optional, Any
from .base import Storage
from .sqlite_storage import SQLiteStorage

logger = logging.getLogger(__name__)

class StorageFactory:
    """
    存储工厂类，用于创建和管理SQLite存储实例
    """
    
    # 存储实例缓存，键为存储类型，值为存储实例
    _instances: Dict[str, Storage] = {}
    
    @classmethod
    def get_storage(cls, storage_type: str, config: Optional[Dict[str, Any]] = None) -> Storage:
        """
        获取指定类型的存储实例
        
        Args:
            storage_type (str): 存储类型，目前只支持'sqlite'
            config (dict, optional): 存储配置. Defaults to None.
        
        Returns:
            Storage: 存储实例
        
        Raises:
            ValueError: 如果指定的存储类型不是'sqlite'
        """
        # 强制使用sqlite存储
        storage_type = 'sqlite'
        
        # 如果实例已存在且无新配置，则返回现有实例
        if storage_type in cls._instances and config is None:
            logger.debug(f"返回已有的 {storage_type} 存储实例")
            return cls._instances[storage_type]
        
        # 创建新的存储实例
        try:
            instance = SQLiteStorage(config)
            
            # 缓存实例
            cls._instances[storage_type] = instance
            logger.info(f"创建新的 {storage_type} 存储实例")
            
            return instance
            
        except Exception as e:
            logger.error(f"创建 {storage_type} 存储实例失败: {e}")
            raise ValueError(f"无法创建SQLite存储实例: {e}")
    
    @classmethod
    def get_default_storage(cls, config: Optional[Dict[str, Any]] = None) -> Storage:
        """
        获取默认的存储实例
        
        Args:
            config (dict, optional): 存储配置. Defaults to None.
        
        Returns:
            Storage: 默认存储实例 (SQLite存储)
        """
        return cls.get_storage('sqlite', config)
    
    @classmethod
    def clear_instances(cls) -> None:
        """
        清除所有缓存的存储实例
        """
        cls._instances.clear()
        logger.info("已清除所有存储实例缓存") 