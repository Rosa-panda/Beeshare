"""
配置管理模块。

该模块提供配置文件的读取、写入和访问功能，用于管理项目的各种配置项。
支持JSON格式的配置文件，提供层次化的配置访问方式。

创建日期: 2024-04-05
最后修改: 2024-04-18
作者: BeeShare开发团队
"""
import os
import json
import logging

class ConfigManager:
    """
    配置管理器类。
    
    负责配置文件的加载、保存和访问，支持JSON格式的配置文件。
    提供简单的键值对方式访问配置项，并能够将配置持久化到文件。
    
    Attributes:
        logger (logging.Logger): 日志记录器实例
        config_path (str): 配置文件的完整路径
        config (dict): 当前加载的配置内容
    """
    
    def __init__(self, config_path=None):
        """
        初始化配置管理器。
        
        Args:
            config_path (str, optional): 配置文件路径，默认为项目根目录下的config/config.json
            
        Raises:
            OSError: 当配置文件路径有问题时可能抛出
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                                      'config', 'config.json')
        self.config = self.load_config()
        
    def load_config(self):
        """
        从文件加载配置。
        
        尝试从指定的配置文件中加载JSON格式的配置。
        如果文件不存在或读取失败，将返回空字典作为默认配置。
        
        Returns:
            dict: 加载的配置字典
            
        Raises:
            JSONDecodeError: 当配置文件格式不正确时可能抛出
            PermissionError: 当没有文件读取权限时可能抛出
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"配置文件 {self.config_path} 不存在，将使用默认配置")
                return {}
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            return {}
            
    def save_config(self, config=None):
        """
        将配置保存到文件。
        
        将当前配置或指定的配置字典保存到配置文件中。
        如果配置文件所在目录不存在，会自动创建。
        
        Args:
            config (dict, optional): 要保存的配置字典，默认为当前配置
            
        Returns:
            bool: 是否保存成功
            
        Raises:
            PermissionError: 当没有文件写入权限时可能抛出
            TypeError: 当配置无法序列化为JSON时可能抛出
        """
        try:
            config_dir = os.path.dirname(self.config_path)
            if not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
                
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config or self.config, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            self.logger.error(f"保存配置文件失败: {e}")
            return False
            
    def get(self, key, default=None):
        """
        获取配置项的值。
        
        Args:
            key (str): 配置项的键名
            default (any, optional): 当配置项不存在时返回的默认值
            
        Returns:
            any: 配置项的值，如果不存在则返回默认值
            
        Examples:
            >>> config = ConfigManager()
            >>> db_host = config.get('database.host', 'localhost')
            >>> timeout = config.get('api.timeout', 30)
        """
        return self.config.get(key, default)
        
    def set(self, key, value):
        """
        设置配置项的值并保存到文件。
        
        Args:
            key (str): 配置项的键名
            value (any): 配置项的值，必须是可JSON序列化的类型
            
        Returns:
            bool: 是否设置并保存成功
            
        Examples:
            >>> config = ConfigManager()
            >>> config.set('database.host', '127.0.0.1')
            >>> config.set('api.timeout', 60)
        """
        self.config[key] = value
        return self.save_config() 