"""
配置管理模块
"""
import os
import json
import logging

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path=None):
        """初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认为项目根目录下的config.json
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                                      'config', 'config.json')
        self.config = self.load_config()
        
    def load_config(self):
        """加载配置文件
        
        Returns:
            dict: 配置字典
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
        """保存配置文件
        
        Args:
            config: 要保存的配置字典，默认为当前配置
            
        Returns:
            bool: 是否保存成功
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
        """获取配置项
        
        Args:
            key: 配置项键
            default: 默认值
            
        Returns:
            配置项值
        """
        return self.config.get(key, default)
        
    def set(self, key, value):
        """设置配置项
        
        Args:
            key: 配置项键
            value: 配置项值
            
        Returns:
            bool: 是否设置成功
        """
        self.config[key] = value
        return self.save_config() 