"""
配置加载工具类，用于加载YAML配置文件
"""
import os
import yaml
from typing import Dict, Any, Optional


class ConfigLoader:
    """配置加载器，负责从YAML文件加载配置"""
    
    _instance = None
    _configs = {}
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    def load_config(self, config_name: str, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_name: 配置名称，用于缓存和引用
            config_path: 配置文件路径
            
        Returns:
            Dict: 配置字典
        """
        if config_name in self._configs:
            return self._configs[config_name]
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                self._configs[config_name] = config
                return config
        except Exception as e:
            raise ValueError(f"加载配置文件失败: {str(e)}")
    
    def get_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        获取已加载的配置
        
        Args:
            config_name: 配置名称
            
        Returns:
            Dict: 配置字典，如果未加载返回None
        """
        return self._configs.get(config_name)
    
    def get_config_value(self, config_name: str, path: str, default=None) -> Any:
        """
        获取配置中的特定值
        
        Args:
            config_name: 配置名称
            path: 配置路径，使用点号分隔，如"server.host"
            default: 默认值，当配置项不存在时返回
            
        Returns:
            Any: 配置值或默认值
        """
        config = self.get_config(config_name)
        if not config:
            return default
        
        keys = path.split('.')
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def reload_config(self, config_name: str, config_path: str) -> Dict[str, Any]:
        """
        重新加载配置文件
        
        Args:
            config_name: 配置名称
            config_path: 配置文件路径
            
        Returns:
            Dict: 更新后的配置字典
        """
        if config_name in self._configs:
            del self._configs[config_name]
        return self.load_config(config_name, config_path)


# 提供全局访问点
config_loader = ConfigLoader()