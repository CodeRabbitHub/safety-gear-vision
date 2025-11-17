"""
Configuration management utility for loading and validating YAML configs.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import json


class ConfigManager:
    """Handles loading and validating configuration files."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize config manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config: Dict[str, Any] = {}
        
        if self.config_path and self.config_path.exists():
            self.load_config()
    
    def load_config(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file (overrides init path)
        
        Returns:
            Configuration dictionary
        """
        path = Path(config_path) if config_path else self.config_path
        
        if not path or not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        return self.config
    
    def save_config(self, config: Dict[str, Any], output_path: Path):
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation: 'parent.child')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def validate_required_keys(self, required_keys: list) -> bool:
        """
        Validate that required keys exist in config.
        
        Args:
            required_keys: List of required keys (dot notation supported)
        
        Returns:
            True if all keys exist
        
        Raises:
            ValueError: If required keys are missing
        """
        missing_keys = []
        
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()
    
    def to_json(self, indent: int = 2) -> str:
        """Return configuration as JSON string."""
        return json.dumps(self.config, indent=indent)
    
    def __repr__(self) -> str:
        return f"ConfigManager(config_path={self.config_path}, keys={len(self.config)})"
