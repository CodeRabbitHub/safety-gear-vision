import pytest
import yaml
from pathlib import Path
from src.utils.config_manager import ConfigManager

class TestConfigManager:
    
    def test_init_empty(self):
        """Test initialization without path."""
        cm = ConfigManager()
        assert cm.config == {}
        assert cm.config_path is None

    def test_init_with_path(self, sample_config):
        """Test initialization with valid path."""
        cm = ConfigManager(sample_config)
        assert cm.config['epochs'] == 1
        assert cm.config['model'] == 'yolov11s.pt'

    def test_load_config_error(self):
        """Test loading non-existent config."""
        cm = ConfigManager()
        with pytest.raises(FileNotFoundError):
            cm.load_config('non_existent.yaml')

    def test_get_value(self, sample_config):
        """Test getting values with dot notation."""
        cm = ConfigManager(sample_config)
        assert cm.get('epochs') == 1
        assert cm.get('non_existent') is None
        assert cm.get('non_existent', 'default') == 'default'

    def test_set_value(self):
        """Test setting values with dot notation."""
        cm = ConfigManager()
        cm.set('training.epochs', 100)
        assert cm.get('training.epochs') == 100
        assert cm.config['training']['epochs'] == 100

    def test_validate_required_keys(self, sample_config):
        """Test key validation."""
        cm = ConfigManager(sample_config)
        assert cm.validate_required_keys(['epochs', 'batch']) is True
        
        with pytest.raises(ValueError):
            cm.validate_required_keys(['epochs', 'missing_key'])

    def test_save_config(self, temp_dir):
        """Test saving configuration."""
        cm = ConfigManager()
        cm.set('test_key', 'test_value')
        
        output_path = temp_dir / 'saved_config.yaml'
        cm.save_config(cm.config, output_path)
        
        assert output_path.exists()
        with open(output_path) as f:
            saved_data = yaml.safe_load(f)
        assert saved_data['test_key'] == 'test_value'
