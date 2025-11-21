import pytest
import shutil
from pathlib import Path
import yaml
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory that is cleaned up after test."""
    return tmp_path

@pytest.fixture
def sample_config(temp_dir):
    """Create a sample training configuration file."""
    config = {
        'model': 'yolov11s.pt',
        'epochs': 1,
        'batch': 2,
        'imgsz': 640,
        'device': 'cpu',
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'lrf': 0.01,
        'patience': 5,
        'save_period': 1,
        'workers': 0,
        'project': str(temp_dir / 'runs'),
        'name': 'test_exp'
    }
    
    config_path = temp_dir / 'test_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

@pytest.fixture
def sample_data_yaml(temp_dir):
    """Create a sample dataset YAML file."""
    data_config = {
        'path': str(temp_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {0: 'person', 1: 'helmet', 2: 'vest'}
    }
    
    data_yaml_path = temp_dir / 'data.yaml'
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f)
        
    return data_yaml_path

@pytest.fixture
def sample_image(temp_dir):
    """Create a sample dummy image."""
    img_path = temp_dir / 'test_image.jpg'
    # Create a black image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return img_path

@pytest.fixture
def mock_yolo():
    """Mock the Ultralytics YOLO class."""
    with patch('ultralytics.YOLO') as mock:
        # Setup mock instance
        instance = mock.return_value
        
        # Mock train method
        instance.train.return_value = MagicMock(
            save_dir='/tmp/runs/detect/exp',
            results_dict={'metrics/mAP50(B)': 0.95}
        )
        
        # Mock predict method
        mock_result = MagicMock()
        box = MagicMock()
        box.cls = MagicMock(item=lambda: 0)
        box.conf = MagicMock(item=lambda: 0.9)
        box.xyxy = MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([[0, 0, 100, 100]])))
        box.xywhn = MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([[0.5, 0.5, 0.2, 0.2]])))
        
        mock_result.boxes = [box]
        mock_result.save_dir = '/tmp/runs/detect/exp'
        instance.predict.return_value = [mock_result]
        
        # Mock val method
        instance.val.return_value = {'metrics/mAP50(B)': 0.95}
        
        yield mock
