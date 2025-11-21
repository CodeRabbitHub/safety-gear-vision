import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.inference.predictor import YOLOPredictor

class TestYOLOPredictor:
    
    def test_init(self, temp_dir):
        """Test predictor initialization."""
        weights_path = temp_dir / 'yolov11s.pt'
        weights_path.touch()
        
        with patch('src.inference.predictor.ModelUtils.load_model') as mock_load:
            predictor = YOLOPredictor(weights_path)
            assert predictor.conf_threshold == 0.5
            mock_load.assert_called_once()

    def test_init_with_model(self, temp_dir):
        """Test initialization with injected model."""
        weights_path = temp_dir / 'yolov11s.pt'
        weights_path.touch()
        mock_model = MagicMock()
        
        predictor = YOLOPredictor(weights_path, model=mock_model)
        assert predictor.model == mock_model

    def test_predict_image(self, temp_dir, sample_image):
        """Test single image prediction."""
        weights_path = temp_dir / 'yolov11s.pt'
        weights_path.touch()
        mock_model = MagicMock()
        
        # Mock prediction result
        mock_result = MagicMock()
        box = MagicMock()
        box.cls = MagicMock(item=lambda: 0)
        box.conf = MagicMock(item=lambda: 0.9)
        box.xyxy = MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([[0, 0, 100, 100]])))
        box.xywhn = MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([[0.5, 0.5, 0.2, 0.2]])))
        
        mock_result.boxes = [box]
        mock_model.predict.return_value = [mock_result]
        
        predictor = YOLOPredictor(weights_path, model=mock_model)
        
        results = predictor.predict_image(sample_image, draw_boxes=False)
        
        assert results['num_detections'] == 1
        assert results['detections'][0]['class_name'] == 'Person'
        assert results['image_path'] == str(sample_image)

    def test_predict_batch(self, temp_dir, sample_image):
        """Test batch prediction."""
        weights_path = temp_dir / 'yolov11s.pt'
        weights_path.touch()
        mock_model = MagicMock()
        
        # Mock prediction result
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_model.predict.return_value = [mock_result]
        
        predictor = YOLOPredictor(weights_path, model=mock_model)
        
        # Create multiple images
        img2 = temp_dir / 'test_image2.jpg'
        import shutil
        shutil.copy(sample_image, img2)
        
        results = predictor.predict_batch(temp_dir, save_results=False)
        
        assert len(results) >= 2  # At least 2 images in temp_dir

    def test_parse_results(self, temp_dir):
        """Test result parsing logic."""
        weights_path = temp_dir / 'yolov11s.pt'
        weights_path.touch()
        predictor = YOLOPredictor(weights_path, model=MagicMock())
        
        mock_result = MagicMock()
        box = MagicMock()
        box.cls = MagicMock(item=lambda: 14) # Helmet
        box.conf = MagicMock(item=lambda: 0.85)
        box.xyxy = MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([[10, 10, 50, 50]])))
        box.xywhn = MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([[0.1, 0.1, 0.05, 0.05]])))
        
        mock_result.boxes = [box]
        
        parsed = predictor._parse_results(mock_result)
        
        assert parsed['num_detections'] == 1
        assert parsed['detections'][0]['class_name'] == 'Helmet'
        assert parsed['detections'][0]['confidence'] == 0.85
