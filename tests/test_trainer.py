import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.training.trainer import YOLOTrainer
from src.utils.config_manager import ConfigManager

class TestYOLOTrainer:
    
    def test_init(self, sample_config, temp_dir):
        """Test trainer initialization."""
        trainer = YOLOTrainer(sample_config)
        assert trainer.config['epochs'] == 1
        assert trainer.experiment_name.startswith('exp_')
        assert trainer.project_root.exists()

    def test_init_with_experiment_name(self, sample_config):
        """Test initialization with custom experiment name."""
        trainer = YOLOTrainer(sample_config, experiment_name="custom_exp")
        assert trainer.experiment_name == "custom_exp"

    def test_init_with_config_manager(self, sample_config):
        """Test initialization with injected ConfigManager."""
        cm = ConfigManager(sample_config)
        trainer = YOLOTrainer(sample_config, config_manager=cm)
        assert trainer.config_manager == cm

    def test_prepare_training_params(self, sample_config, sample_data_yaml):
        """Test preparation of training parameters."""
        trainer = YOLOTrainer(sample_config)
        params = trainer._prepare_training_params(
            data_yaml=sample_data_yaml,
            epochs=10,
            batch_size=4,
            imgsz=320,
            device='cpu',
            project=None
        )
        
        assert params['epochs'] == 10
        assert params['batch'] == 4
        assert params['imgsz'] == 320
        assert params['data'] == str(sample_data_yaml)
        assert params['device'] == 'cpu'

    @patch('src.training.trainer.ModelUtils.check_gpu_availability')
    @patch('src.training.trainer.YOLO')
    def test_train(self, mock_yolo, mock_gpu, sample_config, sample_data_yaml, temp_dir):
        """Test training execution."""
        # Setup mocks
        mock_gpu.return_value = {'cuda_available': False, 'devices': []}
        
        # Mock YOLO instance and train method
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance
        
        mock_results = MagicMock()
        mock_results.save_dir = str(temp_dir / 'runs/detect/exp')
        mock_model_instance.train.return_value = mock_results
        
        # Create dummy pretrained model to avoid download
        pretrained_dir = Path(__file__).parent.parent.parent.parent / 'models' / 'pretrained'
        pretrained_dir.mkdir(parents=True, exist_ok=True)
        (pretrained_dir / 'yolov11s.pt').touch()
        
        trainer = YOLOTrainer(sample_config)
        
        # Mock project root to point to temp dir for isolation if needed, 
        # but here we just need to ensure the pretrained model check passes.
        # We can mock the existence check or create the file.
        # Let's mock the file existence check in the trainer to avoid file system issues
        
        with patch('pathlib.Path.exists', return_value=True):
             results = trainer.train(
                data_yaml=sample_data_yaml,
                epochs=1
            )
        
        assert results['experiment_name'] == trainer.experiment_name
        mock_model_instance.train.assert_called_once()

    def test_resume_training(self, sample_config, temp_dir):
        """Test resuming training."""
        trainer = YOLOTrainer(sample_config)
        checkpoint_path = temp_dir / 'last.pt'
        checkpoint_path.touch()
        
        with patch('src.training.trainer.YOLO') as mock_yolo:
            mock_instance = mock_yolo.return_value
            mock_instance.train.return_value = MagicMock(save_dir='/tmp/save_dir')
            
            results = trainer.resume_training(checkpoint_path)
            
            assert results['resumed_from'] == str(checkpoint_path)
            mock_yolo.assert_called_with(str(checkpoint_path))
