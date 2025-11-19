"""
Training orchestrator for YOLOv11 models.
"""

import os
from pathlib import Path
from typing import Union, Optional, Dict
from datetime import datetime
from ultralytics import YOLO
import torch

from ..utils.config_manager import ConfigManager
from ..utils.logger import get_logger
from ..utils.model_utils import ModelUtils
from ..utils.file_handler import FileHandler


class YOLOTrainer:
    """Handles YOLOv11 model training."""
    
    def __init__(
        self,
        config_path: Union[str, Path],
        experiment_name: Optional[str] = None,
        logger=None
    ):
        """
        Initialize YOLO trainer.
        
        Args:
            config_path: Path to training configuration file
            experiment_name: Name for this training experiment
            logger: Logger instance
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.to_dict()
        
        # Get project root
        self.project_root = Path(__file__).parent.parent.parent
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f"exp_{timestamp}"
        
        self.experiment_name = experiment_name
        self.logger = logger or get_logger('yolo_trainer')
        
        # Initialize TensorBoard directory
        self.tensorboard_dir = self.project_root / 'logs' / 'tensorboard' / self.experiment_name
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.best_model_path = None
    
    def train(
        self,
        data_yaml: Union[str, Path],
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        imgsz: Optional[int] = None,
        device: Optional[str] = None,
        project: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict:
        """
        Train YOLO model.
        
        Args:
            data_yaml: Path to dataset YAML configuration
            epochs: Number of training epochs (overrides config)
            batch_size: Batch size (overrides config)
            imgsz: Image size (overrides config)
            device: Device to train on (overrides config)
            project: Project directory for results
            **kwargs: Additional training arguments
        
        Returns:
            Dictionary with training results
        """
        self.logger.info(f"Starting training experiment: {self.experiment_name}")
        
        # Get training parameters
        train_params = self._prepare_training_params(
            data_yaml, epochs, batch_size, imgsz, device, project, **kwargs
        )
        
        # Log configuration
        self.logger.info("Training configuration:", **train_params)
        
        # Check GPU availability
        gpu_info = ModelUtils.check_gpu_availability()
        self.logger.info(f"GPU available: {gpu_info['cuda_available']}")
        if gpu_info['cuda_available']:
            self.logger.info(f"Using GPU: {gpu_info['devices'][0]['name']}")
        
        # Initialize model
        model_weights = self.config.get('model', 'yolov11s.pt')
        self.logger.info(f"Config file loaded: {self.config_manager.config_path}")
        self.logger.info(f"Loading model: {model_weights}")
        self.logger.info(f"Full config: {self.config}")
        
        # Ensure models/pretrained directory exists
        pretrained_dir = self.project_root / 'models' / 'pretrained'
        FileHandler.ensure_dir(pretrained_dir)
        
        # Set YOLO directories to prevent downloads to random locations
        os.environ['YOLO_HOME'] = str(pretrained_dir)
        os.environ['YOLO_CONFIG_DIR'] = str(self.project_root / 'config')
        
        # Check if model exists in models/pretrained directory BEFORE trying to load
        pretrained_model = pretrained_dir / model_weights
        
        if not pretrained_model.exists():
            self.logger.error(f"❌ Model not found: {pretrained_model}")
            self.logger.error(f"")
            self.logger.error(f"Please download models first using:")
            self.logger.error(f"  poetry run python scripts/00_download_models.py")
            self.logger.error(f"")
            self.logger.error(f"Available models in {pretrained_dir}:")
            if pretrained_dir.exists():
                models = list(pretrained_dir.glob('*.pt'))
                if models:
                    for model_file in sorted(models):
                        self.logger.error(f"  - {model_file.name}")
                else:
                    self.logger.error(f"  (no models found)")
            raise FileNotFoundError(f"Model file not found: {pretrained_model}")
        
        try:
            self.logger.info(f"✓ Loading model from: {pretrained_model}")
            # Use full path to prevent YOLO from trying to download
            self.model = YOLO(str(pretrained_model.absolute()))
            
            # Check if YOLO accidentally downloaded a model to current directory
            cwd_models = list(Path.cwd().glob('yolov*.pt'))
            if cwd_models:
                self.logger.warning(f"⚠️ WARNING: YOLO downloaded models to current directory!")
                for model_file in cwd_models:
                    self.logger.warning(f"  Found: {model_file}")
                    self.logger.warning(f"  Moving to: {pretrained_dir / model_file.name}")
                    model_file.rename(pretrained_dir / model_file.name)
                    
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
        
        # Start training
        self.logger.info("Starting training...")
        
        try:
            results = self.model.train(**train_params)
            
            # Get best model path and TensorBoard logs location
            self.best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
            tensorboard_logs_dir = Path(results.save_dir)
            
            self.logger.info(f"Training complete!")
            self.logger.info(f"Best model saved to: {self.best_model_path}")
            self.logger.info(f"TensorBoard logs saved to: {tensorboard_logs_dir}")
            self.logger.info(f"To view training metrics with TensorBoard, run:")
            self.logger.info(f"  tensorboard --logdir {tensorboard_logs_dir} --port 6006")
            
            # Compile results
            training_results = {
                'experiment_name': self.experiment_name,
                'best_model_path': str(self.best_model_path),
                'save_dir': str(results.save_dir),
                'tensorboard_dir': str(self.tensorboard_dir),
                'config': train_params
            }
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def resume_training(
        self,
        checkpoint_path: Union[str, Path],
        **kwargs
    ) -> Dict:
        """
        Resume training from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            **kwargs: Additional training arguments
        
        Returns:
            Dictionary with training results
        """
        self.logger.info(f"Resuming training from: {checkpoint_path}")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.model = YOLO(str(checkpoint_path))
        
        # Resume with same config
        train_params = kwargs.copy()
        train_params['resume'] = True
        
        results = self.model.train(**train_params)
        
        return {
            'resumed_from': str(checkpoint_path),
            'save_dir': str(results.save_dir)
        }
    
    def _prepare_training_params(
        self,
        data_yaml: Union[str, Path],
        epochs: Optional[int],
        batch_size: Optional[int],
        imgsz: Optional[int],
        device: Optional[str],
        project: Optional[Union[str, Path]],
        **kwargs
    ) -> Dict:
        """Prepare training parameters from config and arguments."""
        # Use logs/tensorboard as default project directory for TensorBoard integration
        default_project = self.project_root / 'logs' / 'tensorboard'
        
        params = {
            'data': str(data_yaml),
            'epochs': epochs or self.config.get('epochs', 100),
            'batch': batch_size or self.config.get('batch', 16),
            'imgsz': imgsz or self.config.get('imgsz', 640),
            'device': device or self.config.get('device', '0'),
            'project': str(project or default_project),  # Use logs/tensorboard for TensorBoard integration
            'name': self.experiment_name,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': self.config.get('optimizer', 'AdamW'),
            'lr0': self.config.get('lr0', 0.01),
            'lrf': self.config.get('lrf', 0.01),
            'momentum': self.config.get('momentum', 0.937),
            'weight_decay': self.config.get('weight_decay', 0.0005),
            'warmup_epochs': self.config.get('warmup_epochs', 3),
            'patience': self.config.get('patience', 50),
            'save_period': self.config.get('save_period', 10),
            'workers': self.config.get('workers', 8),
            'amp': self.config.get('amp', False),  # Disabled by default to prevent auto-downloads during AMP checks
            'plots': True,
            'verbose': True
        }
        
        # Add data augmentation parameters
        aug_params = {
            'hsv_h': self.config.get('hsv_h', 0.015),
            'hsv_s': self.config.get('hsv_s', 0.7),
            'hsv_v': self.config.get('hsv_v', 0.4),
            'degrees': self.config.get('degrees', 0.0),
            'translate': self.config.get('translate', 0.1),
            'scale': self.config.get('scale', 0.5),
            'shear': self.config.get('shear', 0.0),
            'perspective': self.config.get('perspective', 0.0),
            'flipud': self.config.get('flipud', 0.0),
            'fliplr': self.config.get('fliplr', 0.5),
            'mosaic': self.config.get('mosaic', 1.0),
            'mixup': self.config.get('mixup', 0.0)
        }
        
        params.update(aug_params)
        params.update(kwargs)
        
        return params
    
    def validate(
        self,
        data_yaml: Union[str, Path],
        weights: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict:
        """
        Validate model on validation set.
        
        Args:
            data_yaml: Path to dataset YAML
            weights: Model weights path (uses best if not provided)
            **kwargs: Additional validation arguments
        
        Returns:
            Validation results
        """
        if weights:
            model = YOLO(str(weights))
        elif self.best_model_path:
            model = YOLO(str(self.best_model_path))
        else:
            raise ValueError("No weights specified and no trained model available")
        
        self.logger.info("Running validation...")
        results = model.val(data=str(data_yaml), **kwargs)
        
        return results
