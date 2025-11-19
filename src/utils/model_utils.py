"""
Model utilities for loading, saving, and managing YOLO models.
"""

from pathlib import Path
from typing import Optional, Union
import torch
from ultralytics import YOLO


class ModelUtils:
    """Utilities for YOLO model operations."""
    
    @staticmethod
    def load_model(
        weights_path: Union[str, Path],
        device: Optional[str] = None
    ) -> YOLO:
        """
        Load YOLO model from weights.
        
        Args:
            weights_path: Path to model weights (.pt file)
            device: Device to load model on ('cuda', 'cpu', or None for auto)
        
        Returns:
            YOLO model instance
        """
        weights_path = Path(weights_path)
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        
        model = YOLO(str(weights_path))
        
        if device:
            model.to(device)
        
        return model
    
    @staticmethod
    def get_model_info(model: YOLO) -> dict:
        """
        Get model information.
        
        Args:
            model: YOLO model instance
        
        Returns:
            Dictionary with model info
        """
        info = {
            'model_type': model.model.__class__.__name__,
            'task': model.task,
            'device': str(next(model.model.parameters()).device),
        }
        
        # Get parameter count
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        info['total_parameters'] = total_params
        info['trainable_parameters'] = trainable_params
        
        return info
    
    @staticmethod
    def get_device(device_id: Optional[Union[int, str]] = None) -> str:
        """
        Get appropriate device for training/inference.
        
        Args:
            device_id: Device specification (0, 'cuda', 'cpu', etc.)
        
        Returns:
            Device string ('cuda:0', 'cuda:1', 'cpu', etc.)
        """
        if device_id is None:
            return 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # If integer, convert to cuda device format
        if isinstance(device_id, int):
            if torch.cuda.is_available() and device_id < torch.cuda.device_count():
                return f'cuda:{device_id}'
            else:
                return 'cpu'
        
        # If string, normalize it
        device_str = str(device_id).lower()
        
        # If it's just a number, convert to cuda format
        if device_str.isdigit():
            device_num = int(device_str)
            if torch.cuda.is_available() and device_num < torch.cuda.device_count():
                return f'cuda:{device_num}'
            else:
                return 'cpu'
        
        # If it's already 'cuda:X' or 'cpu', return as is
        if device_str.startswith('cuda:') or device_str == 'cpu':
            return device_str
        
        # If it's 'cuda', add default device number
        if device_str == 'cuda':
            return 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        return 'cpu'
    
    @staticmethod
    def check_gpu_availability() -> dict:
        """
        Check GPU availability and info.
        
        Returns:
            Dictionary with GPU information
        """
        info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': 0,
            'devices': []
        }
        
        if torch.cuda.is_available():
            info['device_count'] = torch.cuda.device_count()
            info['current_device'] = torch.cuda.current_device()
            
            for i in range(torch.cuda.device_count()):
                device_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory,
                    'memory_allocated': torch.cuda.memory_allocated(i),
                    'memory_reserved': torch.cuda.memory_reserved(i)
                }
                info['devices'].append(device_info)
        
        return info
    
    @staticmethod
    def get_model_size(weights_path: Union[str, Path]) -> float:
        """
        Get model file size in MB.
        
        Args:
            weights_path: Path to model weights
        
        Returns:
            File size in MB
        """
        weights_path = Path(weights_path)
        size_bytes = weights_path.stat().st_size
        return size_bytes / (1024 * 1024)
    
    @staticmethod
    def export_model(
        model: YOLO,
        export_format: str = 'onnx',
        output_path: Optional[Path] = None,
        **kwargs
    ) -> Path:
        """
        Export YOLO model to different format.
        
        Args:
            model: YOLO model instance
            export_format: Export format ('onnx', 'torchscript', 'tflite', etc.)
            output_path: Output file path (optional)
            **kwargs: Additional export arguments
        
        Returns:
            Path to exported model
        """
        export_path = model.export(format=export_format, **kwargs)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Path(export_path).rename(output_path)
            return output_path
        
        return Path(export_path)
