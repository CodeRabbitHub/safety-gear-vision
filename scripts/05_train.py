#!/usr/bin/env python3
"""
Train YOLOv11 model on safety gear detection dataset.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.trainer import YOLOTrainer
from src.utils.logger import get_logger
from src.utils.model_utils import ModelUtils


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv11 model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/training/yolov11s.yaml',
        help='Path to training config file'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/dataset.yaml',
        help='Path to dataset YAML'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name (auto-generated if not provided)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=None,
        help='Image size (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to train on (0, cpu, etc.)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='models/checkpoints',
        help='Project directory for saving results'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    config_path = project_root / args.config
    data_yaml = project_root / args.data
    project_dir = project_root / args.project
    
    # Initialize logger
    logger = get_logger('train', log_dir=project_root / 'logs')
    
    logger.info("="*60)
    logger.info("YOLOV11 TRAINING - SAFETY GEAR DETECTION")
    logger.info("="*60)
    
    # Check GPU availability
    gpu_info = ModelUtils.check_gpu_availability()
    logger.info(f"CUDA available: {gpu_info['cuda_available']}")
    
    if gpu_info['cuda_available']:
        for device in gpu_info['devices']:
            logger.info(f"GPU {device['id']}: {device['name']}")
            logger.info(f"  Memory: {device['memory_total'] / 1e9:.2f} GB")
    else:
        logger.warning("No GPU detected! Training will be slow on CPU")
    
    # Validate config and data files
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    if not data_yaml.exists():
        logger.error(f"Dataset config not found: {data_yaml}")
        logger.error("Run: python scripts/03_prepare_dataset.py first")
        sys.exit(1)
    
    # Initialize trainer
    logger.info(f"Loading config: {config_path}")
    trainer = YOLOTrainer(
        config_path=config_path,
        experiment_name=args.experiment_name,
        logger=logger
    )
    
    try:
        if args.resume:
            # Resume training
            logger.info(f"Resuming training from: {args.resume}")
            results = trainer.resume_training(
                checkpoint_path=args.resume,
                data=str(data_yaml)
            )
        else:
            # Start new training
            logger.info("Starting training...")
            results = trainer.train(
                data_yaml=data_yaml,
                epochs=args.epochs,
                batch_size=args.batch_size,
                imgsz=args.imgsz,
                device=args.device,
                project=project_dir
            )
        
        # Print results
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Experiment: {results.get('experiment_name', 'N/A')}")
        print(f"Best model: {results.get('best_model_path', 'N/A')}")
        print(f"Results: {results.get('save_dir', 'N/A')}")
        print("="*60)
        
        logger.info("✓ Training completed successfully!")
        print("\nNext steps:")
        print("1. Evaluate: python scripts/06_evaluate.py --weights <path_to_best.pt>")
        print("2. Inference: python scripts/07_inference.py --weights <path_to_best.pt>")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
