#!/usr/bin/env python3
"""
Evaluate trained model on test dataset.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.evaluator import ModelEvaluator
from src.utils.logger import get_logger


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO model')
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to model weights (.pt file)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/dataset.yaml',
        help='Path to dataset YAML'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['test', 'val', 'train'],
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.001,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.6,
        help='IoU threshold for mAP calculation'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = project_root / weights_path
    
    data_yaml = project_root / args.data
    
    # Setup save directory
    if args.save_dir:
        save_dir = project_root / args.save_dir
    else:
        # Auto-generate save directory
        model_name = weights_path.stem
        save_dir = project_root / 'results' / 'evaluations' / model_name
    
    # Initialize logger
    logger = get_logger('evaluate', log_dir=project_root / 'logs')
    
    logger.info("="*60)
    logger.info("MODEL EVALUATION")
    logger.info("="*60)
    logger.info(f"Model: {weights_path}")
    logger.info(f"Data: {data_yaml}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Save dir: {save_dir}")
    
    # Validate paths
    if not weights_path.exists():
        logger.error(f"Weights file not found: {weights_path}")
        sys.exit(1)
    
    if not data_yaml.exists():
        logger.error(f"Dataset config not found: {data_yaml}")
        sys.exit(1)
    
    # Create evaluator
    evaluator = ModelEvaluator(
        weights_path=weights_path,
        logger=logger
    )
    
    try:
        # Run evaluation
        metrics = evaluator.evaluate(
            data_yaml=data_yaml,
            split=args.split,
            save_dir=save_dir,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold
        )
        
        logger.info("✓ Evaluation complete!")
        logger.info(f"Results saved to: {save_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
