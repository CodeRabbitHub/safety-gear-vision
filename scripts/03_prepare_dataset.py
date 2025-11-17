#!/usr/bin/env python3
"""
Prepare dataset by splitting into train, validation, and test sets.
"""

import sys
import argparse
from pathlib import Path
import yaml

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_splitter import DatasetSplitter
from src.utils.logger import get_logger
from src.utils.file_handler import FileHandler


def create_dataset_yaml(output_dir: Path, project_root: Path):
    """Create dataset.yaml for YOLO training."""
    dataset_config = {
        'path': str(project_root / 'data' / 'processed'),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 4,
        'names': {
            0: 'person-with-helmet-and-ppe',
            1: 'person-with-helmet-only',
            2: 'person-with-ppe-only',
            3: 'person-without-safety-gear'
        }
    }
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
    
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset splits')
    parser.add_argument(
        '--raw-image-dir',
        type=str,
        default='data/raw/images',
        help='Directory containing raw images'
    )
    parser.add_argument(
        '--raw-label-dir',
        type=str,
        default='data/raw/labels',
        help='Directory containing raw labels'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for splits'
    )
    parser.add_argument(
        '--split-ratio',
        type=float,
        nargs=3,
        default=[0.8, 0.15, 0.05],
        help='Train/Val/Test split ratios (default: 0.8 0.15 0.05)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of moving them'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    raw_image_dir = project_root / args.raw_image_dir
    raw_label_dir = project_root / args.raw_label_dir
    output_dir = project_root / args.output_dir
    
    # Initialize logger
    logger = get_logger('prepare_dataset', log_dir=project_root / 'logs')
    
    logger.info("Preparing dataset splits...")
    logger.info(f"Raw images: {raw_image_dir}")
    logger.info(f"Raw labels: {raw_label_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Split ratios: Train={args.split_ratio[0]}, "
                f"Val={args.split_ratio[1]}, Test={args.split_ratio[2]}")
    
    # Validate ratios
    if sum(args.split_ratio) != 1.0:
        logger.error(f"Split ratios must sum to 1.0, got {sum(args.split_ratio)}")
        sys.exit(1)
    
    # Create splitter
    splitter = DatasetSplitter(
        raw_image_dir=raw_image_dir,
        raw_label_dir=raw_label_dir,
        output_dir=output_dir,
        logger=logger
    )
    
    # Split dataset
    try:
        stats = splitter.split_dataset(
            train_ratio=args.split_ratio[0],
            val_ratio=args.split_ratio[1],
            test_ratio=args.split_ratio[2],
            seed=args.seed,
            copy_files=args.copy
        )
        
        # Save statistics
        stats_path = output_dir / 'split_stats.json'
        FileHandler.write_json(stats, stats_path)
        logger.info(f"Split statistics saved to: {stats_path}")
        
        # Create dataset.yaml
        yaml_path = create_dataset_yaml(output_dir, project_root)
        logger.info(f"Dataset config created: {yaml_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("DATASET SPLIT SUMMARY")
        print("="*60)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Train: {stats['train']} ({stats['train_ratio']:.1%})")
        print(f"Val:   {stats['val']} ({stats['val_ratio']:.1%})")
        print(f"Test:  {stats['test']} ({stats['test_ratio']:.1%})")
        print(f"Seed: {stats['seed']}")
        print("="*60)
        
        logger.info("✓ Dataset preparation complete!")
        print("\n✓ Dataset split successfully!")
        print(f"Dataset config: {yaml_path}")
        print("\nNext step: python scripts/05_train.py")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to prepare dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
