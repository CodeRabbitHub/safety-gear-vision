#!/usr/bin/env python3
"""
Validate dataset integrity and check for errors.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_validator import DataValidator
from src.utils.logger import get_logger
from src.utils.file_handler import FileHandler


def main():
    parser = argparse.ArgumentParser(description='Validate dataset integrity')
    parser.add_argument(
        '--image-dir',
        type=str,
        default='data/raw/images',
        help='Directory containing images'
    )
    parser.add_argument(
        '--label-dir',
        type=str,
        default='data/raw/labels',
        help='Directory containing labels'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=17,
        help='Number of classes'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    image_dir = project_root / args.image_dir
    label_dir = project_root / args.label_dir
    
    # Initialize logger
    logger = get_logger('validate_data', log_dir=project_root / 'logs')
    
    logger.info("Starting data validation...")
    logger.info(f"Image directory: {image_dir}")
    logger.info(f"Label directory: {label_dir}")
    
    # Validate directories exist
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        sys.exit(1)
    
    if not label_dir.exists():
        logger.error(f"Label directory not found: {label_dir}")
        sys.exit(1)
    
    # Run validation
    validator = DataValidator(
        image_dir=image_dir,
        label_dir=label_dir,
        num_classes=args.num_classes,
        logger=logger
    )
    
    results = validator.validate()
    
    # Save results
    results_dir = project_root / 'results'
    FileHandler.ensure_dir(results_dir)
    
    report_path = results_dir / 'validation_report.json'
    FileHandler.write_json(results, report_path)
    logger.info(f"Validation report saved to: {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Valid pairs: {results['valid_pairs']}")
    print(f"Errors: {results['num_errors']}")
    print(f"Warnings: {results['num_warnings']}")
    
    if results['errors']:
        print("\nErrors:")
        for error in results['errors'][:10]:  # Show first 10
            print(f"  - {error}")
        if len(results['errors']) > 10:
            print(f"  ... and {len(results['errors']) - 10} more")
    
    if results['warnings']:
        print("\nWarnings:")
        for warning in results['warnings'][:10]:
            print(f"  - {warning}")
        if len(results['warnings']) > 10:
            print(f"  ... and {len(results['warnings']) - 10} more")
    
    print("="*60)
    
    if results['is_valid']:
        logger.info("✓ Dataset validation passed!")
        print("\n✓ Dataset is valid!")
        print("Next step: python scripts/04_analyze_dataset.py")
        return 0
    else:
        logger.error("✗ Dataset validation failed!")
        print("\n✗ Please fix errors before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())
