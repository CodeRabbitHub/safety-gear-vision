#!/usr/bin/env python3
"""
Analyze dataset and generate statistics.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_analyzer import DatasetAnalyzer
from src.utils.logger import get_logger
from src.utils.file_handler import FileHandler


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset')
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
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for analysis results'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    image_dir = project_root / args.image_dir
    label_dir = project_root / args.label_dir
    output_dir = project_root / args.output_dir
    
    # Initialize logger
    logger = get_logger('analyze_dataset', log_dir=project_root / 'logs')
    
    logger.info("Starting dataset analysis...")
    
    # Create analyzer
    analyzer = DatasetAnalyzer(
        image_dir=image_dir,
        label_dir=label_dir,
        logger=logger
    )
    
    # Run analysis
    try:
        summary = analyzer.analyze()
        
        if not summary:
            logger.error("No data to analyze")
            return 1
        
        # Save results
        FileHandler.ensure_dir(output_dir)
        
        analysis_path = output_dir / 'dataset_analysis.json'
        FileHandler.write_json(summary, analysis_path)
        logger.info(f"Analysis results saved to: {analysis_path}")
        
        # Print summary
        analyzer.print_summary(summary)
        
        logger.info("✓ Dataset analysis complete!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
