#!/usr/bin/env python3
"""
Export trained model to different formats for deployment.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.model_utils import ModelUtils


def main():
    parser = argparse.ArgumentParser(description='Export YOLO model')
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to model weights (.pt file)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='onnx',
        choices=['onnx', 'torchscript', 'tflite', 'pb', 'saved_model', 'engine'],
        help='Export format'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/production',
        help='Output directory for exported model'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for export'
    )
    parser.add_argument(
        '--half',
        action='store_true',
        help='Use FP16 (half precision)'
    )
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='Dynamic axes for ONNX/TF export'
    )
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Simplify ONNX model'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = project_root / weights_path
    
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = get_logger('export_model', log_dir=project_root / 'logs')
    
    logger.info("="*60)
    logger.info("MODEL EXPORT")
    logger.info("="*60)
    logger.info(f"Model: {weights_path}")
    logger.info(f"Format: {args.format}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Image size: {args.imgsz}")
    
    # Validate weights
    if not weights_path.exists():
        logger.error(f"Weights file not found: {weights_path}")
        sys.exit(1)
    
    # Load model
    logger.info("Loading model...")
    try:
        model = ModelUtils.load_model(weights_path)
        
        # Get model info
        info = ModelUtils.get_model_info(model)
        logger.info(f"Model type: {info['model_type']}")
        logger.info(f"Parameters: {info['total_parameters']:,}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Export model
    logger.info(f"Exporting to {args.format}...")
    
    try:
        export_kwargs = {
            'imgsz': args.imgsz,
            'half': args.half,
        }
        
        # Format-specific options
        if args.format == 'onnx':
            export_kwargs['dynamic'] = args.dynamic
            export_kwargs['simplify'] = args.simplify
        
        # Generate output filename
        model_name = weights_path.stem
        
        exported_path = ModelUtils.export_model(
            model=model,
            export_format=args.format,
            **export_kwargs
        )
        
        # Move to output directory
        final_path = output_dir / f"{model_name}.{args.format}"
        if args.format == 'engine':
            final_path = output_dir / f"{model_name}.engine"
        elif args.format in ['pb', 'saved_model']:
            final_path = output_dir / model_name
        
        # Move exported model
        if Path(exported_path).exists():
            import shutil
            if Path(exported_path).is_dir():
                if final_path.exists():
                    shutil.rmtree(final_path)
                shutil.move(str(exported_path), str(final_path))
            else:
                shutil.move(str(exported_path), str(final_path))
        
        # Print results
        print("\n" + "="*60)
        print("EXPORT COMPLETE!")
        print("="*60)
        print(f"Format: {args.format}")
        print(f"Exported model: {final_path}")
        
        # Get file size
        if final_path.is_file():
            size_mb = ModelUtils.get_model_size(final_path)
            print(f"Model size: {size_mb:.2f} MB")
        
        print("="*60)
        
        logger.info(f"✓ Model exported successfully: {final_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
