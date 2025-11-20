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
        from ultralytics import YOLO
        model = YOLO(str(weights_path))
        
        # Get model info (basic)
        logger.info(f"Model loaded: {weights_path.name}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Export model
    logger.info(f"Exporting to {args.format}...")
    
    try:
        export_kwargs = {
            'format': args.format,
            'imgsz': args.imgsz,
            'half': args.half,
        }
        
        # Format-specific options
        if args.format == 'onnx':
            export_kwargs['dynamic'] = args.dynamic
            export_kwargs['simplify'] = args.simplify
        
        # Export using YOLO's built-in export
        exported_path = model.export(**export_kwargs)
        
        logger.info(f"Model exported to: {exported_path}")
        
        # Move to production directory if different
        exported_path = Path(exported_path)
        model_name = weights_path.stem
        
        # Determine final path based on format
        if args.format in ['pb', 'saved_model']:
            final_path = output_dir / f"{model_name}_{args.format}"
        else:
            # Get the actual extension from exported file
            final_path = output_dir / f"{model_name}{exported_path.suffix}"
        
        # Move if not already in output directory
        if exported_path.parent != output_dir:
            import shutil
            if exported_path.is_dir():
                if final_path.exists():
                    shutil.rmtree(final_path)
                shutil.move(str(exported_path), str(final_path))
            else:
                if final_path.exists():
                    final_path.unlink()
                shutil.move(str(exported_path), str(final_path))
            
            logger.info(f"Moved to: {final_path}")
        else:
            final_path = exported_path
        
        # Print results
        print("\n" + "="*60)
        print("EXPORT COMPLETE!")
        print("="*60)
        print(f"Format: {args.format}")
        print(f"Exported model: {final_path}")
        
        # Get file size
        if final_path.is_file():
            size_mb = final_path.stat().st_size / (1024 * 1024)
            print(f"Model size: {size_mb:.2f} MB")
        elif final_path.is_dir():
            # Calculate directory size
            total_size = sum(f.stat().st_size for f in final_path.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            print(f"Model size: {size_mb:.2f} MB (directory)")
        
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
