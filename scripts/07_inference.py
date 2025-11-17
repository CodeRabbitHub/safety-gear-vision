#!/usr/bin/env python3
"""
Run inference on images using trained model.
"""

import sys
import argparse
from pathlib import Path
import json

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predictor import YOLOPredictor
from src.utils.logger import get_logger
from src.utils.file_handler import FileHandler


def main():
    parser = argparse.ArgumentParser(description='Run YOLO inference')
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to model weights (.pt file)'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to image or directory of images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/predictions',
        help='Output directory for predictions'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.45,
        help='IoU threshold for NMS'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to run on (0, cpu, etc.)'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save annotated images'
    )
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Save predictions as JSON'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = project_root / weights_path
    
    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = project_root / source_path
    
    output_dir = project_root / args.output_dir
    
    # Initialize logger
    logger = get_logger('inference', log_dir=project_root / 'logs')
    
    logger.info("="*60)
    logger.info("YOLO INFERENCE")
    logger.info("="*60)
    logger.info(f"Model: {weights_path}")
    logger.info(f"Source: {source_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Confidence threshold: {args.conf_threshold}")
    
    # Validate paths
    if not weights_path.exists():
        logger.error(f"Weights file not found: {weights_path}")
        sys.exit(1)
    
    if not source_path.exists():
        logger.error(f"Source not found: {source_path}")
        sys.exit(1)
    
    # Create predictor
    predictor = YOLOPredictor(
        weights_path=weights_path,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device,
        logger=logger
    )
    
    try:
        # Run inference
        if source_path.is_file():
            # Single image
            logger.info("Running inference on single image...")
            
            predictions = predictor.predict_image(
                image_path=source_path,
                save_results=args.save_results,
                output_dir=output_dir if args.save_results else None,
                draw_boxes=True
            )
            
            # Print results
            print("\n" + "="*60)
            print("PREDICTION RESULTS")
            print("="*60)
            print(f"Image: {predictions['image_path']}")
            print(f"Detections: {predictions['num_detections']}")
            
            for i, det in enumerate(predictions['detections'], 1):
                print(f"\n  Detection {i}:")
                print(f"    Class: {det['class_name']}")
                print(f"    Confidence: {det['confidence']:.3f}")
            
            print("="*60)
            
            if args.save_json:
                json_path = output_dir / 'prediction.json'
                FileHandler.ensure_dir(output_dir)
                FileHandler.write_json(predictions, json_path)
                logger.info(f"Predictions saved to: {json_path}")
            
        else:
            # Batch processing
            logger.info("Running inference on batch of images...")
            
            all_predictions = predictor.predict_batch(
                image_dir=source_path,
                output_dir=output_dir if args.save_results else None,
                save_results=args.save_results
            )
            
            # Get statistics
            class_counts = predictor.get_class_counts(all_predictions)
            total_detections = sum(class_counts.values())
            
            # Print summary
            print("\n" + "="*60)
            print("BATCH PREDICTION SUMMARY")
            print("="*60)
            print(f"Images processed: {len(all_predictions)}")
            print(f"Total detections: {total_detections}")
            print("\nClass distribution:")
            for class_name, count in class_counts.items():
                percentage = (count / total_detections * 100) if total_detections > 0 else 0
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
            print("="*60)
            
            if args.save_json:
                json_path = output_dir / 'predictions.json'
                FileHandler.ensure_dir(output_dir)
                FileHandler.write_json(all_predictions, json_path)
                logger.info(f"Predictions saved to: {json_path}")
        
        if args.save_results:
            logger.info(f"✓ Results saved to: {output_dir}")
        
        logger.info("✓ Inference complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
