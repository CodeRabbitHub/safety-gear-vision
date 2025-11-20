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
        help='Path to image, video, or directory of images'
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
    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Save individual video frames (video only)'
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
        # Determine source type
        is_video = source_path.is_file() and source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        # Run inference
        if is_video:
            # Video processing
            logger.info("Running inference on video...")
            
            output_video_path = None
            if args.save_results:
                output_video_path = output_dir / f"pred_{source_path.name}"
            
            frames_dir = None
            if args.save_frames:
                frames_dir = output_dir / 'frames' / source_path.stem
            
            results = predictor.predict_video(
                video_path=source_path,
                output_path=output_video_path,
                save_frames=args.save_frames,
                frames_dir=frames_dir,
                show_progress=True
            )
            
            # Print results
            print("\n" + "="*60)
            print("VIDEO PREDICTION RESULTS")
            print("="*60)
            print(f"Video: {results['video_path']}")
            print(f"Total frames: {results['total_frames']}")
            print(f"FPS: {results['fps']}")
            print(f"Resolution: {results['resolution'][0]}x{results['resolution'][1]}")
            print(f"Total detections: {results['total_detections']}")
            
            print("\nClass distribution:")
            for class_name, count in results['class_counts'].items():
                if count > 0:
                    print(f"  {class_name}: {count}")
            
            if output_video_path:
                print(f"\nOutput video: {output_video_path}")
            
            print("="*60)
            
            if args.save_json:
                json_path = output_dir / f"{source_path.stem}_predictions.json"
                FileHandler.ensure_dir(output_dir)
                # Save without frame-by-frame predictions to keep file size manageable
                summary = {
                    'video_path': results['video_path'],
                    'total_frames': results['total_frames'],
                    'fps': results['fps'],
                    'resolution': results['resolution'],
                    'total_detections': results['total_detections'],
                    'class_counts': results['class_counts']
                }
                FileHandler.write_json(summary, json_path)
                logger.info(f"Summary saved to: {json_path}")
        
        elif source_path.is_file():
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
