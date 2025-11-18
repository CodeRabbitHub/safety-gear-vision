#!/usr/bin/env python3
"""
Launch TensorBoard to visualize training metrics.
Automatically converts training results CSV to TensorBoard format if needed.
"""

import sys
import subprocess
from pathlib import Path
import csv

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("Error: torch not installed. Install with: poetry add torch")
    sys.exit(1)


def convert_csv_to_tensorboard(csv_path, output_dir):
    """Convert YOLO results CSV to TensorBoard event files."""
    logger = get_logger('tensorboard')
    
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        logger.warning(f"CSV file not found: {csv_path}")
        return False
    
    # Check if event files already exist
    event_files = list(output_dir.glob("events.out.tfevents.*"))
    if event_files:
        logger.info(f"TensorBoard event files already exist in {output_dir}")
        return True
    
    logger.info(f"Converting {csv_path} to TensorBoard format...")
    
    try:
        writer = SummaryWriter(str(output_dir))
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            step = 0
            
            for row in reader:
                try:
                    step = int(float(row.get('epoch', step)))
                except (ValueError, TypeError):
                    continue
                
                # Add training metrics
                for key in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']:
                    if key in row and row[key]:
                        try:
                            value = float(row[key])
                            writer.add_scalar(key, value, step)
                        except (ValueError, TypeError):
                            pass
                
                # Add validation metrics
                for key in ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']:
                    if key in row and row[key]:
                        try:
                            value = float(row[key])
                            writer.add_scalar(key, value, step)
                        except (ValueError, TypeError):
                            pass
                
                # Add performance metrics
                metrics_map = {
                    'metrics/precision(B)': 'metrics/precision',
                    'metrics/recall(B)': 'metrics/recall',
                    'metrics/mAP50(B)': 'metrics/mAP50',
                    'metrics/mAP50-95(B)': 'metrics/mAP50-95'
                }
                
                for csv_key, tb_key in metrics_map.items():
                    if csv_key in row and row[csv_key]:
                        try:
                            value = float(row[csv_key])
                            writer.add_scalar(tb_key, value, step)
                        except (ValueError, TypeError):
                            pass
        
        writer.flush()
        writer.close()
        logger.info(f"Successfully converted {step} epochs to TensorBoard format")
        return True
        
    except Exception as e:
        logger.error(f"Error converting CSV: {e}")
        return False


def main():
    """Launch TensorBoard server."""
    logger = get_logger('tensorboard', log_dir=project_root / 'logs')
    
    # Look for YOLO training runs in multiple possible locations
    possible_dirs = [
        project_root / "runs" / "detect",
        project_root / "models" / "checkpoints",
        project_root / "logs" / "tensorboard"
    ]
    
    tensorboard_dir = None
    
    # Find the latest experiment directory
    for base_dir in possible_dirs:
        if not base_dir.exists():
            continue
        
        exp_dirs = sorted(base_dir.glob("exp_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if exp_dirs:
            tensorboard_dir = exp_dirs[0]
            break
    
    if not tensorboard_dir:
        logger.error(f"No training runs found in: {', '.join(str(d) for d in possible_dirs)}")
        logger.info("Please run training first: poetry run python scripts/05_train.py")
        print("\n✗ No training logs found")
        return 1
    
    # Try to convert CSV to TensorBoard format if needed
    csv_path = tensorboard_dir / "results.csv"
    if csv_path.exists():
        convert_csv_to_tensorboard(csv_path, tensorboard_dir)
    
    # Check if there are any event files
    event_files = list(tensorboard_dir.glob("events.out.tfevents.*"))
    if not event_files:
        logger.warning(f"No TensorBoard event files found in: {tensorboard_dir}")
        logger.info("Training may still be in progress or not yet started")
    
    print("\n" + "="*70)
    print("TENSORBOARD LAUNCHER")
    print("="*70)
    print(f"Latest training run: {tensorboard_dir.name}")
    print(f"Logs directory: {tensorboard_dir}")
    print("\nStarting TensorBoard server...")
    print("Open your browser at: http://localhost:6006")
    print("Press Ctrl+C to stop TensorBoard")
    print("="*70 + "\n")
    
    logger.info(f"Launching TensorBoard with logs from: {tensorboard_dir}")
    
    try:
        # Launch TensorBoard
        subprocess.run([
            "tensorboard",
            "--logdir", str(tensorboard_dir),
            "--port", "6006",
            "--host", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n\nTensorBoard stopped by user")
        logger.info("TensorBoard stopped")
        return 0
    except FileNotFoundError:
        logger.error("TensorBoard not found. Install with: poetry add tensorboard")
        print("\n✗ TensorBoard is not installed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
