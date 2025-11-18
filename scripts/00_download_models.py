#!/usr/bin/env python3
"""
Download pretrained YOLO models to models/pretrained directory.
"""

import sys
from pathlib import Path
import urllib.request

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

# YOLOv11 model URLs
YOLO_URLS = {
    'yolov11n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',
    'yolov11s.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt',
    'yolov11m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt',
    'yolov11l.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt',
    'yolov11x.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt',
}


def download_model(model_name: str, destination_dir: Path):
    """Download a YOLO model directly to destination directory."""
    if model_name not in YOLO_URLS:
        print(f"✗ Unknown model: {model_name}")
        return False
    
    url = YOLO_URLS[model_name]
    dest_file = destination_dir / model_name
    
    # Check if already downloaded
    if dest_file.exists():
        size_mb = dest_file.stat().st_size / (1024 * 1024)
        print(f"✓ Already downloaded: {model_name} ({size_mb:.1f} MB)")
        return True
    
    print(f"\nDownloading {model_name}...")
    print(f"From: {url}")
    
    try:
        # Define progress callback
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(downloaded * 100 / total_size, 100)
                print(f"  Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", end='\r')
        
        urllib.request.urlretrieve(url, dest_file, reporthook=download_progress)
        size_mb = dest_file.stat().st_size / (1024 * 1024)
        print(f"\n✓ Downloaded: {model_name} ({size_mb:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to download {model_name}: {e}")
        # Clean up partial download
        if dest_file.exists():
            dest_file.unlink()
        return False


def main():
    logger = get_logger('download_models', log_dir=project_root / 'logs')
    
    # Download to project models/pretrained directory
    models_dir = project_root / 'models' / 'pretrained'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("YOLO MODEL DOWNLOADER")
    print("="*70)
    print(f"Models will be stored in: {models_dir}\n")
    
    # Download YOLOv11 models (latest)
    yolov11_models = [
        'yolov11n.pt',   # Nano - fastest, lowest accuracy
        'yolov11s.pt',   # Small - balanced
        'yolov11m.pt',   # Medium
        'yolov11l.pt',   # Large
        'yolov11x.pt',   # Extra Large - slowest, highest accuracy
    ]
    
    print("Available YOLOv11 models:")
    for i, model in enumerate(yolov11_models, 1):
        print(f"  {i}. {model}")
    
    print("\n" + "-"*70)
    print("Downloading YOLOv11 models (this may take several minutes)...")
    print("-"*70)
    
    success_count = 0
    total_count = len(yolov11_models)
    
    for model in yolov11_models:
        if download_model(model, models_dir):
            success_count += 1
    
    print("\n" + "="*70)
    print(f"DOWNLOAD SUMMARY: {success_count}/{total_count} models downloaded successfully")
    print("="*70)
    print(f"\nModels stored at: {models_dir}")
    
    if success_count == total_count:
        print("\n✓ All models downloaded successfully!")
        print("You can now run: poetry run python scripts/05_train.py")
        return 0
    else:
        print(f"\n⚠ {total_count - success_count} model(s) failed to download")
        print("Check your internet connection and try again")
        return 1


if __name__ == "__main__":
    sys.exit(main())
