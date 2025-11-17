#!/usr/bin/env python3
"""
Reduce dataset to specific counts for testing.
Keep only 50 images in train, 10 in val, 10 in test with their labels.
"""

import sys
from pathlib import Path
import shutil
from tqdm import tqdm

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger


def reduce_dataset():
    """Reduce dataset to specified counts."""
    logger = get_logger('reduce_dataset', log_dir=project_root / 'logs')
    
    # Define paths
    processed_dir = project_root / 'data' / 'processed'
    
    splits = {
        'train': 50,
        'val': 10,
        'test': 10
    }
    
    print("="*70)
    print("DATASET REDUCER")
    print("="*70)
    print(f"Reducing dataset to:")
    for split, count in splits.items():
        print(f"  {split}: {count} images")
    print()
    
    for split, target_count in splits.items():
        img_dir = processed_dir / 'images' / split
        label_dir = processed_dir / 'labels' / split
        
        # Handle case where images/train might actually be processed_dir/train
        if not img_dir.exists():
            img_dir = processed_dir / split
        
        if not label_dir.exists():
            label_dir = processed_dir / 'labels' / split
            if not label_dir.exists():
                # Try looking in raw labels
                label_dir = project_root / 'data' / 'raw' / 'labels'
                if not label_dir.exists():
                    logger.error(f"Label directory not found for {split}")
                    continue
        
        if not img_dir.exists():
            logger.error(f"Image directory not found: {img_dir}")
            continue
        
        # Get all image files (jpg and png)
        image_files = sorted(list(img_dir.glob('*.jpg')) + 
                           list(img_dir.glob('*.jpeg')) +
                           list(img_dir.glob('*.png')) +
                           list(img_dir.glob('*.JPG')) +
                           list(img_dir.glob('*.PNG')))
        current_count = len(image_files)
        
        print(f"Processing {split}:")
        print(f"  Location: {img_dir}")
        print(f"  Current count: {current_count}")
        print(f"  Target count: {target_count}")
        
        # If we need to reduce
        if current_count > target_count:
            files_to_keep = image_files[:target_count]
            files_to_delete = image_files[target_count:]
            
            print(f"  Deleting: {len(files_to_delete)} images and their labels")
            
            # Delete extra images and labels
            for img_file in tqdm(files_to_delete, desc=f"  Removing {split}"):
                img_file.unlink()
                
                # Delete corresponding label
                label_file = label_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    label_file.unlink()
            
            print(f"  ✓ {split}: Reduced to {target_count} images")
        elif current_count < target_count:
            print(f"  ⚠ {split}: Only {current_count} images available (need {target_count})")
        else:
            print(f"  ✓ {split}: Already has {target_count} images")
        
        print()
    
    print("="*70)
    print("✓ Dataset reduction complete!")
    print("="*70)
    
    # Print summary
    print("\nFinal dataset structure:")
    for split in ['train', 'val', 'test']:
        img_dir = processed_dir / 'images' / split
        label_dir = processed_dir / 'labels' / split
        
        if img_dir.exists():
            img_count = len(list(img_dir.glob('*')))
            label_count = len(list(label_dir.glob('*.txt'))) if label_dir.exists() else 0
            print(f"  {split}: {img_count} images, {label_count} labels")


if __name__ == "__main__":
    try:
        reduce_dataset()
        print("\n✓ Success! You can now train with the reduced dataset.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
