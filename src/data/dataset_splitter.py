"""
Dataset splitting utility for creating train/val/test splits.
"""

import random
from pathlib import Path
from typing import List, Tuple, Union
import shutil
from tqdm import tqdm

from ..utils.file_handler import FileHandler
from ..utils.logger import get_logger


class DatasetSplitter:
    """Handles splitting dataset into train, validation, and test sets."""
    
    def __init__(
        self,
        raw_image_dir: Union[str, Path],
        raw_label_dir: Union[str, Path],
        output_dir: Union[str, Path],
        logger=None
    ):
        """
        Initialize dataset splitter.
        
        Args:
            raw_image_dir: Directory containing all images
            raw_label_dir: Directory containing all labels
            output_dir: Output directory for splits
            logger: Logger instance
        """
        self.raw_image_dir = Path(raw_image_dir)
        self.raw_label_dir = Path(raw_label_dir)
        self.output_dir = Path(output_dir)
        self.logger = logger or get_logger('dataset_splitter')
        
        # Validate input directories
        if not self.raw_image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.raw_image_dir}")
        if not self.raw_label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.raw_label_dir}")
    
    def split_dataset(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
        seed: int = 42,
        copy_files: bool = True
    ) -> dict:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            seed: Random seed for reproducibility
            copy_files: Whether to copy files (True) or move them (False)
        
        Returns:
            Dictionary with split statistics
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Get image-label pairs
        self.logger.info("Finding image-label pairs...")
        pairs = FileHandler.get_image_label_pairs(
            self.raw_image_dir,
            self.raw_label_dir
        )
        
        if not pairs:
            raise ValueError("No valid image-label pairs found")
        
        self.logger.info(f"Found {len(pairs)} image-label pairs")
        
        # Shuffle pairs
        random.seed(seed)
        random.shuffle(pairs)
        
        # Calculate split indices
        total = len(pairs)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_pairs = pairs[:train_end]
        val_pairs = pairs[train_end:val_end]
        test_pairs = pairs[val_end:]
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            FileHandler.ensure_dir(self.output_dir / 'images' / split)
            FileHandler.ensure_dir(self.output_dir / 'labels' / split)
        
        # Copy/move files
        self.logger.info("Organizing files into splits...")
        self._organize_split(train_pairs, 'train', copy_files)
        self._organize_split(val_pairs, 'val', copy_files)
        self._organize_split(test_pairs, 'test', copy_files)
        
        # Generate statistics
        stats = {
            'total_samples': total,
            'train': len(train_pairs),
            'val': len(val_pairs),
            'test': len(test_pairs),
            'train_ratio': len(train_pairs) / total,
            'val_ratio': len(val_pairs) / total,
            'test_ratio': len(test_pairs) / total,
            'seed': seed
        }
        
        self.logger.info(f"Dataset split complete: Train={stats['train']}, "
                        f"Val={stats['val']}, Test={stats['test']}")
        
        return stats
    
    def _organize_split(
        self,
        pairs: List[Tuple[Path, Path]],
        split_name: str,
        copy_files: bool
    ):
        """
        Organize files for a specific split.
        
        Args:
            pairs: List of (image_path, label_path) tuples
            split_name: Name of the split ('train', 'val', or 'test')
            copy_files: Whether to copy or move files
        """
        operation = FileHandler.copy_file if copy_files else FileHandler.move_file
        
        for img_path, label_path in tqdm(pairs, desc=f"Processing {split_name} split"):
            # Copy/move image
            dst_img = self.output_dir / 'images' / split_name / img_path.name
            operation(img_path, dst_img)
            
            # Copy/move label
            dst_label = self.output_dir / 'labels' / split_name / label_path.name
            operation(label_path, dst_label)
    
    def get_split_stats(self) -> dict:
        """
        Get statistics for existing splits.
        
        Returns:
            Dictionary with split statistics
        """
        stats = {}
        
        for split in ['train', 'val', 'test']:
            image_dir = self.output_dir / 'images' / split
            label_dir = self.output_dir / 'labels' / split
            
            if image_dir.exists():
                images = FileHandler.list_files(image_dir, ['.jpg', '.jpeg', '.png'])
                labels = FileHandler.list_files(label_dir, ['.txt'])
                
                stats[split] = {
                    'images': len(images),
                    'labels': len(labels),
                    'matched': len(FileHandler.get_image_label_pairs(image_dir, label_dir))
                }
        
        return stats
