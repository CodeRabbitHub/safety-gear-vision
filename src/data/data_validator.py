"""
Data validation utility for checking dataset integrity.
"""

from pathlib import Path
from typing import List, Dict, Union, Tuple
from PIL import Image
import numpy as np
from tqdm import tqdm

from ..utils.file_handler import FileHandler
from ..utils.logger import get_logger


class DataValidator:
    """Validates YOLO format dataset integrity."""
    
    def __init__(
        self,
        image_dir: Union[str, Path],
        label_dir: Union[str, Path],
        num_classes: int = 4,
        logger=None
    ):
        """
        Initialize data validator.
        
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing labels
            num_classes: Number of classes in dataset
            logger: Logger instance
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.num_classes = num_classes
        self.logger = logger or get_logger('data_validator')
        
        self.errors = []
        self.warnings = []
    
    def validate(self) -> Dict[str, any]:
        """
        Run complete validation.
        
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Starting dataset validation...")
        
        # Get image-label pairs
        pairs = FileHandler.get_image_label_pairs(self.image_dir, self.label_dir)
        
        if not pairs:
            self.errors.append("No valid image-label pairs found")
            return self._get_results(False)
        
        self.logger.info(f"Validating {len(pairs)} image-label pairs...")
        
        valid_pairs = 0
        class_distribution = {i: 0 for i in range(self.num_classes)}
        invalid_classes = []
        
        for img_path, label_path in tqdm(pairs, desc="Validating"):
            # Validate image
            img_valid, img_error = self._validate_image(img_path)
            if not img_valid:
                self.errors.append(f"{img_path.name}: {img_error}")
                continue
            
            # Validate label
            label_valid, label_error, classes = self._validate_label(
                label_path,
                img_path
            )
            
            if not label_valid:
                self.errors.append(f"{label_path.name}: {label_error}")
                continue
            
            # Update statistics
            valid_pairs += 1
            for cls in classes:
                if 0 <= cls < self.num_classes:
                    class_distribution[cls] += 1
                else:
                    invalid_classes.append((label_path.name, cls))
        
        # Check for class imbalance
        self._check_class_imbalance(class_distribution)
        
        # Check for orphaned files
        self._check_orphaned_files()
        
        is_valid = len(self.errors) == 0
        
        self.logger.info(f"Validation complete: {valid_pairs}/{len(pairs)} valid pairs")
        if self.errors:
            self.logger.warning(f"Found {len(self.errors)} errors")
        if self.warnings:
            self.logger.warning(f"Found {len(self.warnings)} warnings")
        
        return self._get_results(is_valid, valid_pairs, class_distribution)
    
    def _validate_image(self, img_path: Path) -> Tuple[bool, str]:
        """
        Validate image file.
        
        Args:
            img_path: Path to image
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            with Image.open(img_path) as img:
                # Check if image can be loaded
                img.verify()
                
            # Re-open to check dimensions
            with Image.open(img_path) as img:
                width, height = img.size
                
                if width < 32 or height < 32:
                    return False, f"Image too small: {width}x{height}"
                
                if width > 10000 or height > 10000:
                    self.warnings.append(f"{img_path.name}: Very large image {width}x{height}")
            
            return True, ""
            
        except Exception as e:
            return False, f"Cannot open image: {str(e)}"
    
    def _validate_label(
        self,
        label_path: Path,
        img_path: Path
    ) -> Tuple[bool, str, List[int]]:
        """
        Validate YOLO format label file.
        
        Args:
            label_path: Path to label file
            img_path: Corresponding image path
        
        Returns:
            Tuple of (is_valid, error_message, class_ids)
        """
        try:
            # Get image dimensions
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            classes = []
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                self.warnings.append(f"{label_path.name}: Empty label file")
                return True, "", []
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                
                if len(parts) != 5:
                    return False, f"Line {line_num}: Expected 5 values, got {len(parts)}", []
                
                try:
                    cls = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError:
                    return False, f"Line {line_num}: Invalid numeric values", []
                
                # Validate class ID
                if cls < 0 or cls >= self.num_classes:
                    return False, f"Line {line_num}: Invalid class {cls}", []
                
                classes.append(cls)
                
                # Validate bounding box values (should be normalized 0-1)
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                    return False, f"Line {line_num}: Center coordinates out of range", []
                
                if not (0 < width <= 1 and 0 < height <= 1):
                    return False, f"Line {line_num}: Width/height out of range", []
                
                # Check if box extends beyond image bounds
                x_min = x_center - width / 2
                x_max = x_center + width / 2
                y_min = y_center - height / 2
                y_max = y_center + height / 2
                
                if x_min < 0 or x_max > 1 or y_min < 0 or y_max > 1:
                    self.warnings.append(
                        f"{label_path.name} line {line_num}: Box extends beyond image bounds"
                    )
            
            return True, "", classes
            
        except Exception as e:
            return False, f"Error reading label: {str(e)}", []
    
    def _check_class_imbalance(self, class_distribution: Dict[int, int]):
        """Check for severe class imbalance."""
        counts = list(class_distribution.values())
        
        if max(counts) == 0:
            return
        
        # Calculate imbalance ratio
        max_count = max(counts)
        min_count = min([c for c in counts if c > 0]) if any(counts) else 0
        
        if min_count == 0:
            self.warnings.append("Some classes have no samples")
            return
        
        imbalance_ratio = max_count / min_count
        
        if imbalance_ratio > 10:
            self.warnings.append(
                f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f})"
            )
    
    def _check_orphaned_files(self):
        """Check for orphaned images or labels."""
        images = set(f.stem for f in FileHandler.list_files(
            self.image_dir,
            ['.jpg', '.jpeg', '.png']
        ))
        labels = set(f.stem for f in FileHandler.list_files(
            self.label_dir,
            ['.txt']
        ))
        
        orphaned_images = images - labels
        orphaned_labels = labels - images
        
        if orphaned_images:
            self.warnings.append(
                f"Found {len(orphaned_images)} images without labels"
            )
        
        if orphaned_labels:
            self.warnings.append(
                f"Found {len(orphaned_labels)} labels without images"
            )
    
    def _get_results(
        self,
        is_valid: bool,
        valid_pairs: int = 0,
        class_distribution: Dict[int, int] = None
    ) -> Dict:
        """Compile validation results."""
        return {
            'is_valid': is_valid,
            'valid_pairs': valid_pairs,
            'errors': self.errors,
            'warnings': self.warnings,
            'class_distribution': class_distribution or {},
            'num_errors': len(self.errors),
            'num_warnings': len(self.warnings)
        }
