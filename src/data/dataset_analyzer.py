"""
Dataset analysis utility for exploratory data analysis.
"""

from pathlib import Path
from typing import Dict, Union, List
from collections import Counter
from PIL import Image
import numpy as np
from tqdm import tqdm

from ..utils.file_handler import FileHandler
from ..utils.logger import get_logger


class DatasetAnalyzer:
    """Analyzes dataset and generates statistics."""
    
    # Class names mapping
    CLASS_NAMES = {
        0: "person-with-helmet-and-ppe",
        1: "person-with-helmet-only",
        2: "person-with-ppe-only",
        3: "person-without-safety-gear"
    }
    
    def __init__(
        self,
        image_dir: Union[str, Path],
        label_dir: Union[str, Path],
        logger=None
    ):
        """
        Initialize dataset analyzer.
        
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing labels
            logger: Logger instance
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.logger = logger or get_logger('dataset_analyzer')
    
    def analyze(self) -> Dict:
        """
        Perform complete dataset analysis.
        
        Returns:
            Dictionary with analysis results
        """
        self.logger.info("Starting dataset analysis...")
        
        # Get image-label pairs
        pairs = FileHandler.get_image_label_pairs(self.image_dir, self.label_dir)
        
        if not pairs:
            self.logger.warning("No valid image-label pairs found")
            return {}
        
        self.logger.info(f"Analyzing {len(pairs)} samples...")
        
        # Initialize statistics
        stats = {
            'total_samples': len(pairs),
            'class_distribution': Counter(),
            'objects_per_image': [],
            'image_dimensions': [],
            'bbox_areas': [],
            'bbox_aspect_ratios': []
        }
        
        # Analyze each sample
        for img_path, label_path in tqdm(pairs, desc="Analyzing"):
            # Analyze image
            img_stats = self._analyze_image(img_path)
            stats['image_dimensions'].append(img_stats['dimensions'])
            
            # Analyze labels
            label_stats = self._analyze_label(label_path, img_stats['dimensions'])
            stats['class_distribution'].update(label_stats['classes'])
            stats['objects_per_image'].append(label_stats['num_objects'])
            stats['bbox_areas'].extend(label_stats['bbox_areas'])
            stats['bbox_aspect_ratios'].extend(label_stats['aspect_ratios'])
        
        # Calculate summary statistics
        summary = self._calculate_summary(stats)
        
        self.logger.info("Analysis complete")
        
        return summary
    
    def _analyze_image(self, img_path: Path) -> Dict:
        """
        Analyze single image.
        
        Args:
            img_path: Path to image
        
        Returns:
            Dictionary with image statistics
        """
        with Image.open(img_path) as img:
            width, height = img.size
            
        return {
            'dimensions': (width, height),
            'aspect_ratio': width / height if height > 0 else 0
        }
    
    def _analyze_label(self, label_path: Path, img_dims: tuple) -> Dict:
        """
        Analyze single label file.
        
        Args:
            label_path: Path to label file
            img_dims: Image dimensions (width, height)
        
        Returns:
            Dictionary with label statistics
        """
        classes = []
        bbox_areas = []
        aspect_ratios = []
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                
                if len(parts) != 5:
                    continue
                
                cls = int(parts[0])
                width = float(parts[3])
                height = float(parts[4])
                
                classes.append(cls)
                bbox_areas.append(width * height)  # Normalized area
                
                if height > 0:
                    aspect_ratios.append(width / height)
        
        except Exception as e:
            self.logger.warning(f"Error analyzing {label_path.name}: {e}")
        
        return {
            'classes': classes,
            'num_objects': len(classes),
            'bbox_areas': bbox_areas,
            'aspect_ratios': aspect_ratios
        }
    
    def _calculate_summary(self, stats: Dict) -> Dict:
        """
        Calculate summary statistics.
        
        Args:
            stats: Raw statistics dictionary
        
        Returns:
            Summary statistics dictionary
        """
        # Class distribution
        class_dist = dict(stats['class_distribution'])
        total_objects = sum(class_dist.values())
        
        class_summary = {}
        for cls_id, count in class_dist.items():
            class_summary[cls_id] = {
                'class_name': self.CLASS_NAMES.get(cls_id, f"Unknown-{cls_id}"),
                'count': count,
                'percentage': (count / total_objects * 100) if total_objects > 0 else 0
            }
        
        # Image dimensions
        widths = [d[0] for d in stats['image_dimensions']]
        heights = [d[1] for d in stats['image_dimensions']]
        
        # Objects per image
        obj_counts = stats['objects_per_image']
        
        # Bounding box statistics
        bbox_areas = stats['bbox_areas']
        bbox_ratios = stats['bbox_aspect_ratios']
        
        summary = {
            'dataset_info': {
                'total_samples': stats['total_samples'],
                'total_objects': total_objects,
                'avg_objects_per_image': np.mean(obj_counts) if obj_counts else 0
            },
            'class_distribution': class_summary,
            'image_statistics': {
                'width': {
                    'min': min(widths) if widths else 0,
                    'max': max(widths) if widths else 0,
                    'mean': np.mean(widths) if widths else 0,
                    'std': np.std(widths) if widths else 0
                },
                'height': {
                    'min': min(heights) if heights else 0,
                    'max': max(heights) if heights else 0,
                    'mean': np.mean(heights) if heights else 0,
                    'std': np.std(heights) if heights else 0
                }
            },
            'bbox_statistics': {
                'area': {
                    'min': min(bbox_areas) if bbox_areas else 0,
                    'max': max(bbox_areas) if bbox_areas else 0,
                    'mean': np.mean(bbox_areas) if bbox_areas else 0,
                    'median': np.median(bbox_areas) if bbox_areas else 0
                },
                'aspect_ratio': {
                    'min': min(bbox_ratios) if bbox_ratios else 0,
                    'max': max(bbox_ratios) if bbox_ratios else 0,
                    'mean': np.mean(bbox_ratios) if bbox_ratios else 0
                }
            }
        }
        
        return summary
    
    def print_summary(self, summary: Dict):
        """
        Print formatted summary.
        
        Args:
            summary: Summary dictionary from analyze()
        """
        print("\n" + "="*60)
        print("DATASET ANALYSIS SUMMARY")
        print("="*60)
        
        # Dataset info
        info = summary['dataset_info']
        print(f"\nDataset Information:")
        print(f"  Total Samples: {info['total_samples']}")
        print(f"  Total Objects: {info['total_objects']}")
        print(f"  Avg Objects/Image: {info['avg_objects_per_image']:.2f}")
        
        # Class distribution
        print(f"\nClass Distribution:")
        for cls_id, data in sorted(summary['class_distribution'].items()):
            print(f"  {cls_id}: {data['class_name']}")
            print(f"      Count: {data['count']} ({data['percentage']:.1f}%)")
        
        # Image statistics
        img_stats = summary['image_statistics']
        print(f"\nImage Dimensions:")
        print(f"  Width:  {img_stats['width']['min']}-{img_stats['width']['max']} "
              f"(mean: {img_stats['width']['mean']:.0f})")
        print(f"  Height: {img_stats['height']['min']}-{img_stats['height']['max']} "
              f"(mean: {img_stats['height']['mean']:.0f})")
        
        print("="*60 + "\n")
