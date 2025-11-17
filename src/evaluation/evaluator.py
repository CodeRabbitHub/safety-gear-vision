"""
Model evaluation module for computing metrics and generating reports.
"""

from pathlib import Path
from typing import Union, Dict, Optional
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from ..utils.logger import get_logger
from ..utils.model_utils import ModelUtils
from ..utils.file_handler import FileHandler


class ModelEvaluator:
    """Evaluates YOLO model performance on test data."""
    
    CLASS_NAMES = {
        0: "helmet-and-ppe",
        1: "helmet-only",
        2: "ppe-only",
        3: "no-safety-gear"
    }
    
    def __init__(
        self,
        weights_path: Union[str, Path],
        logger=None
    ):
        """
        Initialize model evaluator.
        
        Args:
            weights_path: Path to model weights
            logger: Logger instance
        """
        self.weights_path = Path(weights_path)
        self.logger = logger or get_logger('model_evaluator')
        
        # Load model
        self.logger.info(f"Loading model: {self.weights_path}")
        self.model = ModelUtils.load_model(self.weights_path)
    
    def evaluate(
        self,
        data_yaml: Union[str, Path],
        split: str = 'test',
        save_dir: Optional[Path] = None,
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.6
    ) -> Dict:
        """
        Evaluate model on dataset.
        
        Args:
            data_yaml: Path to dataset YAML
            split: Dataset split to evaluate ('test', 'val')
            save_dir: Directory to save results
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for mAP calculation
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Evaluating model on {split} set...")
        
        # Run validation
        results = self.model.val(
            data=str(data_yaml),
            split=split,
            conf=conf_threshold,
            iou=iou_threshold,
            save_json=True,
            plots=True
        )
        
        # Extract metrics
        metrics = self._extract_metrics(results)
        
        # Save results
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self._save_results(metrics, save_dir)
            self.logger.info(f"Results saved to: {save_dir}")
        
        # Print summary
        self._print_summary(metrics)
        
        return metrics
    
    def _extract_metrics(self, results) -> Dict:
        """
        Extract metrics from YOLO results.
        
        Args:
            results: YOLO validation results
        
        Returns:
            Dictionary with metrics
        """
        metrics = {
            'overall': {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1_score': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-10)
            },
            'per_class': {}
        }
        
        # Per-class metrics
        if hasattr(results.box, 'maps') and results.box.maps is not None:
            for i, class_map in enumerate(results.box.maps):
                class_name = self.CLASS_NAMES.get(i, f"class_{i}")
                metrics['per_class'][class_name] = {
                    'mAP50-95': float(class_map)
                }
        
        # Add per-class precision and recall if available
        if hasattr(results.box, 'p') and results.box.p is not None:
            for i, (p, r) in enumerate(zip(results.box.p, results.box.r)):
                class_name = self.CLASS_NAMES.get(i, f"class_{i}")
                if class_name not in metrics['per_class']:
                    metrics['per_class'][class_name] = {}
                metrics['per_class'][class_name]['precision'] = float(p)
                metrics['per_class'][class_name]['recall'] = float(r)
                metrics['per_class'][class_name]['f1_score'] = 2 * (p * r) / (p + r + 1e-10)
        
        return metrics
    
    def _save_results(self, metrics: Dict, save_dir: Path):
        """
        Save evaluation results.
        
        Args:
            metrics: Metrics dictionary
            save_dir: Directory to save results
        """
        # Save metrics as JSON
        metrics_path = save_dir / 'metrics.json'
        FileHandler.write_json(metrics, metrics_path)
        
        # Save as readable text
        report_path = save_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("Overall Metrics:\n")
            for metric, value in metrics['overall'].items():
                f.write(f"  {metric}: {value:.4f}\n")
            
            f.write("\nPer-Class Metrics:\n")
            for class_name, class_metrics in metrics['per_class'].items():
                f.write(f"\n  {class_name}:\n")
                for metric, value in class_metrics.items():
                    f.write(f"    {metric}: {value:.4f}\n")
        
        self.logger.info(f"Metrics saved to: {metrics_path}")
        self.logger.info(f"Report saved to: {report_path}")
    
    def _print_summary(self, metrics: Dict):
        """
        Print evaluation summary.
        
        Args:
            metrics: Metrics dictionary
        """
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        print("\nOverall Metrics:")
        for metric, value in metrics['overall'].items():
            print(f"  {metric:15s}: {value:.4f}")
        
        print("\nPer-Class Metrics:")
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"\n  {class_name}:")
            for metric, value in class_metrics.items():
                print(f"    {metric:15s}: {value:.4f}")
        
        print("="*60 + "\n")
    
    def compare_models(
        self,
        model_weights: Dict[str, Path],
        data_yaml: Union[str, Path],
        save_dir: Optional[Path] = None
    ) -> Dict:
        """
        Compare multiple models.
        
        Args:
            model_weights: Dictionary of {model_name: weights_path}
            data_yaml: Path to dataset YAML
            save_dir: Directory to save comparison
        
        Returns:
            Comparison results
        """
        self.logger.info(f"Comparing {len(model_weights)} models...")
        
        comparison = {}
        
        for model_name, weights_path in model_weights.items():
            self.logger.info(f"Evaluating {model_name}...")
            
            # Load and evaluate model
            model = ModelUtils.load_model(weights_path)
            results = model.val(data=str(data_yaml))
            
            comparison[model_name] = {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr)
            }
        
        # Save comparison
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            comp_path = save_dir / 'model_comparison.json'
            FileHandler.write_json(comparison, comp_path)
            
            # Create comparison plot
            self._plot_comparison(comparison, save_dir)
        
        return comparison
    
    def _plot_comparison(self, comparison: Dict, save_dir: Path):
        """
        Create comparison visualization.
        
        Args:
            comparison: Comparison results
            save_dir: Directory to save plot
        """
        metrics = ['mAP50', 'mAP50-95', 'precision', 'recall']
        models = list(comparison.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            values = [comparison[model][metric] for model in models]
            
            axes[idx].bar(models, values, color='skyblue', edgecolor='navy')
            axes[idx].set_title(metric.upper(), fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Score')
            axes[idx].set_ylim(0, 1)
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comparison plot saved to: {save_dir / 'model_comparison.png'}")
