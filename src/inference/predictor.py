"""
Inference module for running predictions with trained YOLO models.
"""

from pathlib import Path
from typing import Union, List, Optional, Dict
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

from ..utils.logger import get_logger
from ..utils.file_handler import FileHandler
from ..utils.model_utils import ModelUtils


class YOLOPredictor:
    """Handles inference with trained YOLO models."""
    
    # Class names and colors
    CLASS_NAMES = {
        0: "person-with-helmet-and-ppe",
        1: "person-with-helmet-only",
        2: "person-with-ppe-only",
        3: "person-without-safety-gear"
    }
    
    CLASS_COLORS = {
        0: (0, 255, 0),      # Green - Safe
        1: (0, 165, 255),    # Orange - Partial
        2: (0, 165, 255),    # Orange - Partial
        3: (0, 0, 255)       # Red - Unsafe
    }
    
    def __init__(
        self,
        weights_path: Union[str, Path],
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        logger=None
    ):
        """
        Initialize YOLO predictor.
        
        Args:
            weights_path: Path to model weights
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on
            logger: Logger instance
        """
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.logger = logger or get_logger('yolo_predictor')
        
        # Load model
        self.logger.info(f"Loading model from: {self.weights_path}")
        self.device = ModelUtils.get_device(device)
        self.model = ModelUtils.load_model(self.weights_path, self.device)
        
        # Log model info
        model_info = ModelUtils.get_model_info(self.model)
        self.logger.info(f"Model loaded on device: {model_info['device']}")
    
    def predict_image(
        self,
        image_path: Union[str, Path],
        save_results: bool = False,
        output_dir: Optional[Path] = None,
        draw_boxes: bool = True
    ) -> Dict:
        """
        Run prediction on single image.
        
        Args:
            image_path: Path to input image
            save_results: Whether to save results
            output_dir: Directory to save results
            draw_boxes: Whether to draw bounding boxes
        
        Returns:
            Dictionary with prediction results
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Run inference
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        # Parse results
        predictions = self._parse_results(results)
        
        # Draw boxes if requested
        if draw_boxes:
            img = cv2.imread(str(image_path))
            img = self._draw_predictions(img, predictions)
            
            if save_results and output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"pred_{image_path.name}"
                cv2.imwrite(str(output_path), img)
                predictions['output_path'] = str(output_path)
        
        predictions['image_path'] = str(image_path)
        
        return predictions
    
    def predict_batch(
        self,
        image_dir: Union[str, Path],
        output_dir: Optional[Path] = None,
        save_results: bool = True,
        image_extensions: List[str] = ['.jpg', '.jpeg', '.png']
    ) -> List[Dict]:
        """
        Run predictions on batch of images.
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save results
            save_results: Whether to save annotated images
            image_extensions: Valid image file extensions
        
        Returns:
            List of prediction dictionaries
        """
        image_dir = Path(image_dir)
        
        # Get all images
        images = FileHandler.list_files(image_dir, image_extensions)
        
        if not images:
            self.logger.warning(f"No images found in {image_dir}")
            return []
        
        self.logger.info(f"Running inference on {len(images)} images...")
        
        all_predictions = []
        
        for img_path in tqdm(images, desc="Processing images"):
            try:
                predictions = self.predict_image(
                    img_path,
                    save_results=save_results,
                    output_dir=output_dir
                )
                all_predictions.append(predictions)
                
            except Exception as e:
                self.logger.error(f"Error processing {img_path.name}: {e}")
        
        self.logger.info(f"Processed {len(all_predictions)} images")
        
        return all_predictions
    
    def _parse_results(self, results) -> Dict:
        """
        Parse YOLO results into structured format.
        
        Args:
            results: YOLO results object
        
        Returns:
            Dictionary with parsed predictions
        """
        boxes = results.boxes
        
        predictions = {
            'num_detections': len(boxes),
            'detections': []
        }
        
        for box in boxes:
            detection = {
                'class_id': int(box.cls.item()),
                'class_name': self.CLASS_NAMES.get(int(box.cls.item()), 'Unknown'),
                'confidence': float(box.conf.item()),
                'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                'bbox_norm': box.xywhn[0].cpu().numpy().tolist()  # [x_center, y_center, w, h] normalized
            }
            
            predictions['detections'].append(detection)
        
        # Sort by confidence
        predictions['detections'].sort(key=lambda x: x['confidence'], reverse=True)
        
        return predictions
    
    def _draw_predictions(
        self,
        image: np.ndarray,
        predictions: Dict,
        line_thickness: int = 2
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: Input image (BGR)
            predictions: Predictions dictionary
            line_thickness: Box line thickness
        
        Returns:
            Annotated image
        """
        img = image.copy()
        
        for det in predictions['detections']:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, det['bbox'])
            
            # Get class info
            class_id = det['class_id']
            class_name = det['class_name']
            confidence = det['confidence']
            color = self.CLASS_COLORS.get(class_id, (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
            
            # Create label
            label = f"{class_name}: {confidence:.2f}"
            
            # Get label size
            (label_width, label_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )
            
            # Draw label background
            cv2.rectangle(
                img,
                (x1, y1 - label_height - baseline - 5),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                img,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return img
    
    def get_class_counts(self, predictions: List[Dict]) -> Dict[str, int]:
        """
        Get class counts from batch predictions.
        
        Args:
            predictions: List of prediction dictionaries
        
        Returns:
            Dictionary with class counts
        """
        class_counts = {name: 0 for name in self.CLASS_NAMES.values()}
        
        for pred in predictions:
            for det in pred['detections']:
                class_name = det['class_name']
                class_counts[class_name] += 1
        
        return class_counts
