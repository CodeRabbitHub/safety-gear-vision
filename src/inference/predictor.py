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
    
    # 17 safety gear classes from dataset.yaml
    CLASS_NAMES = {
        0: "Person",
        1: "Head",
        2: "Face",
        3: "Glasses",
        4: "Face-Mask-Medical",
        5: "Face-Shield",
        6: "Ear",
        7: "Earmuffs",
        8: "Hands",
        9: "Gloves",
        10: "Foot",
        11: "Shoes",
        12: "Safety-Vest",
        13: "Tools",
        14: "Helmet",
        15: "Medical-Suit",
        16: "Safety-Suit"
    }
    
    # Color mapping for visualization (BGR format)
    # Person-related: Blue/Purple tones
    # Head/Face-related: Cyan/Green tones
    # Body Protection: Orange/Yellow tones
    # Foot Protection: Red/Pink tones
    CLASS_COLORS = {
        0: (200, 100, 0),      # Person - Dark Blue
        1: (255, 200, 0),      # Head - Light Blue
        2: (0, 255, 255),      # Face - Cyan
        3: (0, 255, 0),        # Glasses - Green
        4: (0, 255, 100),      # Face-Mask - Green-Cyan
        5: (100, 255, 0),      # Face-Shield - Yellow-Green
        6: (255, 255, 0),      # Ear - Cyan-Yellow
        7: (0, 165, 255),      # Earmuffs - Orange
        8: (0, 100, 255),      # Hands - Orange-Red
        9: (0, 0, 255),        # Gloves - Red
        10: (255, 0, 255),     # Foot - Magenta
        11: (255, 0, 200),     # Shoes - Pink-Magenta
        12: (0, 165, 255),     # Safety-Vest - Orange
        13: (100, 100, 255),   # Tools - Light Red
        14: (0, 255, 0),       # Helmet - Bright Green
        15: (100, 255, 100),   # Medical-Suit - Light Green
        16: (0, 200, 100)      # Safety-Suit - Green-Cyan
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
    
    def predict_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        save_frames: bool = False,
        frames_dir: Optional[Path] = None,
        show_progress: bool = True
    ) -> Dict:
        """
        Run prediction on video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            save_frames: Whether to save individual frames
            frames_dir: Directory to save frames (if save_frames=True)
            show_progress: Show progress bar
        
        Returns:
            Dictionary with video processing results
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Video properties: {width}x{height} @ {fps} fps, {total_frames} frames")
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height)
            )
        
        # Setup frames directory if needed
        if save_frames and frames_dir:
            frames_dir = Path(frames_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Process video
        frame_predictions = []
        frame_idx = 0
        
        pbar = tqdm(total=total_frames, desc="Processing video") if show_progress else None
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Run inference on frame
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )[0]
                
                # Parse results
                predictions = self._parse_results(results)
                predictions['frame_idx'] = frame_idx
                frame_predictions.append(predictions)
                
                # Draw predictions on frame
                annotated_frame = self._draw_predictions(frame, predictions)
                
                # Write to output video
                if writer:
                    writer.write(annotated_frame)
                
                # Save frame if requested
                if save_frames and frames_dir:
                    frame_filename = frames_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(frame_filename), annotated_frame)
                
                frame_idx += 1
                
                if pbar:
                    pbar.update(1)
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if pbar:
                pbar.close()
        
        # Compile results
        results_dict = {
            'video_path': str(video_path),
            'output_path': str(output_path) if output_path else None,
            'total_frames': frame_idx,
            'fps': fps,
            'resolution': (width, height),
            'frame_predictions': frame_predictions,
            'total_detections': sum(pred['num_detections'] for pred in frame_predictions)
        }
        
        # Get aggregated class counts
        all_class_counts = {name: 0 for name in self.CLASS_NAMES.values()}
        for pred in frame_predictions:
            for det in pred['detections']:
                class_name = det['class_name']
                all_class_counts[class_name] += 1
        
        results_dict['class_counts'] = all_class_counts
        
        self.logger.info(f"Processed {frame_idx} frames, {results_dict['total_detections']} total detections")
        
        return results_dict
