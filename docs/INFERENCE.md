# Inference Guide

Guide for running predictions using trained YOLOv11 models.

## Overview

The inference system supports:
- Single image prediction
- Batch processing
- Real-time visualization
- JSON export
- Confidence thresholding

## Basic Usage

### Single Image

```bash
poetry run python scripts/07_inference.py \
    --weights models/checkpoints/exp_20251118_114655/weights/best.pt \
    --source path/to/image.jpg \
    --save-results
```

Output:
- Annotated image with bounding boxes
- Detection statistics
- Console output with results

### Batch Processing

```bash
poetry run python scripts/07_inference.py \
    --weights models/checkpoints/exp_20251118_114655/weights/best.pt \
    --source path/to/images/ \
    --save-results \
    --save-json
```

Processes all images in directory and saves:
- Annotated images in `results/predictions/`
- JSON file with all detections

## Advanced Options

### Confidence Threshold

```bash
poetry run python scripts/07_inference.py \
    --weights models/checkpoints/exp_20251118_114655/weights/best.pt \
    --source image.jpg \
    --conf-threshold 0.7  # Higher = fewer false positives
```

**Recommended thresholds:**
- High precision: 0.7-0.9
- Balanced: 0.5
- High recall: 0.3-0.4

### Custom Output Directory

```bash
poetry run python scripts/07_inference.py \
    --weights models/checkpoints/exp_20251118_114655/weights/best.pt \
    --source images/ \
    --output-dir results/my_predictions \
    --save-results
```

### Device Selection

```bash
# GPU
--device 0

# CPU (slow)
--device cpu

# Specific GPU
--device 1
```

## Understanding Results

### Console Output

```
PREDICTION RESULTS
==================================================
Image: test_image.jpg
Detections: 3

  Detection 1:
    Class: person-with-helmet-and-ppe
    Confidence: 0.923
  
  Detection 2:
    Class: person-without-safety-gear
    Confidence: 0.856
==================================================
```

### JSON Format

```json
{
  "image_path": "test.jpg",
  "num_detections": 2,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person-with-helmet-and-ppe",
      "confidence": 0.923,
      "bbox": [100, 50, 300, 400],
      "bbox_norm": [0.125, 0.078, 0.375, 0.625]
    }
  ]
}
```

### Visual Output

Annotated images show:
- **Green boxes**: Fully compliant (helmet + PPE)
- **Orange boxes**: Partially compliant (helmet or PPE only)
- **Red boxes**: Non-compliant (no safety gear)

## Batch Statistics

After batch processing:

```
BATCH PREDICTION SUMMARY
==================================================
Images processed: 150
Total detections: 287

Class distribution:
  helmet-and-ppe: 120 (41.8%)
  helmet-only: 52 (18.1%)
  ppe-only: 43 (15.0%)
  no-safety-gear: 72 (25.1%)
==================================================
```

## Use Cases

### 1. Safety Compliance Monitoring

```bash
# Process daily site images
poetry run python scripts/07_inference.py \
    --weights models/checkpoints/exp_20251118_114655/weights/best.pt \
    --source /path/to/site_cameras/ \
    --conf-threshold 0.6 \
    --save-results \
    --save-json

# Parse JSON to generate compliance reports
```

### 2. Real-Time Video Processing

```python
from src.inference.predictor import YOLOPredictor
import cv2

predictor = YOLOPredictor(
    weights_path="models/checkpoints/exp_20251118_114655/weights/best.pt",
    conf_threshold=0.5
)

# Open video
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save frame temporarily
    cv2.imwrite("temp.jpg", frame)
    
    # Run prediction
    results = predictor.predict_image("temp.jpg")
    
    # Process results...
```

### 3. API Integration

```python
from src.inference.predictor import YOLOPredictor
from flask import Flask, request, jsonify

app = Flask(__name__)
predictor = YOLOPredictor("models/checkpoints/exp_20251118_114655/weights/best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    file.save('temp.jpg')
    
    results = predictor.predict_image('temp.jpg')
    return jsonify(results)
```

## Performance Optimization

### Speed vs Accuracy

| Setting | Speed | Accuracy |
|---------|-------|----------|
| Model: YOLOv11n, Conf: 0.5 | ~50 FPS | Lower |
| Model: YOLOv11s, Conf: 0.5 | ~35 FPS | Balanced |
| Model: YOLOv11m, Conf: 0.7 | ~20 FPS | Higher |

### Batch Processing Tips

```bash
# Process in parallel (if multiple GPUs)
# Split dataset and run multiple processes

# GPU 0
poetry run python scripts/07_inference.py \
    --source images_batch_1/ \
    --device 0 &

# GPU 1
poetry run python scripts/07_inference.py \
    --source images_batch_2/ \
    --device 1 &
```

## Evaluation

Evaluate model performance on test set:

```bash
poetry run python scripts/06_evaluate.py \
    --weights models/checkpoints/exp_20251118_114655/weights/best.pt \
    --data data/processed/dataset.yaml \
    --split test
```

Results include:
- mAP@0.5 and mAP@0.5:0.95
- Per-class precision/recall
- Confusion matrix
- F1-scores

## Troubleshooting

### Low Detection Rate

```bash
# Try lower confidence threshold
--conf-threshold 0.3

# Check image quality
# Verify model trained properly
```

### Too Many False Positives

```bash
# Increase confidence threshold
--conf-threshold 0.7

# Adjust IoU threshold
--iou-threshold 0.5
```

### Slow Inference

```bash
# Use GPU
--device 0

# Use smaller model
# Reduce image size in model config
```

### Wrong Predictions

- Check if similar to training data
- Verify model performance (run evaluation)
- Ensure proper preprocessing
- Consider retraining with more data

## Export for Deployment

### ONNX Export

```bash
poetry run python scripts/08_export_model.py \
    --weights models/checkpoints/exp_20251118_114655/weights/best.pt \
    --format onnx \
    --imgsz 640
```

Benefits:
- Platform-independent
- Optimized inference
- Smaller file size

### TensorRT Export

```bash
poetry run python scripts/08_export_model.py \
    --weights models/checkpoints/exp_20251118_114655/weights/best.pt \
    --format engine \
    --imgsz 640 \
    --half  # FP16 precision
```

Benefits:
- Fastest inference on NVIDIA GPUs
- Optimized for specific hardware

## Best Practices

1. **Always validate on test set first**
2. **Use appropriate confidence thresholds**
3. **Monitor inference speed in production**
4. **Log predictions for quality control**
5. **Implement fallback mechanisms**

---

For questions, see `TROUBLESHOOTING.md` or open an issue.
