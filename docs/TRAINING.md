# Training Guide

Complete guide for training YOLOv11 models on safety gear detection.

## Prerequisites

- Completed setup (see `SETUP.md`)
- Data validated and split
- GPU available and tested

## Training Workflow

### 1. Data Validation

```bash
# Validate dataset integrity
python scripts/02_validate_data.py

# Should output: ✓ Dataset is valid!
```

### 2. Dataset Analysis

```bash
# Analyze class distribution and image statistics
python scripts/04_analyze_dataset.py
```

Review the analysis to check for:
- Class imbalance
- Image size consistency
- Annotation quality

### 3. Prepare Dataset

```bash
# Split into train/val/test (80/15/5)
python scripts/03_prepare_dataset.py

# Custom split ratios
python scripts/03_prepare_dataset.py --split-ratio 0.7 0.2 0.1
```

This creates `data/processed/dataset.yaml` needed for training.

## Training Commands

### Basic Training

```bash
# Start training with default config
python scripts/05_train.py \
    --config config/training/yolov11s.yaml \
    --experiment-name my_first_model
```

### Full Training Command

```bash
python scripts/05_train.py \
    --config config/training/yolov11s.yaml \
    --data data/processed/dataset.yaml \
    --experiment-name safety_gear_v1 \
    --epochs 200 \
    --batch-size 16 \
    --imgsz 640 \
    --device 0
```

### Training in Tmux (Recommended)

Tmux keeps training running after SSH disconnect:

```bash
# Create new tmux session
tmux new -s yolo-training

# Start training
python scripts/05_train.py \
    --config config/training/yolov11s.yaml \
    --experiment-name safety_gear_v1

# Detach from tmux: Ctrl+b, then d

# Reattach later
tmux attach -t yolo-training

# Kill session when done
tmux kill-session -t yolo-training
```

## Model Selection

### YOLOv11n (Nano) - Fastest
```bash
python scripts/05_train.py \
    --config config/training/yolov11n.yaml \
    --batch-size 32  # Can use larger batch
```

**Pros**: Fast training, small model size  
**Cons**: Lower accuracy  
**Use case**: Edge deployment, real-time processing

### YOLOv11s (Small) - Recommended
```bash
python scripts/05_train.py \
    --config config/training/yolov11s.yaml \
    --batch-size 16
```

**Pros**: Good balance of speed and accuracy  
**Cons**: None significant  
**Use case**: Most production scenarios (recommended for T4)

### YOLOv11m (Medium) - Higher Accuracy
```bash
python scripts/05_train.py \
    --config config/training/yolov11m.yaml \
    --batch-size 12  # Reduced for T4
```

**Pros**: Better accuracy  
**Cons**: Slower, larger model  
**Use case**: High-accuracy requirements

### YOLOv11l (Large) - Very High Accuracy
```bash
python scripts/05_train.py \
    --config config/training/yolov11l.yaml \
    --batch-size 8  # Further reduced for T4
```

**Pros**: Excellent accuracy  
**Cons**: Slow training, large model size  
**Use case**: Mission-critical applications

### YOLOv11x (Extra-Large) - Maximum Accuracy
```bash
python scripts/05_train.py \
    --config config/training/yolov11x.yaml \
    --batch-size 4  # Minimal for T4
```

**Pros**: Best possible accuracy  
**Cons**: Very slow, very large model  
**Use case**: Offline inference, research

## Configuration Options

### Adjusting Hyperparameters

Edit `config/training/yolov11s.yaml`:

```yaml
# Training duration
epochs: 200          # Increase for better convergence
patience: 50         # Early stopping patience

# Batch and image size
batch: 16           # Reduce if OOM errors
imgsz: 640          # Or 1280 for better accuracy

# Learning rate
lr0: 0.01           # Initial learning rate
lrf: 0.01           # Final LR multiplier

# Data augmentation
mosaic: 1.0         # Mosaic augmentation
mixup: 0.0          # MixUp (can enable: 0.1)
fliplr: 0.5         # Horizontal flip probability
```

### Command-Line Overrides

Override config without editing files:

```bash
python scripts/05_train.py \
    --config config/training/yolov11s.yaml \
    --epochs 300 \           # Override
    --batch-size 12 \        # Override
    --device 0
```

## Monitoring Training

### TensorBoard

1. **On remote server:**
```bash
tensorboard --logdir models/checkpoints --port 6006
```

2. **On local machine (forward port):**
```bash
ssh -L 6006:localhost:6006 user@remote
```

3. **Access:** http://localhost:6006

### GPU Monitoring

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or use nvtop
nvtop

# Check specific GPU
nvidia-smi -i 0 -l 1
```

### Log Files

```bash
# View training logs
tail -f logs/train_*.log

# Or check in results directory
cat models/checkpoints/safety_gear_v1/train.log
```

## Resume Training

If training stops, resume from last checkpoint:

```bash
python scripts/05_train.py \
    --resume models/checkpoints/safety_gear_v1/weights/last.pt \
    --data data/processed/dataset.yaml
```

## Training Best Practices

### 1. Start Small
```bash
# First, test with small dataset/epochs
python scripts/05_train.py \
    --config config/training/yolov11n.yaml \
    --epochs 10
```

### 2. Monitor Metrics
Watch for:
- Loss curves (should decrease)
- mAP@0.5 (should increase)
- Overfitting (val loss > train loss)

### 3. Adjust Based on Results

**If overfitting:**
- Increase data augmentation
- Add dropout/regularization
- Reduce model size

**If underfitting:**
- Increase epochs
- Increase model capacity
- Reduce augmentation

### 4. Class Imbalance

If class distribution is imbalanced:
- Use weighted sampling
- Adjust class weights in config
- Collect more data for minority classes

## Expected Training Time

Tesla T4 GPU estimates:

| Model | Batch Size | Image Size | Time (200 epochs) |
|-------|-----------|------------|-------------------|
| YOLOv11n | 32 | 640 | ~2 hours |
| YOLOv11s | 16 | 640 | ~3-4 hours |
| YOLOv11m | 8 | 640 | ~6-8 hours |

## Troubleshooting

### Out of Memory (OOM)

```bash
# Solution 1: Reduce batch size
--batch-size 8

# Solution 2: Smaller image size
--imgsz 512

# Solution 3: Use smaller model
--config config/training/yolov11n.yaml

# Solution 4: Disable AMP (if needed)
# Edit config: amp: false
```

### Slow Training

```bash
# Check GPU utilization
nvidia-smi

# If GPU underutilized:
# - Increase batch size
# - Increase workers
# - Check data loading speed
```

### No Improvement

If loss plateaus:
- Check learning rate
- Try different optimizer
- Verify data quality
- Check for bugs in labels

### CUDA Errors

```bash
# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"

# Check CUDA version match
python -c "import torch; print(torch.version.cuda)"
nvcc --version  # Should match
```

## After Training

### 1. Find Best Model

```bash
# Best model is saved at:
models/checkpoints/<experiment_name>/weights/best.pt

# Last checkpoint:
models/checkpoints/<experiment_name>/weights/last.pt
```

### 2. Copy to Production

```bash
# Copy best model to production folder
cp models/checkpoints/safety_gear_v1/weights/best.pt \
   models/production/safety_gear_v1.0.pt
```

### 3. Evaluate

See `INFERENCE.md` for evaluation and inference instructions.

## Advanced Topics

### Transfer Learning

Using custom pretrained weights:

```bash
# Download custom weights
# Place in models/pretrained/

# Update config
model: models/pretrained/custom_weights.pt
```

### Multi-GPU Training

```bash
# Use multiple GPUs
python scripts/05_train.py \
    --device 0,1 \  # GPUs 0 and 1
    --batch-size 32
```

### Hyperparameter Tuning

Use Weights & Biases or Optuna for systematic tuning.

---

Training complete! Proceed to evaluation and inference.
