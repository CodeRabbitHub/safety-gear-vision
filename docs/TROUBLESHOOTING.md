# Troubleshooting Guide

Common issues and solutions for the Safety Gear Detection system.

## Environment Issues

### CUDA Not Available

**Problem:** `CUDA available: False`

**Solutions:**

1. Check NVIDIA driver:
```bash
nvidia-smi
```

2. Check PyTorch CUDA:
```bash
python -c "import torch; print(torch.version.cuda)"
```

3. Reinstall PyTorch:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

4. Verify CUDA installation:
```bash
nvcc --version
ls /usr/local/cuda/
```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'ultralytics'`

**Solutions:**

```bash
# Verify Poetry environment
poetry shell
poetry show ultralytics

# Reinstall
poetry add ultralytics

# Or reinstall all dependencies
poetry install
```

## Training Issues

### Out of Memory (OOM)

**Problem:** `CUDA out of memory`

**Solutions:**

1. Reduce batch size:
```bash
poetry run python scripts/05_train.py --batch-size 8  # or 4
```

2. Reduce image size:
```bash
poetry run python scripts/05_train.py --imgsz 512
```

3. Use smaller model:
```bash
poetry run python scripts/05_train.py --config config/training/yolov11n.yaml
```

4. Clear GPU memory:
```bash
# Kill other processes
nvidia-smi
kill <PID>

# In Python
import torch
torch.cuda.empty_cache()
```

5. Monitor memory usage:
```bash
watch -n 1 nvidia-smi
```

### Training Crashes

**Problem:** Training stops unexpectedly

**Solutions:**

1. Use tmux to prevent SSH disconnections:
```bash
tmux new -s training
poetry run python scripts/05_train.py ...
# Detach: Ctrl+b, d
```

2. Check logs:
```bash
tail -f logs/train_*.log
```

3. Resume from checkpoint:
```bash
poetry run python scripts/05_train.py \
    --resume models/checkpoints/exp/weights/last.pt
```

### No Improvement in Metrics

**Problem:** Loss plateaus, mAP doesn't improve

**Solutions:**

1. Check data quality:
```bash
poetry run python scripts/02_validate_data.py
poetry run python scripts/04_analyze_dataset.py
```

2. Verify class distribution (should be balanced)

3. Increase training duration:
```bash
--epochs 300
```

4. Adjust learning rate:
```yaml
# In config file
lr0: 0.001  # Lower initial LR
```

5. Check for label errors:
   - Review random samples
   - Verify bbox coordinates
   - Check class IDs

### Slow Training

**Problem:** Training is very slow

**Solutions:**

1. Check GPU utilization:
```bash
nvidia-smi -l 1
```

If GPU usage < 80%:

2. Increase batch size:
```bash
--batch-size 32
```

3. Increase workers:
```bash
# Edit config
workers: 8  # or 16
```

4. Check data loading:
```python
# Profile data loading time
import time
from torch.utils.data import DataLoader
# Time your data pipeline
```

5. Use faster storage (SSD vs HDD)

## Data Issues

### Validation Errors

**Problem:** `poetry run python scripts/02_validate_data.py` shows errors

**Common Issues:**

1. **Missing labels:**
```
Error: Found 50 images without labels
```
Solution: Ensure every image has corresponding .txt file

2. **Invalid bbox coordinates:**
```
Error: Box coordinates out of range
```
Solution: Verify coordinates are normalized (0-1)

3. **Wrong class IDs:**
```
Error: Invalid class 5
```
Solution: Ensure class IDs are 0-3 only

4. **Corrupted images:**
```
Error: Cannot open image
```
Solution: Remove or replace corrupted files

### Class Imbalance

**Problem:** One class dominates dataset

**Solutions:**

1. Collect more data for minority classes

2. Use weighted sampling:
```python
# In training code, add class weights
```

3. Apply data augmentation more to minority classes

### Image-Label Mismatch

**Problem:** Labels don't match images

**Solutions:**

1. Verify naming convention:
```bash
# Image: image_001.jpg
# Label: image_001.txt (same stem)
```

2. Check case sensitivity:
```bash
# Rename if needed
rename 'y/A-Z/a-z/' *.jpg
```

## Inference Issues

### Low Detection Rate

**Problem:** Model detects very few objects

**Solutions:**

1. Lower confidence threshold:
```bash
--conf-threshold 0.3
```

2. Check if test images similar to training data

3. Verify model performance:
```bash
poetry run python scripts/06_evaluate.py ...
```

### Too Many False Positives

**Problem:** Detecting objects that aren't there

**Solutions:**

1. Increase confidence threshold:
```bash
--conf-threshold 0.7
```

2. Retrain with more negative examples

3. Check if test domain matches training domain

### Wrong Classifications

**Problem:** Detects correct object, wrong class

**Solutions:**

1. Check class confusion matrix from evaluation

2. Collect more training data for confused classes

3. Review training labels for errors

4. Increase training epochs

## File System Issues

### Permission Denied

**Problem:** Cannot write to directory

**Solutions:**

```bash
# Fix ownership
sudo chown -R $USER:$USER ~/projects/safety-gear-detection

# Fix permissions
chmod -R 755 ~/projects/safety-gear-detection

# Make scripts executable
chmod +x scripts/*.py
```

### Disk Space

**Problem:** No space left on device

**Solutions:**

```bash
# Check disk usage
df -h

# Check project size
du -sh ~/projects/safety-gear-detection/*

# Clean up
rm -rf models/checkpoints/old_experiments
rm -rf results/old_results
```

### Path Issues

**Problem:** File not found errors

**Solutions:**

1. Use absolute paths or paths relative to project root

2. Check current directory:
```bash
pwd
# Should be: /home/user/projects/safety-gear-detection
```

3. Add project to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$HOME/projects/safety-gear-detection"
```

## VS Code Remote Issues

### Connection Drops

**Problem:** SSH connection drops during training

**Solutions:**

1. Use tmux/screen:
```bash
tmux new -s training
# Run training
# Detach: Ctrl+b, d
```

2. Configure SSH keepalive:
```bash
# In ~/.ssh/config
Host *
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

### Slow File Transfer

**Problem:** Uploading data is very slow

**Solutions:**

1. Use rsync instead of scp:
```bash
rsync -avz --progress local/ remote:~/path/
```

2. Compress before transfer:
```bash
tar -czf images.tar.gz images/
scp images.tar.gz remote:~/
ssh remote "cd ~/path && tar -xzf images.tar.gz"
```

3. Use tools like FileZilla for large transfers

## Performance Issues

### Inference Too Slow

**Problem:** Predictions take too long

**Solutions:**

1. Use GPU:
```bash
--device 0
```

2. Use smaller model:
```bash
# YOLOv11n instead of YOLOv11m
```

3. Reduce image size:
```bash
--imgsz 512
```

4. Export to TensorRT:
```bash
poetry run python scripts/08_export_model.py --format engine
```

5. Batch processing:
```python
# Process multiple images at once
predictor.predict_batch(...)
```

## Getting Help

If issue persists:

1. Check logs:
```bash
cat logs/train_*.log
cat logs/validate_data_*.log
```

2. Enable debug mode:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

3. Check GPU status:
```bash
nvidia-smi
nvidia-smi -q  # Detailed info
```

4. Verify installation:
```bash
python -c "from ultralytics import YOLO; print('OK')"
python -c "import torch; print(torch.cuda.is_available())"
```

5. Open GitHub issue with:
   - Error message
   - Full command used
   - System info (GPU, CUDA version)
   - Relevant logs

---

Still having issues? Contact [your-email] or open an issue on GitHub.
