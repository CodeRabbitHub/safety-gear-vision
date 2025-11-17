# Safety Gear Detection with YOLOv11

A production-grade computer vision system for detecting Personal Protective Equipment (PPE) compliance using YOLOv11.

## 🎯 Project Overview

This system detects and classifies people based on their safety equipment compliance in images:

- **Class 0**: Person with helmet AND PPE (✅ Fully compliant)
- **Class 1**: Person with helmet ONLY (⚠️ Partially compliant)
- **Class 2**: Person with PPE ONLY (⚠️ Partially compliant)
- **Class 3**: Person without safety gear (❌ Non-compliant)

## 📋 Features

- ✅ Modular, production-ready codebase
- ✅ Comprehensive data validation and EDA
- ✅ Automated train/val/test splitting
- ✅ YOLOv11 training with automatic mixed precision
- ✅ TensorBoard integration for monitoring
- ✅ Batch inference with visualization
- ✅ Model evaluation and metrics reporting
- ✅ Model export (ONNX, TorchScript, TFLite)
- ✅ Extensive logging and error handling

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Activate conda environment
conda activate yolo

# Install dependencies
pip install ultralytics pyyaml pillow opencv-python matplotlib seaborn tqdm numpy

# Clone/navigate to project
cd ~/projects/safety-gear-detection

# Initialize project structure
python scripts/01_setup_project.py
```

### 2. Prepare Data

```bash
# Place your data:
# - Images in data/raw/images/
# - Labels in data/raw/labels/ (YOLO format)

# Validate data
python scripts/02_validate_data.py

# Analyze dataset
python scripts/04_analyze_dataset.py

# Split into train/val/test
python scripts/03_prepare_dataset.py
```

### 3. Train Model

```bash
# Start training (in tmux session)
tmux new -s yolo-training

python scripts/05_train.py \
    --config config/training/yolov11s.yaml \
    --experiment-name safety_gear_v1 \
    --epochs 200 \
    --batch-size 16 \
    --device 0

# Detach: Ctrl+b, then d
# Reattach: tmux attach -t yolo-training
```

### 4. Evaluate Model

```bash
python scripts/06_evaluate.py \
    --weights models/checkpoints/safety_gear_v1/weights/best.pt \
    --data data/processed/dataset.yaml
```

### 5. Run Inference

```bash
# Single image
python scripts/07_inference.py \
    --weights models/checkpoints/safety_gear_v1/weights/best.pt \
    --source path/to/image.jpg \
    --save-results

# Batch processing
python scripts/07_inference.py \
    --weights models/checkpoints/safety_gear_v1/weights/best.pt \
    --source path/to/images/ \
    --save-results \
    --save-json
```

## 📁 Project Structure

```
safety-gear-detection/
├── config/                    # Configuration files
│   └── training/             # Training configs
├── data/
│   ├── raw/                  # Original data
│   │   ├── images/          # All images
│   │   └── labels/          # All labels
│   └── processed/            # Split data
│       ├── images/          # train/val/test
│       ├── labels/          # train/val/test
│       └── dataset.yaml     # YOLO config
├── models/
│   ├── pretrained/          # Base YOLO weights
│   ├── checkpoints/         # Training checkpoints
│   └── production/          # Final models
├── src/                      # Source code
│   ├── data/                # Data processing
│   ├── training/            # Training logic
│   ├── inference/           # Prediction
│   ├── evaluation/          # Metrics
│   └── utils/               # Utilities
├── scripts/                  # Executable scripts
├── results/                  # Outputs
├── logs/                     # Log files
└── docs/                     # Documentation
```

## 🔧 Configuration

### Training Configuration

Edit `config/training/yolov11s.yaml` to adjust:
- Epochs, batch size, image size
- Learning rate, optimizer
- Data augmentation
- Early stopping patience

### Model Variants

- `yolov11n.yaml` - Nano (fastest, smallest)
- `yolov11s.yaml` - Small (recommended for T4)
- `yolov11m.yaml` - Medium (higher accuracy)
- `yolov11l.yaml` - Large (very high accuracy)
- `yolov11x.yaml` - Extra-Large (maximum accuracy)

## 📊 Monitoring Training

### TensorBoard

```bash
# Forward port via SSH
ssh -L 6006:localhost:6006 user@remote

# On remote machine
tensorboard --logdir models/checkpoints --port 6006

# Access at: http://localhost:6006
```

### GPU Monitoring

```bash
# Real-time GPU stats
watch -n 1 nvidia-smi

# Or use nvtop
nvtop
```

## 📈 Performance

Typical metrics on safety gear dataset:
- **mAP@0.5**: ~0.85-0.92
- **Inference speed**: 30-50 FPS (T4 GPU)
- **Model size**: 10-25 MB (depending on variant)

## 🛠️ Troubleshooting

### Out of Memory Errors
```bash
# Reduce batch size
python scripts/05_train.py --batch-size 8

# Or use smaller model
--config config/training/yolov11n.yaml
```

### Slow Training
```bash
# Check GPU usage
nvidia-smi

# Increase workers (if CPU bottleneck)
--workers 8
```

## 📖 Documentation

See `docs/` folder for detailed guides:
- `SETUP.md` - Environment setup
- `TRAINING.md` - Training guide
- `INFERENCE.md` - Inference guide
- `TROUBLESHOOTING.md` - Common issues

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit pull request

## 📝 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- YOLO architecture by Joseph Redmon

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [amanroland@gmail.com]

---

**Last Updated**: 2025-01-16
**Version**: 1.0.0
