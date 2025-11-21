# Safety Gear Detection with YOLOv11

A production-grade computer vision system for detecting Personal Protective Equipment (PPE) compliance using YOLOv11.

> **📖 New to this project?** See [QUICKSTART.md](QUICKSTART.md) for the fastest path to get started!

## 🎯 Project Overview

This system detects and classifies people and PPE items using 17 safety-gear classes (see `data/processed/dataset.yaml`):

- `0` : Person
- `1` : Head
- `2` : Face
- `3` : Glasses
- `4` : Face-Mask-Medical
- `5` : Face-Shield
- `6` : Ear
- `7` : Earmuffs
- `8` : Hands
- `9` : Gloves
- `10`: Foot
- `11`: Shoes
- `12`: Safety-Vest
- `13`: Tools
- `14`: Helmet
- `15`: Medical-Suit
- `16`: Safety-Suit

**Current Status:** ✅ Fully tested and production-ready with trained YOLOv11s model

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

This project uses `poetry` for dependency and virtual environment management. Recommended steps:

```bash
# Install poetry if you don't have it
# pip install poetry

# Install project dependencies and create virtualenv
poetry install

# Initialize project structure (creates logs, models, results folders etc.)
poetry run python scripts/01_setup_project.py
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

Recommended workflow (use `poetry run` to execute scripts inside the project virtualenv):

```bash
# 1) Download pretrained YOLO weights into `models/pretrained/` (required)
poetry run python scripts/00_download_models.py

# 2) Start training (example uses the small model config)
poetry run python scripts/05_train.py \
    --config config/training/yolov11s.yaml \
    --experiment-name safety_gear_v1 \
    --epochs 200 \
    --batch-size 24 \
    --device cpu  # or 0 for GPU

# Optional: run inside tmux/screen if training remotely
```

Notes:
- The trainer expects pretrained weights in `models/pretrained/` (script `00_download_models.py` places them there).
- Training outputs are written to `runs/detect/{experiment_name}/` by default (YOLO standard structure).
- The best model is saved at `runs/detect/{experiment_name}/weights/best.pt`.
- AMP (automatic mixed precision) is enabled by default; set `amp: false` in config if needed.

### 4. Evaluate Model

Example (use the `best.pt` produced by training):

```bash
poetry run python scripts/06_evaluate.py \
    --weights runs/detect/safety_gear_v1/weights/best.pt \
    --data data/processed/dataset.yaml
```

If you don't know the exact experiment name, list the `runs/detect/` or `models/checkpoints/` folders and pick the latest experiment.

### 5. Run Inference

Run inference with a trained model (point `--weights` to the `best.pt` saved by training):

```bash
# Single image
poetry run python scripts/07_inference.py \
    --weights models/checkpoints/exp_20251118_114655/weights/best.pt \
    --source path/to/image.jpg \
    --output-dir results/predictions \
    --save-results

# Batch processing
poetry run python scripts/07_inference.py \
    --weights models/checkpoints/exp_20251118_114655/weights/best.pt \
    --source path/to/images/ \
    --output-dir results/predictions \
    --save-results \
    --save-json
```

The inference script will save annotated images (if `--save-results`) to `results/predictions/` by default and can also output JSON when requested.

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

Start TensorBoard to monitor training:

```bash
# Use the integrated launcher (auto-finds latest run and converts CSV to TensorBoard)
poetry run python scripts/09_tensorboard.py

# Or manually specify directory
tensorboard --logdir runs/detect --port 6006
# For older runs:
tensorboard --logdir models/checkpoints --port 6006

# Access at: http://localhost:6006
# If remote, forward port: ssh -L 6006:localhost:6006 user@server
```

### GPU Monitoring

```bash
# Real-time GPU stats
watch -n 1 nvidia-smi

# Or use nvtop
nvtop
```

## 📈 Performance

Typical metrics will vary by model variant and dataset split. Example ranges:
- **mAP@0.5**: dataset-dependent (often 0.70+ for initial runs)
- **Inference speed**: depends on GPU (Tesla T4 ~30-50 FPS for small models)
- **Model size**: varies by variant (small/medium/large)

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

**Last Updated**: 2025-11-21
**Version**: 1.0.0
**Status**: ✅ Fully Tested & Production Ready
