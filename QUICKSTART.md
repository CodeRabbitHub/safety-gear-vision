# Safety Gear Detection - Quick Start Guide

> **📌 This is the fastest way to get started!** Follow these steps to go from setup to training in minutes.

## 📦 What's Included

This complete YOLOv11 safety gear detection system includes:

- ✅ Production-ready Python codebase
- ✅ 8 executable scripts for complete workflow
- ✅ Modular architecture (data, training, inference, evaluation)
- ✅ Configuration files for different YOLOv11 models
- ✅ Comprehensive documentation
- ✅ Requirements and environment files

## 🚀 Immediate Next Steps

### 1. Upload to Your Remote Server

```bash
# On your local machine
scp -r safety-gear-detection user@your-server:~/projects/
```

Or use rsync:
```bash
rsync -avz safety-gear-detection/ user@your-server:~/projects/safety-gear-detection/
```

### 2. On Remote Server

```bash
# Navigate to project
cd ~/projects/safety-gear-vision

# Install dependencies with Poetry
poetry install

# Activate Poetry environment
poetry shell

# Initialize project structure
poetry run python scripts/01_setup_project.py
```

### 3. Add Your Data

```bash
# Copy your images
cp /path/to/your/images/* data/raw/images/

# Copy your labels
cp /path/to/your/labels/* data/raw/labels/
```

### 4. Validate and Prepare

```bash
# Validate dataset
python scripts/02_validate_data.py

# Analyze dataset
python scripts/04_analyze_dataset.py

# Split dataset
python scripts/03_prepare_dataset.py
```

### 5. Start Training

```bash
# In tmux session
tmux new -s yolo-training

poetry run python scripts/05_train.py \
    --config config/training/yolov11s.yaml \
    --experiment-name safety_gear_v1 \
    --epochs 200 \
    --batch-size 24

# Detach: Ctrl+b, then d
```

### 6. Monitor Progress

```bash
# Launch TensorBoard (auto-finds latest run)
poetry run python scripts/09_tensorboard.py

# Or check GPU
nvidia-smi

# View logs
tail -f logs/*.log
```

## 📁 Project Structure

```
safety-gear-vision/
├── src/                    # Core modules
│   ├── data/              # Dataset processing
│   ├── training/          # Training logic
│   ├── inference/         # Predictions
│   ├── evaluation/        # Metrics
│   └── utils/             # Utilities
├── scripts/               # Executable scripts (10 files)
│   ├── 00_download_models.py
│   ├── 01_setup_project.py
│   ├── 02_validate_data.py
│   ├── 03_prepare_dataset.py
│   ├── 04_analyze_dataset.py
│   ├── 05_train.py
│   ├── 06_evaluate.py
│   ├── 07_inference.py
│   ├── 08_export_model.py
│   └── 09_tensorboard.py
├── config/                # Configuration files
├── docs/                  # Documentation
├── data/                  # Data directory
├── models/                # Model storage
│   ├── pretrained/        # YOLOv11 pretrained (5 models)
│   ├── checkpoints/       # Training outputs
│   └── production/        # Final models
├── runs/                  # YOLO training runs
├── results/               # Outputs
├── logs/                  # Log files
├── pyproject.toml         # Poetry dependencies
└── poetry.lock            # Locked dependencies
```

## 📖 Documentation

- `README.md` - Main documentation
- `docs/SETUP.md` - Environment setup
- `docs/TRAINING.md` - Training guide
- `docs/INFERENCE.md` - Inference guide
- `docs/TROUBLESHOOTING.md` - Common issues

## 🎯 Workflow Summary

```
Download  Setup   Validate  Analyze  Prepare  Train  Evaluate  Inference  Export  Monitor
Models  Project    Data     Dataset  Dataset                                       
  ↓       ↓         ↓        ↓        ↓       ↓       ↓         ↓         ↓       ↓
Script  Script   Script   Script   Script  Script Script    Script    Script  Script
  00      01       02       04       03      05     06        07        08      09
```

## ⚙️ Configuration

Training configurations are in `config/training/`:
- `yolov11n.yaml` - Nano (fast, small - testing)
- `yolov11s.yaml` - Small (balanced - recommended)
- `yolov11m.yaml` - Medium (better accuracy)
- `yolov11l.yaml` - Large (high accuracy)
- `yolov11x.yaml` - Extra-Large (maximum accuracy)

Adjust hyperparameters by editing these files or using command-line arguments.

## 🐛 Troubleshooting

See `docs/TROUBLESHOOTING.md` for common issues.

Quick fixes:
- **OOM errors**: Reduce batch size (`--batch-size 8`)
- **Slow training**: Check GPU usage with `nvidia-smi`
- **Import errors**: Reinstall requirements (`pip install -r requirements.txt`)

## 📊 Expected Results

**With GPU (Tesla T4):**
- Training time: 3-4 hours (200 epochs, YOLOv11s)
- mAP@0.5: ~0.85-0.92
- Inference: 30-50 FPS

**With CPU (tested):**
- Training time: Longer (8-12 hours for 200 epochs)
- mAP@0.5: Same accuracy as GPU
- Inference: 2-5 FPS

**Current Status:**
- ✅ Tested with 50 train / 10 val / 10 test images
- ✅ Trained YOLOv11s model (82 epochs, 54.5 MB)
- ✅ All scripts verified and working

## 🔗 Resources

- [YOLOv11 Documentation](https://docs.ultralytics.com/)
- [Project README](README.md)
- [Full Documentation](docs/)

## 📞 Support

For issues:
1. Check `docs/TROUBLESHOOTING.md`
2. Review logs in `logs/`
3. Open GitHub issue or contact maintainer

---

**Ready to start!** Follow the numbered steps above, and you'll have a working safety gear detection system. Good luck! 🎉
