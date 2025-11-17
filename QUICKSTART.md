# Safety Gear Detection - Quick Start

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
cd ~/projects/safety-gear-detection

# Activate your conda environment
conda activate yolo

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Initialize project structure
python scripts/01_setup_project.py
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

python scripts/05_train.py \
    --config config/training/yolov11s.yaml \
    --experiment-name safety_gear_v1 \
    --epochs 200 \
    --batch-size 16

# Detach: Ctrl+b, then d
```

### 6. Monitor Progress

```bash
# Check GPU
nvidia-smi

# View logs
tail -f logs/train_*.log

# Or use TensorBoard
tensorboard --logdir models/checkpoints --port 6006
```

## 📁 Project Structure

```
safety-gear-detection/
├── src/                    # Core modules
│   ├── data/              # Dataset processing
│   ├── training/          # Training logic
│   ├── inference/         # Predictions
│   ├── evaluation/        # Metrics
│   └── utils/             # Utilities
├── scripts/               # Executable scripts
│   ├── 01_setup_project.py
│   ├── 02_validate_data.py
│   ├── 03_prepare_dataset.py
│   ├── 04_analyze_dataset.py
│   ├── 05_train.py
│   ├── 06_evaluate.py
│   ├── 07_inference.py
│   └── 08_export_model.py
├── config/                # Configuration files
├── docs/                  # Documentation
├── data/                  # Data directory
├── models/                # Model storage
├── results/               # Outputs
└── logs/                  # Log files
```

## 📖 Documentation

- `README.md` - Main documentation
- `docs/SETUP.md` - Environment setup
- `docs/TRAINING.md` - Training guide
- `docs/INFERENCE.md` - Inference guide
- `docs/TROUBLESHOOTING.md` - Common issues

## 🎯 Workflow Summary

```
Data → Validate → Analyze → Split → Train → Evaluate → Inference
  ↓         ↓         ↓        ↓       ↓        ↓          ↓
Script  Script    Script   Script  Script   Script    Script
  02      02        04       03      05       06        07
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

Tesla T4 GPU:
- Training time: 3-4 hours (200 epochs, YOLOv11s)
- mAP@0.5: ~0.85-0.92
- Inference: 30-50 FPS

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
