# Complete Project Folder Structure

```
safety-gear-detection/
│
├── 📄 README.md                     # Main documentation
├── 📄 QUICKSTART.md                 # ← START HERE for fastest setup!
├── 📄 BUILD_SUMMARY.md              # Complete testing report
├── 📄 FOLDER_STRUCTURE.md           # This file - directory reference
├── 📄 pyproject.toml                # Python dependencies
├── 📄 .gitignore                    # Git exclusions
│
├── 📁 config/                       # Configuration files
│   └── 📁 training/
│       ├── yolov11n.yaml           # Nano model config (fastest)
│       └── yolov11s.yaml           # Small model config (recommended)
│
├── 📁 data/                         # Data directory
│   ├── 📁 raw/                     # Your original data goes here
│   │   ├── 📁 images/              # ← Put all your images here
│   │   └── 📁 labels/              # ← Put all your labels here
│   │
│   └── 📁 processed/               # Auto-generated after split
│       ├── 📁 images/
│       │   ├── 📁 train/           # Training images (80%)
│       │   ├── 📁 val/             # Validation images (15%)
│       │   └── 📁 test/            # Test images (5%)
│       ├── 📁 labels/
│       │   ├── 📁 train/           # Training labels
│       │   ├── 📁 val/             # Validation labels
│       │   └── 📁 test/            # Test labels
│       └── dataset.yaml            # YOLO dataset config (auto-generated)
│
├── 📁 models/                       # Model storage
│   ├── 📁 pretrained/              # Base YOLO weights (auto-downloaded)
│   ├── 📁 checkpoints/             # Training outputs
│   │   └── 📁 <experiment_name>/
│   │       ├── 📁 weights/
│   │       │   ├── best.pt         # Best model checkpoint
│   │       │   └── last.pt         # Latest checkpoint
│   │       ├── results.csv         # Training metrics
│   │       ├── confusion_matrix.png
│   │       └── ...                 # Other plots and logs
│   └── 📁 production/              # Final production models
│       └── safety_gear_v1.0.pt     # Your deployed model
│
├── 📁 results/                      # Experiment results
│   ├── 📁 experiments/             # Training run outputs
│   ├── 📁 evaluations/             # Evaluation reports
│   │   └── 📁 <model_name>/
│   │       ├── metrics.json
│   │       └── evaluation_report.txt
│   └── 📁 predictions/             # Inference outputs
│       ├── pred_image_001.jpg      # Annotated images
│       └── predictions.json        # Detection results
│
├── 📁 logs/                         # Log files
│   ├── train_20250116_123456.log
│   ├── validate_data_20250116.log
│   └── ...
│
├── 📁 scripts/                      # Executable scripts (8 files)
│   ├── 01_setup_project.py         # Initialize directory structure
│   ├── 02_validate_data.py         # Validate dataset integrity
│   ├── 03_prepare_dataset.py       # Split into train/val/test
│   ├── 04_analyze_dataset.py       # Generate dataset statistics
│   ├── 05_train.py                 # Train YOLOv11 model
│   ├── 06_evaluate.py              # Evaluate model performance
│   ├── 07_inference.py             # Run predictions
│   └── 08_export_model.py          # Export to ONNX/TensorRT
│
├── 📁 src/                          # Source code modules
│   ├── __init__.py
│   │
│   ├── 📁 data/                    # Data processing
│   │   ├── __init__.py
│   │   ├── dataset_splitter.py    # Train/val/test splitting
│   │   ├── data_validator.py      # Dataset integrity checks
│   │   └── dataset_analyzer.py    # EDA and statistics
│   │
│   ├── 📁 training/                # Training logic
│   │   ├── __init__.py
│   │   └── trainer.py             # YOLOv11 training orchestrator
│   │
│   ├── 📁 inference/               # Prediction engine
│   │   ├── __init__.py
│   │   └── predictor.py           # Inference with visualization
│   │
│   ├── 📁 evaluation/              # Model evaluation
│   │   ├── __init__.py
│   │   └── evaluator.py           # Metrics and reporting
│   │
│   └── 📁 utils/                   # Utility modules
│       ├── __init__.py
│       ├── logger.py              # Structured logging
│       ├── config_manager.py      # YAML config handling
│       ├── file_handler.py        # File I/O utilities
│       └── model_utils.py         # YOLO model operations
│
├── 📁 docs/                         # Documentation
│   ├── SETUP.md                    # Environment setup guide
│   ├── TRAINING.md                 # Complete training guide
│   ├── INFERENCE.md                # Inference guide
│   └── TROUBLESHOOTING.md          # Common issues & fixes
│
├── 📁 notebooks/                    # Jupyter notebooks (optional)
│   └── (empty - for your experiments)
│
└── 📁 tests/                        # Unit tests
    └── __init__.py

```

## 📊 File Count Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Python Modules** | 16 | Core source code in `src/` |
| **Scripts** | 10 | Executable scripts in `scripts/` |
| **Config Files** | 6 | YAML training configs (5 models + 1 dataset) |
| **Documentation** | 9 | Markdown docs (README, guides, etc.) |
| **Environment** | 1 | pyproject.toml (Poetry config) |
| **Total Files** | 42+ | Complete production system |

## 🎯 Key Directories Explained

### Where You'll Work:

1. **`data/raw/`** - Put your images and labels here first
2. **`scripts/`** - Run these Python scripts in order (01 → 08)
3. **`config/training/`** - Edit to adjust hyperparameters
4. **`docs/`** - Read these for detailed guides

### Auto-Generated During Workflow:

1. **`data/processed/`** - Created by script 03
2. **`models/checkpoints/`** - Created during training
3. **`results/`** - Created during evaluation/inference
4. **`logs/`** - Created automatically

### Static/Reference:

1. **`src/`** - Python modules (don't need to edit)
2. **`docs/`** - Documentation (reference)
3. **`notebooks/`** - For your experiments (optional)

## 🔥 Critical Paths

```bash
# Your images
data/raw/images/

# Your labels
data/raw/labels/

# Training config
config/training/yolov11s.yaml

# Training script
scripts/05_train.py

# Best trained model
models/checkpoints/<experiment_name>/weights/best.pt

# Inference script
scripts/07_inference.py
```

## 🚀 Workflow Through Folders

```
1. Add data to data/raw/
2. Run scripts/02_validate_data.py
3. Run scripts/03_prepare_dataset.py → Creates data/processed/
4. Run scripts/05_train.py → Creates models/checkpoints/
5. Run scripts/06_evaluate.py → Creates results/evaluations/
6. Run scripts/07_inference.py → Creates results/predictions/
```

---

**Total Size**: ~50 KB (without data/models)  
**After Training**: ~100-500 MB (with models and results)
