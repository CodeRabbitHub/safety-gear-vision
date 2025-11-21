# Setup Guide

Complete setup instructions for the Safety Gear Detection system.

## Prerequisites

- Linux server with NVIDIA GPU (Tesla T4 or better)
- CUDA 12.1+ installed
- pyenv installed
- Poetry installed
- SSH access configured (for remote development)

## 1. Verify GPU

```bash
# Check GPU availability
nvidia-smi

# Expected output should show:
# - Tesla T4 or compatible GPU
# - CUDA Version: 12.2 or higher
# - Available memory: ~16GB
```

## 2. Environment Setup

### Option A: Using pyenv and Poetry (Recommended)

```bash
# Install Python 3.12 using pyenv
pyenv install 3.12

# Set local Python version for the project
cd ~/projects/safety-gear-detection
pyenv local 3.12

# Verify Python version
python --version

# Install dependencies using Poetry
poetry install

# Activate Poetry environment
poetry shell
```

### Option B: Manual Python Setup with Poetry

```bash
# Install Python 3.12 using pyenv
pyenv install 3.12

# Navigate to project directory
cd ~/projects/safety-gear-detection

# Set Python version
pyenv local 3.12

# Create Poetry virtual environment with Python 3.12
poetry env use ~/.pyenv/versions/3.12.*/bin/python

# Install dependencies
poetry install
```

### Option C: Using system Python with Poetry

```bash
# Navigate to project directory
cd ~/projects/safety-gear-detection

# Install dependencies using Poetry
poetry install

# Activate environment
poetry shell
```

### Option D: Create New Environment Manually

```bash
# Install Python 3.12 using pyenv
pyenv install 3.12

# Navigate to project directory
cd ~/projects/safety-gear-detection

# Set local Python version
pyenv local 3.12

# Install Poetry dependencies
poetry install
```

## 3. Verify Installation

```bash
# Verify pyenv and Poetry
pyenv --version
poetry --version

# Verify Python version
python --version

# Test PyTorch CUDA (within Poetry environment)
poetry run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test Ultralytics
poetry run python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

Expected output:
```
CUDA available: True
Ultralytics OK
```

## 4. Project Initialization

```bash
# Navigate to project directory
cd ~/projects/safety-gear-detection

# Activate Poetry environment
poetry shell

# Run setup script
poetry run python scripts/01_setup_project.py
```

This creates all necessary directories.

## 5. Data Organization

### Your Current Data Format

You mentioned you have images and YOLO format labels. Organize as follows:

```bash
# Place all images in one folder
data/raw/images/
├── image_001.jpg
├── image_002.jpg
└── ...

# Place corresponding labels in another folder
data/raw/labels/
├── image_001.txt
├── image_002.txt
└── ...
```

### YOLO Label Format

Each `.txt` file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```

All values normalized (0-1):
- `class_id`: 0, 1, 2, or 3
- `x_center, y_center`: box center coordinates
- `width, height`: box dimensions

Example `image_001.txt`:
```
0 0.5 0.4 0.3 0.6
3 0.2 0.7 0.15 0.25
```

## 6. VS Code Remote Setup

### Connect to Remote Server

1. Install "Remote - SSH" extension in VS Code
2. Configure SSH:

```bash
# On your local machine, edit ~/.ssh/config
Host myserver
    HostName your.server.ip
    User your_username
    IdentityFile ~/.ssh/id_rsa
```

3. Connect: `Ctrl+Shift+P` → "Remote-SSH: Connect to Host"
4. Open folder: `/home/your_username/projects/safety-gear-detection`

### Install Extensions on Remote

- Python
- Pylance
- Jupyter (optional)

### Configure Python Interpreter

1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Choose: Poetry environment (typically shown as `.venv` or `poetry-env`)

## 7. Transfer Data to Server

### Option A: Using rsync (Recommended)

```bash
# From your local machine
rsync -avz --progress /path/to/local/images/ user@remote:~/projects/safety-gear-detection/data/raw/images/

rsync -avz --progress /path/to/local/labels/ user@remote:~/projects/safety-gear-detection/data/raw/labels/
```

### Option B: Using scp

```bash
scp -r /path/to/local/images/* user@remote:~/projects/safety-gear-detection/data/raw/images/

scp -r /path/to/local/labels/* user@remote:~/projects/safety-gear-detection/data/raw/labels/
```

### Option C: Direct Upload via VS Code

- Use VS Code's file explorer
- Drag and drop files to `data/raw/images/` and `data/raw/labels/`

## 8. Verification

```bash
# Check data is present
ls data/raw/images/ | wc -l
ls data/raw/labels/ | wc -l

# Should show equal numbers
```

## 9. Next Steps

1. Validate data: `poetry run python scripts/02_validate_data.py`
2. Analyze dataset: `poetry run python scripts/04_analyze_dataset.py`
3. Prepare splits: `poetry run python scripts/03_prepare_dataset.py`

## Troubleshooting

### CUDA Not Available

```bash
# Check CUDA installation
nvcc --version

# Reinstall PyTorch with correct CUDA version (via Poetry)
poetry remove torch torchvision
poetry add torch torchvision --platform linux
```

### pyenv Issues

```bash
# List installed Python versions
pyenv versions

# Rehash pyenv after installation
pyenv rehash

# Check if correct Python is being used
which python
pyenv which python
```

### Poetry Issues

```bash
# Clear Poetry cache
poetry cache clear . --all

# Reinstall dependencies
poetry install --no-cache

# Show current environment
poetry env info
```

### Permission Denied

```bash
# Make scripts executable
chmod +x scripts/*.py

# Fix ownership
chown -R $USER:$USER ~/projects/safety-gear-detection
```

### Import Errors

```bash
# Ensure project root in PYTHONPATH (within Poetry environment)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run scripts via Poetry
poetry run python scripts/02_validate_data.py
```

## Resource Allocation

For Tesla T4 (16GB VRAM):
- **Recommended batch size**: 16 (YOLOv11s)
- **Image size**: 640x640
- **Workers**: 8
- **Expected training time**: 2-6 hours for 200 epochs

---

Setup complete! Proceed to `TRAINING.md` for training instructions.
