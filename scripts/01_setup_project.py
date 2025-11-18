#!/usr/bin/env python3
"""
Setup script to initialize project directory structure.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.file_handler import FileHandler


def setup_project():
    """Initialize project directory structure."""
    logger = get_logger('setup', log_dir=project_root / 'logs')
    
    logger.info("Initializing project structure...")
    
    # Define directory structure
    directories = [
        # Data directories
        "data/raw/images",
        "data/raw/labels",
        "data/processed/images/train",
        "data/processed/images/val",
        "data/processed/images/test",
        "data/processed/labels/train",
        "data/processed/labels/val",
        "data/processed/labels/test",
        
        # Model directories
        "models/pretrained",
        "models/checkpoints",
        "models/production",
        
        # Results directories
        "results/experiments",
        "results/evaluations",
        "results/predictions",
        
        # Logs
        "logs",
        
        # Notebooks
        "notebooks",
    ]
    
    for dir_path in directories:
        full_path = project_root / dir_path
        FileHandler.ensure_dir(full_path)
        logger.info(f"Created: {dir_path}")
    
    # Create .gitkeep files in empty directories
    for dir_path in directories:
        full_path = project_root / dir_path
        gitkeep = full_path / '.gitkeep'
        if not gitkeep.exists():
            gitkeep.touch()
    
    logger.info("✓ Project structure initialized successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Place your images in data/raw/images/")
    logger.info("2. Place your labels in data/raw/labels/")
    logger.info("3. Run: python scripts/02_validate_data.py")
    
    return True


if __name__ == "__main__":
    try:
        setup_project()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
