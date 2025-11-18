"""
File handling utilities for I/O operations.
"""

import shutil
from pathlib import Path
from typing import List, Optional, Union, Tuple
import json


class FileHandler:
    """Utilities for file and directory operations."""
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, create if not.
        
        Args:
            path: Directory path
        
        Returns:
            Path object
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def list_files(
        directory: Union[str, Path],
        extensions: Optional[List[str]] = None,
        recursive: bool = False
    ) -> List[Path]:
        """
        List files in directory with optional filtering.
        
        Args:
            directory: Directory path
            extensions: File extensions to filter (e.g., ['.jpg', '.png'])
            recursive: Search recursively
        
        Returns:
            List of file paths
        """
        directory = Path(directory)
        
        if not directory.exists():
            return []
        
        if recursive:
            files = list(directory.rglob('*'))
        else:
            files = list(directory.glob('*'))
        
        # Filter files only
        files = [f for f in files if f.is_file()]
        
        # Filter by extensions
        if extensions:
            extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                         for ext in extensions]
            files = [f for f in files if f.suffix.lower() in extensions]
        
        return sorted(files)
    
    @staticmethod
    def get_image_label_pairs(
        image_dir: Union[str, Path],
        label_dir: Union[str, Path],
        image_extensions: List[str] = ['.jpg', '.jpeg', '.png']
    ) -> List[Tuple[Path, Path]]:
        """
        Get matching image-label file pairs.
        
        Args:
            image_dir: Image directory
            label_dir: Label directory
            image_extensions: Valid image extensions
        
        Returns:
            List of (image_path, label_path) tuples
        """
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)
        
        images = FileHandler.list_files(image_dir, image_extensions)
        pairs = []
        
        for img_path in images:
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                pairs.append((img_path, label_path))
        
        return pairs
    
    @staticmethod
    def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> Path:
        """
        Copy file from source to destination.
        
        Args:
            src: Source file path
            dst: Destination file path
        
        Returns:
            Destination path
        """
        src = Path(src)
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst
    
    @staticmethod
    def move_file(src: Union[str, Path], dst: Union[str, Path]) -> Path:
        """
        Move file from source to destination.
        
        Args:
            src: Source file path
            dst: Destination file path
        
        Returns:
            Destination path
        """
        src = Path(src)
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return dst
    
    @staticmethod
    def read_json(path: Union[str, Path]) -> dict:
        """Read JSON file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def write_json(data: dict, path: Union[str, Path], indent: int = 2):
        """Write JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=indent)
    
    @staticmethod
    def read_lines(path: Union[str, Path]) -> List[str]:
        """Read file lines."""
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    @staticmethod
    def write_lines(lines: List[str], path: Union[str, Path]):
        """Write lines to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
    
    @staticmethod
    def get_file_size(path: Union[str, Path]) -> int:
        """Get file size in bytes."""
        return Path(path).stat().st_size
    
    @staticmethod
    def get_dir_size(path: Union[str, Path]) -> int:
        """Get total size of directory in bytes."""
        total = 0
        for item in Path(path).rglob('*'):
            if item.is_file():
                total += item.stat().st_size
        return total
