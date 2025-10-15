"""
Efficient HDF5 image storage module.
Handles single file storage to avoid millions of individual files.
"""
import h5py
import numpy as np
import os
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from .utils import setup_logger, ensure_dir

logger = setup_logger(__name__)


class ImageStorageWriter:
    """Unified writer for different image storage formats."""
    
    def __init__(
        self,
        output_path: str,
        storage_format: str,
        mode: str,
        images_per_file: int,
        resolution: Dict[str, int],
        metadata: Optional[Dict] = None
    ):
        self.output_path = output_path
        self.storage_format = storage_format.lower()
        self.mode = mode
        self.images_per_file = images_per_file
        self.resolution = resolution
        self.metadata = metadata or {}
        self.current_file_idx = 0
        self.images_in_current_file = 0
        self.total_images = 0
        
        ensure_dir(os.path.dirname(output_path) if '.' in os.path.basename(output_path) else output_path)
        
        # Always use HDF5
        self._init_hdf5()
    
    def _get_file_path(self, file_idx: int = 0) -> str:
        """Get file path for batch mode or single mode."""
        if self.mode == 'single':
            base_path = self.output_path
            if not base_path.endswith(self._get_extension()):
                base_path += self._get_extension()
            return base_path
        else:
            base_name = Path(self.output_path).stem
            directory = Path(self.output_path).parent
            ext = self._get_extension()
            return str(directory / f"{base_name}_batch_{file_idx:04d}{ext}")
    
    def _get_extension(self) -> str:
        """Get file extension for current format."""
        extensions = {'hdf5': '.h5', 'zarr': '.zarr', 'npz': '.npz'}
        return extensions.get(self.storage_format, '')
    
    def _init_hdf5(self):
        """Initialize HDF5 file."""
        self.hdf5_file = None
        self.hdf5_images = None
        self.hdf5_sequences = None
        self._create_new_hdf5_file()
    
    def _create_new_hdf5_file(self):
        """Create a new HDF5 file."""
        if self.hdf5_file is not None:
            self.hdf5_file.close()
        
        file_path = self._get_file_path(self.current_file_idx)
        logger.info(f"Creating HDF5 file: {file_path}")
        
        self.hdf5_file = h5py.File(file_path, 'w')
        
        height = self.resolution['height']
        seq_len = self.metadata.get('seq_len', 100)
        width = seq_len * 4  # Auto-calculate width for OHLC bars
        
        max_shape = (None, height, width) if self.mode == 'single' else (self.images_per_file, height, width)
        
        self.hdf5_images = self.hdf5_file.create_dataset(
            'images',
            shape=(0, height, width),
            maxshape=max_shape,
            dtype='float32',
            chunks=(1, height, width),
            compression='gzip',
            compression_opts=4
        )
        
        seq_len = self.metadata.get('seq_len', 100)
        # OHLC sequences have shape (seq_len, 4) for [Open, High, Low, Close]
        self.hdf5_sequences = self.hdf5_file.create_dataset(
            'sequences',
            shape=(0, seq_len, 4),
            maxshape=(None, seq_len, 4) if self.mode == 'single' else (self.images_per_file, seq_len, 4),
            dtype='float32',
            chunks=(1, seq_len, 4),
            compression='gzip',
            compression_opts=4
        )
        
        for key, value in self.metadata.items():
            self.hdf5_file.attrs[key] = value
        
        self.hdf5_file.attrs['resolution'] = [
            width,  # Auto-calculated width
            self.resolution['height']
        ]
        self.hdf5_file.attrs['created_at'] = datetime.now().isoformat()
        
        self.images_in_current_file = 0
    
    def write_batch(self, images: List[np.ndarray], sequences: List[np.ndarray]):
        """Write a batch of images and sequences to HDF5."""
        self._write_batch_hdf5(images, sequences)
    
    def _write_batch_hdf5(self, images: List[np.ndarray], sequences: List[np.ndarray]):
        """Write batch to HDF5."""
        batch_size = len(images)
        
        if self.mode == 'batch' and self.images_in_current_file + batch_size > self.images_per_file:
            remaining = self.images_per_file - self.images_in_current_file
            if remaining > 0:
                self._write_hdf5_arrays(images[:remaining], sequences[:remaining])
            
            self.current_file_idx += 1
            self._create_new_hdf5_file()
            
            if remaining < batch_size:
                self._write_hdf5_arrays(images[remaining:], sequences[remaining:])
        else:
            self._write_hdf5_arrays(images, sequences)
    
    def _write_hdf5_arrays(self, images: List[np.ndarray], sequences: List[np.ndarray]):
        """Write arrays to current HDF5 file."""
        batch_size = len(images)
        current_size = self.hdf5_images.shape[0]
        
        self.hdf5_images.resize((current_size + batch_size, *self.hdf5_images.shape[1:]))
        self.hdf5_sequences.resize((current_size + batch_size, *self.hdf5_sequences.shape[1:]))
        
        for i, (img, seq) in enumerate(zip(images, sequences)):
            self.hdf5_images[current_size + i] = img
            self.hdf5_sequences[current_size + i] = seq
        
        self.images_in_current_file += batch_size
        self.total_images += batch_size
        self.hdf5_file.flush()
    
    def close(self):
        """Close the HDF5 storage writer."""
        if self.hdf5_file is not None:
            self.hdf5_file.close()
            logger.info(f"Closed HDF5 file. Total images: {self.total_images}")


def load_images_from_storage(
    file_path: str,
    storage_format: str,
    indices: Optional[Union[List[int], slice]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load images from HDF5 storage file.
    
    Args:
        file_path: Path to HDF5 file
        storage_format: Format type (must be 'hdf5')
        indices: Optional indices to load (default: load all)
    
    Returns:
        Tuple of (images, sequences, metadata)
    """
    return _load_hdf5(file_path, indices)


def _load_hdf5(file_path: str, indices: Optional[Union[List[int], slice]]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load from HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        if indices is None:
            images = f['images'][:]
            sequences = f['sequences'][:]
        else:
            images = f['images'][indices]
            sequences = f['sequences'][indices]
        
        metadata = dict(f.attrs)
    
    return images, sequences, metadata


def get_storage_info(file_path: str, storage_format: str) -> Dict:
    """
    Get information about stored HDF5 images.
    
    Args:
        file_path: Path to HDF5 file
        storage_format: Format type (must be 'hdf5')
    
    Returns:
        Dictionary with storage information
    """
    with h5py.File(file_path, 'r') as f:
        return {
            'num_images': f['images'].shape[0],
            'image_shape': f['images'].shape[1:],
            'sequence_length': f['sequences'].shape[1],
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
            'metadata': dict(f.attrs)
        }

