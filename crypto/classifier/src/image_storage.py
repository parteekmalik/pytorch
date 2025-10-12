"""
Efficient image storage module supporting multiple formats (HDF5, NPZ, Zarr).
Handles single file or batch file storage to avoid millions of individual files.
"""
import h5py
import numpy as np
import os
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from .utils import setup_logger, ensure_dir

logger = setup_logger(__name__)

try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    logger.warning("Zarr not available. Install with: pip install zarr")


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
        
        if self.storage_format == 'hdf5':
            self._init_hdf5()
        elif self.storage_format == 'zarr':
            self._init_zarr()
        elif self.storage_format == 'npz':
            self.image_buffer = []
            self.sequence_buffer = []
    
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
        width = self.resolution['width']
        
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
        self.hdf5_sequences = self.hdf5_file.create_dataset(
            'sequences',
            shape=(0, seq_len),
            maxshape=(None, seq_len) if self.mode == 'single' else (self.images_per_file, seq_len),
            dtype='float32',
            chunks=(1, seq_len),
            compression='gzip',
            compression_opts=4
        )
        
        for key, value in self.metadata.items():
            self.hdf5_file.attrs[key] = value
        
        self.hdf5_file.attrs['resolution'] = [
            self.resolution['width'],
            self.resolution['height'],
            self.resolution['dpi']
        ]
        self.hdf5_file.attrs['created_at'] = datetime.now().isoformat()
        
        self.images_in_current_file = 0
    
    def _init_zarr(self):
        """Initialize Zarr storage."""
        if not ZARR_AVAILABLE:
            raise ImportError("Zarr is not installed. Install with: pip install zarr")
        
        file_path = self._get_file_path(self.current_file_idx)
        logger.info(f"Creating Zarr file: {file_path}")
        
        self.zarr_root = zarr.open(file_path, mode='w')
        
        height = self.resolution['height']
        width = self.resolution['width']
        seq_len = self.metadata.get('seq_len', 100)
        
        self.zarr_images = self.zarr_root.create_dataset(
            'images',
            shape=(0, height, width),
            chunks=(1, height, width),
            dtype='float32',
            compressor=zarr.Blosc(cname='zstd', clevel=3)
        )
        
        self.zarr_sequences = self.zarr_root.create_dataset(
            'sequences',
            shape=(0, seq_len),
            chunks=(1, seq_len),
            dtype='float32',
            compressor=zarr.Blosc(cname='zstd', clevel=3)
        )
        
        self.zarr_root.attrs.update(self.metadata)
        self.zarr_root.attrs['resolution'] = [
            self.resolution['width'],
            self.resolution['height'],
            self.resolution['dpi']
        ]
        self.zarr_root.attrs['created_at'] = datetime.now().isoformat()
    
    def write_batch(self, images: List[np.ndarray], sequences: List[np.ndarray]):
        """Write a batch of images and sequences."""
        if self.storage_format == 'hdf5':
            self._write_batch_hdf5(images, sequences)
        elif self.storage_format == 'zarr':
            self._write_batch_zarr(images, sequences)
        elif self.storage_format == 'npz':
            self._write_batch_npz(images, sequences)
    
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
    
    def _write_batch_zarr(self, images: List[np.ndarray], sequences: List[np.ndarray]):
        """Write batch to Zarr."""
        batch_size = len(images)
        current_size = self.zarr_images.shape[0]
        
        self.zarr_images.resize((current_size + batch_size, *self.zarr_images.shape[1:]))
        self.zarr_sequences.resize((current_size + batch_size, *self.zarr_sequences.shape[1:]))
        
        for i, (img, seq) in enumerate(zip(images, sequences)):
            self.zarr_images[current_size + i] = img
            self.zarr_sequences[current_size + i] = seq
        
        self.total_images += batch_size
    
    def _write_batch_npz(self, images: List[np.ndarray], sequences: List[np.ndarray]):
        """Buffer data for NPZ (written on close)."""
        self.image_buffer.extend(images)
        self.sequence_buffer.extend(sequences)
        self.total_images += len(images)
        
        if self.mode == 'batch' and len(self.image_buffer) >= self.images_per_file:
            self._flush_npz()
    
    def _flush_npz(self):
        """Flush NPZ buffer to file."""
        if not self.image_buffer:
            return
        
        file_path = self._get_file_path(self.current_file_idx)
        logger.info(f"Saving NPZ file: {file_path}")
        
        np.savez_compressed(
            file_path,
            images=np.array(self.image_buffer),
            sequences=np.array(self.sequence_buffer),
            **self.metadata,
            resolution=np.array([
                self.resolution['width'],
                self.resolution['height'],
                self.resolution['dpi']
            ]),
            created_at=datetime.now().isoformat()
        )
        
        self.image_buffer = []
        self.sequence_buffer = []
        self.current_file_idx += 1
    
    def close(self):
        """Close the storage writer."""
        if self.storage_format == 'hdf5' and self.hdf5_file is not None:
            self.hdf5_file.close()
            logger.info(f"Closed HDF5 file. Total images: {self.total_images}")
        elif self.storage_format == 'npz' and self.image_buffer:
            self._flush_npz()
            logger.info(f"Saved NPZ files. Total images: {self.total_images}")
        elif self.storage_format == 'zarr':
            logger.info(f"Closed Zarr storage. Total images: {self.total_images}")


def load_images_from_storage(
    file_path: str,
    storage_format: str,
    indices: Optional[Union[List[int], slice]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load images from storage file.
    
    Args:
        file_path: Path to storage file
        storage_format: Format type ('hdf5', 'zarr', 'npz')
        indices: Optional indices to load (default: load all)
    
    Returns:
        Tuple of (images, sequences, metadata)
    """
    storage_format = storage_format.lower()
    
    if storage_format == 'hdf5':
        return _load_hdf5(file_path, indices)
    elif storage_format == 'zarr':
        return _load_zarr(file_path, indices)
    elif storage_format == 'npz':
        return _load_npz(file_path, indices)
    else:
        raise ValueError(f"Unsupported format: {storage_format}")


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


def _load_zarr(file_path: str, indices: Optional[Union[List[int], slice]]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load from Zarr storage."""
    if not ZARR_AVAILABLE:
        raise ImportError("Zarr is not installed")
    
    root = zarr.open(file_path, mode='r')
    
    if indices is None:
        images = root['images'][:]
        sequences = root['sequences'][:]
    else:
        images = root['images'][indices]
        sequences = root['sequences'][indices]
    
    metadata = dict(root.attrs)
    
    return images, sequences, metadata


def _load_npz(file_path: str, indices: Optional[Union[List[int], slice]]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load from NPZ file."""
    data = np.load(file_path, allow_pickle=True)
    
    images = data['images']
    sequences = data['sequences']
    
    if indices is not None:
        images = images[indices]
        sequences = sequences[indices]
    
    metadata = {key: data[key].item() if data[key].ndim == 0 else data[key] 
                for key in data.files if key not in ['images', 'sequences']}
    
    return images, sequences, metadata


def get_storage_info(file_path: str, storage_format: str) -> Dict:
    """
    Get information about stored images.
    
    Args:
        file_path: Path to storage file
        storage_format: Format type
    
    Returns:
        Dictionary with storage information
    """
    storage_format = storage_format.lower()
    
    if storage_format == 'hdf5':
        with h5py.File(file_path, 'r') as f:
            return {
                'num_images': f['images'].shape[0],
                'image_shape': f['images'].shape[1:],
                'sequence_length': f['sequences'].shape[1],
                'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
                'metadata': dict(f.attrs)
            }
    elif storage_format == 'zarr':
        if not ZARR_AVAILABLE:
            raise ImportError("Zarr is not installed")
        root = zarr.open(file_path, mode='r')
        return {
            'num_images': root['images'].shape[0],
            'image_shape': root['images'].shape[1:],
            'sequence_length': root['sequences'].shape[1],
            'metadata': dict(root.attrs)
        }
    elif storage_format == 'npz':
        data = np.load(file_path, allow_pickle=True)
        return {
            'num_images': data['images'].shape[0],
            'image_shape': data['images'].shape[1:],
            'sequence_length': data['sequences'].shape[1],
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
        }
    else:
        raise ValueError(f"Unsupported format: {storage_format}")

