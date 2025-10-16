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
        self.hdf5_original_data = None
        self.hdf5_coordinates = None
        self._create_new_hdf5_file()
    
    def _create_new_hdf5_file(self):
        """Create HDF5 file with original data + coordinates (no indices needed)."""
        if self.hdf5_file is not None:
            self.hdf5_file.close()
        
        file_path = self._get_file_path(self.current_file_idx)
        logger.info(f"Creating HDF5 file: {file_path}")
        
        self.hdf5_file = h5py.File(file_path, 'w')
        
        height = self.resolution['height']
        seq_len = self.metadata.get('seq_len', 100)
        
        # Dataset 1: Original OHLC data (shared, no compression)
        self.hdf5_original_data = self.hdf5_file.create_dataset(
            'original_data',
            shape=(0, 4),
            maxshape=(None, 4),
            dtype='float32',
            chunks=(10000, 4)
            # NO compression - already compressed + parallel I/O ready
        )
        
        # Dataset 2: Y-coordinates (already 99.9% compressed, no HDF5 compression)
        self.hdf5_coordinates = self.hdf5_file.create_dataset(
            'coordinates',
            shape=(0, seq_len, 4),
            maxshape=(None, seq_len, 4) if self.mode == 'single' else (self.images_per_file, seq_len, 4),
            dtype='uint16',
            chunks=(1000, seq_len, 4)
            # NO compression - coordinates are already compressed
        )
        
        # NO start_indices dataset - image_idx = start_idx (implicit)
        
        # Store metadata
        for key, value in self.metadata.items():
            self.hdf5_file.attrs[key] = value
        
        self.hdf5_file.attrs['resolution'] = [seq_len * 4, height]
        self.hdf5_file.attrs['storage_format'] = 'index_based_implicit'
        self.hdf5_file.attrs['created_at'] = datetime.now().isoformat()
        
        self.images_in_current_file = 0
        self.original_data_written = False
    
    def write_with_original_data(self, original_data: np.ndarray, coordinates: List[np.ndarray]):
        """
        Write original data and coordinates (no indices needed - implicit indexing).
        
        Args:
            original_data: (total_points, 4) original OHLC data
            coordinates: List of (seq_len, 4) coordinate arrays
        """
        # Write original data (only once - shared dataset)
        if not self.original_data_written:
            logger.info(f"Writing shared original data: {original_data.shape}")
            self.hdf5_original_data.resize((len(original_data), 4))
            self.hdf5_original_data[:] = original_data
            self.hdf5_file.attrs['total_data_points'] = len(original_data)
            self.original_data_written = True
            logger.info(f"Shared data stored: {len(original_data)} points")
        
        # Write coordinates (implicit indexing: image_idx = start_idx)
        batch_size = len(coordinates)
        current_size = self.hdf5_coordinates.shape[0]
        new_size = current_size + batch_size
        
        self.hdf5_coordinates.resize((new_size, self.hdf5_coordinates.shape[1], 
                                       self.hdf5_coordinates.shape[2]))
        
        self.hdf5_coordinates[current_size:new_size] = np.array(coordinates)
        
        self.images_in_current_file += batch_size
        self.total_images += batch_size
        self.hdf5_file.flush()
    
    def get_sequence_by_image_index(self, image_idx: int, seq_len: int) -> np.ndarray:
        """
        Get OHLC sequence from shared original data using implicit indexing.
        
        Args:
            image_idx: Image index (also the start index in original_data)
            seq_len: Sequence length
            
        Returns:
            sequence: (seq_len, 4) OHLC sequence
        """
        # Add bounds checking
        if image_idx < 0 or image_idx + seq_len > self.hdf5_original_data.shape[0]:
            raise IndexError(f"Image index {image_idx} with seq_len {seq_len} out of bounds")
        
        # image_idx IS the start_idx in original_data
        return self.hdf5_original_data[image_idx:image_idx + seq_len]

    def load_image_by_index(self, image_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load image and its original sequence using implicit indexing.
        
        Args:
            image_idx: Image index
            
        Returns:
            image: Recreated (height, width) image
            sequence: Original (seq_len, 4) OHLC sequence
        """
        # Load coordinate
        coordinates = self.hdf5_coordinates[image_idx]
        
        # Load original sequence using implicit indexing
        seq_len = self.metadata.get('seq_len', 100)
        sequence = self.hdf5_original_data[image_idx:image_idx + seq_len]
        
        # Recreate image from coordinates
        metadata = {
            'height': self.resolution['height'],
            'seq_len': seq_len,
            'width': seq_len * 4
        }
        image = self.recreate_image_from_coordinates(coordinates, metadata)
        
        return image, sequence
    
    def recreate_image_from_coordinates(self, coordinates: np.ndarray, metadata: dict) -> np.ndarray:
        """
        Recreate full image from compressed Y-coordinates.
        
        Args:
            coordinates: (seq_len, 4) array of [opens_y, highs_y, lows_y, closes_y]
            metadata: dict with 'height', 'seq_len', 'width'
            
        Returns:
            image: (height, width) grayscale image
        """
        height = metadata['height']
        seq_len = metadata['seq_len']
        width = seq_len * 4
        
        # Create white background
        image = np.ones((height, width), dtype=np.float32)
        
        # Extract coordinates
        opens_y = coordinates[:, 0]
        highs_y = coordinates[:, 1]
        lows_y = coordinates[:, 2]
        closes_y = coordinates[:, 3]
        
        # Draw pixels
        for bar_idx in range(seq_len):
            # Open pixel (x = bar_idx * 4 + 0)
            image[opens_y[bar_idx], bar_idx * 4 + 0] = 0.0
            
            # High-Low line (x = bar_idx * 4 + 1)
            y_start = min(highs_y[bar_idx], lows_y[bar_idx])
            y_end = max(highs_y[bar_idx], lows_y[bar_idx])
            for y in range(y_start, y_end + 1):
                image[y, bar_idx * 4 + 1] = 0.0
            
            # Close pixel (x = bar_idx * 4 + 2)
            image[closes_y[bar_idx], bar_idx * 4 + 2] = 0.0
        
        return image
    
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
        # Check storage format
        storage_format = f.attrs.get('storage_format', 'coordinates')
        
        if storage_format == 'index_based_implicit':
            # New format: original_data + coordinates
            if 'coordinates' not in f or 'original_data' not in f:
                raise ValueError(f"Invalid HDF5 file format: missing required datasets 'coordinates' or 'original_data'")
            
            if indices is None:
                coordinates = f['coordinates'][:]
                original_data = f['original_data'][:]
            else:
                coordinates = f['coordinates'][indices]
                original_data = f['original_data'][:]
            
            # Recreate images from coordinates
            images = []
            seq_len = f.attrs.get('seq_len', 100)
            height = f.attrs.get('resolution', [0, 500])[1]
            
            for coord in coordinates:
                metadata = {'height': height, 'seq_len': seq_len, 'width': seq_len * 4}
                image = recreate_image_from_coordinates_static(coord, metadata)
                images.append(image)
            
            images = np.array(images)
            
            # Extract sequences using implicit indexing
            sequences = []
            for i in range(len(coordinates)):
                start_idx = i
                end_idx = start_idx + seq_len
                sequence = original_data[start_idx:end_idx]
                sequences.append(sequence)
            
            sequences = np.array(sequences)
            
        else:
            # Old format: images + sequences
            if indices is None:
                images = f['images'][:]
                sequences = f['sequences'][:]
            else:
                images = f['images'][indices]
                sequences = f['sequences'][indices]
        
        metadata = dict(f.attrs)
    
    return images, sequences, metadata


def recreate_image_from_coordinates_static(coordinates: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Static version of recreate_image_from_coordinates for loading.
    
    Args:
        coordinates: (seq_len, 4) array of [opens_y, highs_y, lows_y, closes_y]
        metadata: dict with 'height', 'seq_len', 'width'
        
    Returns:
        image: (height, width) grayscale image
    """
    height = metadata['height']
    seq_len = metadata['seq_len']
    width = seq_len * 4
    
    # Create white background
    image = np.ones((height, width), dtype=np.float32)
    
    # Extract coordinates
    opens_y = coordinates[:, 0]
    highs_y = coordinates[:, 1]
    lows_y = coordinates[:, 2]
    closes_y = coordinates[:, 3]
    
    # Draw pixels
    for bar_idx in range(seq_len):
        # Open pixel (x = bar_idx * 4 + 0)
        image[opens_y[bar_idx], bar_idx * 4 + 0] = 0.0
        
        # High-Low line (x = bar_idx * 4 + 1)
        y_start = min(highs_y[bar_idx], lows_y[bar_idx])
        y_end = max(highs_y[bar_idx], lows_y[bar_idx])
        for y in range(y_start, y_end + 1):
            image[y, bar_idx * 4 + 1] = 0.0
        
        # Close pixel (x = bar_idx * 4 + 2)
        image[closes_y[bar_idx], bar_idx * 4 + 2] = 0.0
    
    return image


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
        storage_format_attr = f.attrs.get('storage_format', 'coordinates')
        
        if storage_format_attr == 'index_based_implicit':
            # New format: original_data + coordinates
            num_images = f['coordinates'].shape[0]
            seq_len = f['coordinates'].shape[1]
            height = f.attrs.get('resolution', [0, 500])[1]
            image_shape = (height, seq_len * 4)
            sequence_length = seq_len
        else:
            # Old format: images + sequences
            num_images = f['images'].shape[0]
            image_shape = f['images'].shape[1:]
            sequence_length = f['sequences'].shape[1]
        
        return {
            'num_images': num_images,
            'image_shape': image_shape,
            'sequence_length': sequence_length,
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
            'metadata': dict(f.attrs)
        }

