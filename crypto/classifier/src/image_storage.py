import h5py
import numpy as np
import os
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from .utils import setup_logger, ensure_dir

logger = setup_logger(__name__)


class ImageStorageWriter:
    
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
        
        self._init_hdf5()
    
    def _get_file_path(self, file_idx: int = 0) -> str:
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
        extensions = {'hdf5': '.h5', 'zarr': '.zarr', 'npz': '.npz'}
        return extensions.get(self.storage_format, '')
    
    def _init_hdf5(self):
        self.hdf5_file = None
        self.hdf5_original_data = None
        self.hdf5_coordinates = None
        self._create_new_hdf5_file()
    
    def _create_new_hdf5_file(self):
        if self.hdf5_file is not None:
            self.hdf5_file.close()
        
        file_path = self._get_file_path(self.current_file_idx)
        logger.info(f"Creating HDF5 file: {file_path}")
        
        self.hdf5_file = h5py.File(file_path, 'w')
        
        height = self.resolution['height']
        seq_len = self.metadata.get('seq_len', 100)
        
        self.hdf5_original_data = self.hdf5_file.create_dataset(
            'original_data',
            shape=(0, 4),
            maxshape=(None, 4),
            dtype='float32',
            chunks=(10000, 4)
        )
        
        self.hdf5_coordinates = self.hdf5_file.create_dataset(
            'coordinates',
            shape=(0, seq_len, 4),
            maxshape=(None, seq_len, 4) if self.mode == 'single' else (self.images_per_file, seq_len, 4),
            dtype='uint16',
            chunks=(1000, seq_len, 4)
        )
        
        for key, value in self.metadata.items():
            self.hdf5_file.attrs[key] = value
        
        self.hdf5_file.attrs['resolution'] = [seq_len * 4, height]
        self.hdf5_file.attrs['storage_format'] = 'index_based_implicit'
        self.hdf5_file.attrs['created_at'] = datetime.now().isoformat()
        
        self.images_in_current_file = 0
        self.original_data_written = False
    
    def write_with_original_data(self, original_data: np.ndarray, coordinates: List[np.ndarray]):
        if not self.original_data_written:
            logger.info(f"Writing shared original data: {original_data.shape}")
            self.hdf5_original_data.resize((len(original_data), 4))
            self.hdf5_original_data[:] = original_data
            self.hdf5_file.attrs['total_data_points'] = len(original_data)
            self.original_data_written = True
            logger.info(f"Shared data stored: {len(original_data)} points")
        
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
        if image_idx < 0 or image_idx + seq_len > self.hdf5_original_data.shape[0]:
            raise IndexError(f"Image index {image_idx} with seq_len {seq_len} out of bounds")
        
        return self.hdf5_original_data[image_idx:image_idx + seq_len]

    def load_image_by_index(self, image_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        coordinates = self.hdf5_coordinates[image_idx]
        
        seq_len = self.metadata.get('seq_len', 100)
        sequence = self.hdf5_original_data[image_idx:image_idx + seq_len]
        
        metadata = {
            'height': self.resolution['height'],
            'seq_len': seq_len,
            'width': seq_len * 4
        }
        image = self.recreate_image_from_coordinates(coordinates, metadata)
        
        return image, sequence
    
    def recreate_image_from_coordinates(self, coordinates: np.ndarray, metadata: dict) -> np.ndarray:
        height = metadata['height']
        seq_len = metadata['seq_len']
        width = seq_len * 4
        
        image = np.ones((height, width), dtype=np.float32)
        
        opens_y = coordinates[:, 0]
        highs_y = coordinates[:, 1]
        lows_y = coordinates[:, 2]
        closes_y = coordinates[:, 3]
        
        for bar_idx in range(seq_len):
            image[opens_y[bar_idx], bar_idx * 4 + 0] = 0.0
            
            y_start = min(highs_y[bar_idx], lows_y[bar_idx])
            y_end = max(highs_y[bar_idx], lows_y[bar_idx])
            for y in range(y_start, y_end + 1):
                image[y, bar_idx * 4 + 1] = 0.0
            
            image[closes_y[bar_idx], bar_idx * 4 + 2] = 0.0
        
        return image
    
    def close(self):
        if self.hdf5_file is not None:
            self.hdf5_file.close()
            logger.info(f"Closed HDF5 file. Total images: {self.total_images}")


def load_images_from_storage(
    file_path: str,
    storage_format: str,
    indices: Optional[Union[List[int], slice]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    return _load_hdf5(file_path, indices)


def _load_hdf5(file_path: str, indices: Optional[Union[List[int], slice]]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    with h5py.File(file_path, 'r') as f:
        storage_format = f.attrs.get('storage_format', 'coordinates')
        
        if storage_format == 'index_based_implicit':
            if 'coordinates' not in f or 'original_data' not in f:
                raise ValueError(f"Invalid HDF5 file format: missing required datasets 'coordinates' or 'original_data'")
            
            if indices is None:
                coordinates = f['coordinates'][:]
                original_data = f['original_data'][:]
            else:
                coordinates = f['coordinates'][indices]
                original_data = f['original_data'][:]
            
            images = []
            seq_len = f.attrs.get('seq_len', 100)
            height = f.attrs.get('resolution', [0, 500])[1]
            
            for coord in coordinates:
                metadata = {'height': height, 'seq_len': seq_len, 'width': seq_len * 4}
                image = recreate_image_from_coordinates_static(coord, metadata)
                images.append(image)
            
            images = np.array(images)
            
            sequences = []
            for i in range(len(coordinates)):
                start_idx = i
                end_idx = start_idx + seq_len
                sequence = original_data[start_idx:end_idx]
                sequences.append(sequence)
            
            sequences = np.array(sequences)
            
        else:
            if indices is None:
                images = f['images'][:]
                sequences = f['sequences'][:]
            else:
                images = f['images'][indices]
                sequences = f['sequences'][indices]
        
        metadata = dict(f.attrs)
    
    return images, sequences, metadata


def recreate_image_from_coordinates_static(coordinates: np.ndarray, metadata: dict) -> np.ndarray:
    height = metadata['height']
    seq_len = metadata['seq_len']
    width = seq_len * 4
    
    image = np.ones((height, width), dtype=np.float32)
    
    opens_y = coordinates[:, 0]
    highs_y = coordinates[:, 1]
    lows_y = coordinates[:, 2]
    closes_y = coordinates[:, 3]
    
    for bar_idx in range(seq_len):
        image[opens_y[bar_idx], bar_idx * 4 + 0] = 0.0
        
        y_start = min(highs_y[bar_idx], lows_y[bar_idx])
        y_end = max(highs_y[bar_idx], lows_y[bar_idx])
        for y in range(y_start, y_end + 1):
            image[y, bar_idx * 4 + 1] = 0.0
        
        image[closes_y[bar_idx], bar_idx * 4 + 2] = 0.0
    
    return image


def get_storage_info(file_path: str, storage_format: str) -> Dict:
    with h5py.File(file_path, 'r') as f:
        storage_format_attr = f.attrs.get('storage_format', 'coordinates')
        
        if storage_format_attr == 'index_based_implicit':
            num_images = f['coordinates'].shape[0]
            seq_len = f['coordinates'].shape[1]
            height = f.attrs.get('resolution', [0, 500])[1]
            image_shape = (height, seq_len * 4)
            sequence_length = seq_len
        else:
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


def load_single_image(file_path: str, image_index: int = 0) -> np.ndarray:
    with h5py.File(file_path, 'r') as f:
        storage_format = f.attrs.get('storage_format', 'coordinates')
        
        if storage_format == 'index_based_implicit':
            coordinates = f['coordinates'][image_index]
            seq_len = f['coordinates'].shape[1]
            height = f.attrs.get('resolution', [0, 500])[1]
            metadata = {'height': height, 'seq_len': seq_len, 'width': seq_len * 4}
            return recreate_image_from_coordinates_static(coordinates, metadata)
        else:
            return f['images'][image_index]