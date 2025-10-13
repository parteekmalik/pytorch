"""Feature extraction utilities for HDF5 image datasets."""

import h5py
import numpy as np
import cv2
from tqdm import tqdm
import gc
from tensorflow.keras.applications.vgg16 import preprocess_input


def extract_features_from_hdf5_batched(hdf5_path, model, size=(224, 224), batch_size=1000):
    """
    Extract VGG16 features from HDF5 file in batches to avoid RAM issues.
    
    Args:
        hdf5_path (Path): Path to HDF5 file containing images
        model: Keras model for feature extraction
        size (tuple): Target size for resizing images (default: (224, 224))
        batch_size (int): Number of images to process at once (default: 1000)
    
    Returns:
        dict: Dictionary of {image_id: feature_vector}
    
    Raises:
        FileNotFoundError: If HDF5 file doesn't exist
    """
    print(f"Extracting features from HDF5: {hdf5_path}")
    
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
    feature_dict = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        total_images = f['images'].shape[0]
        print(f"Total images: {total_images}")
        print(f"Processing in batches of {batch_size}...")
        
        pbar = tqdm(total=total_images, desc="Extracting features", unit="img")
        
        for start_idx in range(0, total_images, batch_size):
            end_idx = min(start_idx + batch_size, total_images)
            batch_size_actual = end_idx - start_idx
            
            batch_images = []
            for i in range(start_idx, end_idx):
                img = f['images'][i]
                img_rgb = np.stack([img, img, img], axis=-1)
                img_resized = cv2.resize(img_rgb, size)
                img_uint8 = (np.clip(img_resized, 0, 1) * 255).astype(np.uint8)
                batch_images.append(img_uint8)
            
            batch_array = np.array(batch_images)
            batch_preprocessed = preprocess_input(batch_array)
            features = model.predict(batch_preprocessed, verbose=0)
            
            for i, feature in enumerate(features):
                image_id = f'image_{start_idx + i:06d}'
                feature_dict[image_id] = feature
            
            del batch_images, batch_array, batch_preprocessed, features
            gc.collect()
            
            pbar.update(batch_size_actual)
        
        pbar.close()
    
    print(f"✓ Extracted features from {len(feature_dict)} images")
    return feature_dict
