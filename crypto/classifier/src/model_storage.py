import joblib
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from .utils import setup_logger, ensure_dir

logger = setup_logger(__name__)


def save_model(
    model: Any,
    model_dir: str,
    metadata: Optional[Dict] = None,
    model_name: Optional[str] = None
) -> str:
    ensure_dir(model_dir)
    
    if model_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"model_{timestamp}"
    
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
    
    joblib.dump(model, model_path, compress=3)
    logger.info(f"Model saved to: {model_path}")
    
    if metadata is None:
        metadata = {}
    
    metadata['saved_at'] = datetime.now().isoformat()
    metadata['model_file'] = model_path
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to: {metadata_path}")
    
    return model_path


def load_model(model_path: str) -> Tuple[Any, Dict]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    logger.info(f"Model loaded from: {model_path}")
    
    metadata_path = model_path.replace('.pkl', '_metadata.json')
    
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Metadata loaded from: {metadata_path}")
    
    return model, metadata


def list_models(model_dir: str) -> List[Dict]:
    if not os.path.exists(model_dir):
        logger.warning(f"Model directory does not exist: {model_dir}")
        return []
    
    models = []
    
    for file in sorted(os.listdir(model_dir)):
        if file.endswith('.pkl'):
            model_path = os.path.join(model_dir, file)
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            
            model_info = {
                'name': Path(file).stem,
                'path': model_path,
                'size_mb': os.path.getsize(model_path) / (1024 * 1024),
                'modified': datetime.fromtimestamp(
                    os.path.getmtime(model_path)
                ).isoformat()
            }
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                model_info['metadata'] = metadata
            
            models.append(model_info)
    
    logger.info(f"Found {len(models)} models in {model_dir}")
    return models


def get_latest_model(model_dir: str) -> Optional[Tuple[Any, Dict]]:
    models = list_models(model_dir)
    
    if not models:
        logger.warning("No models found")
        return None
    
    latest = max(models, key=lambda x: x['modified'])
    logger.info(f"Loading latest model: {latest['name']}")
    
    return load_model(latest['path'])