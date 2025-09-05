import psutil
import gc
import os
from typing import Dict, Union

def get_memory_usage() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def check_memory_limit(max_memory_mb: int = 4000) -> bool:
    current_memory = get_memory_usage()
    if current_memory > max_memory_mb:
        return False
    return True

def force_garbage_collection() -> float:
    gc.collect()
    return get_memory_usage()

def get_memory_stats() -> Dict[str, float]:
    current_memory = get_memory_usage()
    available_memory = psutil.virtual_memory().available / 1024 / 1024
    total_memory = psutil.virtual_memory().total / 1024 / 1024
    
    return {
        'current_mb': current_memory,
        'available_mb': available_memory,
        'total_mb': total_memory,
        'usage_percent': (current_memory / total_memory) * 100,
        'available_percent': (available_memory / total_memory) * 100
    }

def print_memory_stats() -> None:
    stats = get_memory_stats()
    print(f"💾 Memory: {stats['current_mb']:.1f}MB / {stats['total_mb']:.1f}MB ({stats['usage_percent']:.1f}%)")