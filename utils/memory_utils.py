"""
Memory management utilities for 16GB M4 MacBook.
"""

import psutil
import gc
import os

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def check_memory_limit(max_memory_mb=4000):
    """Check if memory usage is within limits."""
    current_memory = get_memory_usage()
    if current_memory > max_memory_mb:
        print(f"⚠️  Memory usage high: {current_memory:.1f} MB (limit: {max_memory_mb} MB)")
        return False
    return True

def force_garbage_collection():
    """Force garbage collection to free memory."""
    gc.collect()
    return get_memory_usage()

def get_memory_stats():
    """Get comprehensive memory statistics."""
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

def print_memory_stats():
    """Print formatted memory statistics."""
    stats = get_memory_stats()
    print(f"💾 Memory Stats:")
    print(f"   Current: {stats['current_mb']:.1f} MB")
    print(f"   Available: {stats['available_mb']:.1f} MB")
    print(f"   Total: {stats['total_mb']:.1f} MB")
    print(f"   Usage: {stats['usage_percent']:.1f}%")
    print(f"   Available: {stats['available_percent']:.1f}%")
