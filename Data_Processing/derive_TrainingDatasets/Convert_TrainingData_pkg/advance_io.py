"""
Advanced I/O optimizations for training dataset derivation

This module provides additional optimizations for I/O-intensive scenarios,
particularly useful when reading from S3 or slow storage.
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time


class DataCache:
    """Simple LRU cache for frequently accessed data"""
    def __init__(self, maxsize=100):
        self.maxsize = maxsize
        self.cache = {}
        self.access_count = {}
    
    def get(self, key):
        """Get item from cache"""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def set(self, key, value):
        """Set item in cache with LRU eviction"""
        if len(self.cache) >= self.maxsize:
            # Remove least recently used item
            lru_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[key] = value
        self.access_count[key] = 1
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.access_count.clear()


# Global cache instance
_data_cache = DataCache(maxsize=50)


def load_with_cache(load_func, filepath):
    """Load data with caching support"""
    cached_data = _data_cache.get(filepath)
    if cached_data is not None:
        return cached_data
    
    data = load_func(filepath)
    _data_cache.set(filepath, data)
    return data


def prefetch_files(filepaths, load_func, max_workers=10):
    """
    Prefetch multiple files concurrently and cache them
    
    Parameters:
    -----------
    filepaths : list
        List of file paths to prefetch
    load_func : callable
        Function to load a single file
    max_workers : int
        Maximum number of concurrent workers
    
    Returns:
    --------
    dict : Dictionary mapping filepath to loaded data
    """
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(load_func, path): path 
            for path in filepaths
        }
        
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                data = future.result()
                results[path] = data
                _data_cache.set(path, data)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                results[path] = None
    
    return results


def batch_load_year_month(channel_name, years, months, inputfiles_func, load_func, max_workers=10):
    """
    Load all year/month combinations for a channel in parallel
    
    Parameters:
    -----------
    channel_name : str
        Name of the channel to load
    years : list
        List of years
    months : list
        List of months
    inputfiles_func : callable
        Function that returns input files dictionary given (YYYY, MM)
    load_func : callable
        Function to load a single file
    max_workers : int
        Maximum number of concurrent loaders
    
    Returns:
    --------
    dict : Nested dictionary {year: {month: data}}
    """
    # Build list of all files to load
    file_list = []
    file_to_ym = {}
    
    for year in years:
        for month in months:
            inputfiles_dic = inputfiles_func(YYYY=year, MM=month)
            filepath = inputfiles_dic[channel_name]
            file_list.append(filepath)
            file_to_ym[filepath] = (year, month)
    
    print(f"Batch loading {len(file_list)} files for channel {channel_name}...")
    start_time = time.time()
    
    # Load all files concurrently
    loaded_data = prefetch_files(file_list, load_func, max_workers=max_workers)
    
    elapsed = time.time() - start_time
    print(f"Loaded {len(file_list)} files in {elapsed:.2f}s ({len(file_list)/elapsed:.2f} files/s)")
    
    # Organize by year/month
    result = {}
    for filepath, data in loaded_data.items():
        year, month = file_to_ym[filepath]
        if year not in result:
            result[year] = {}
        result[year][month] = data
    
    return result


def clear_cache():
    """Clear the data cache"""
    global _data_cache
    _data_cache.clear()
    print("Data cache cleared")


def get_cache_stats():
    """Get cache statistics"""
    return {
        'size': len(_data_cache.cache),
        'maxsize': _data_cache.maxsize,
        'total_accesses': sum(_data_cache.access_count.values())
    }
