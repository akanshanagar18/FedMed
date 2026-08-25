"""
Module: data.datasets.cache

Purpose:
Configurable Dataset Caching Engine for MONAI medical datasets.
Supports standard Dataset, RAM CacheDataset, and Disk PersistentDataset.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from monai.data import CacheDataset, Dataset, PersistentDataset
from monai.transforms import Compose

logger = logging.getLogger(__name__)


def build_monai_dataset(
    data_list: List[Dict[str, Any]],
    transforms: Compose,
    cache_type: str = "persistent",
    cache_dir: Optional[Union[str, Path]] = ".cache/monai",
    cache_num: int = 16,
    num_workers: int = 0,
) -> Dataset:
    """
    Factory function instantiating MONAI Dataset, CacheDataset, or PersistentDataset.
    
    Args:
        data_list: List of sample dictionary entries (e.g. {"image": [...], "label": ...})
        transforms: MONAI transform pipeline Compose object
        cache_type: "dataset", "cache", or "persistent"
        cache_dir: Disk cache location for PersistentDataset
        cache_num: Number of items to cache in RAM for CacheDataset
        num_workers: Number of cache pre-processing worker threads
    """
    norm_cache_type = (cache_type or "dataset").lower().strip()

    if norm_cache_type == "cache":
        logger.info(f"Building MONAI CacheDataset (RAM cache items={cache_num}, workers={num_workers})...")
        return CacheDataset(
            data=data_list,
            transform=transforms,
            cache_num=min(cache_num, max(len(data_list), 1)),
            num_workers=num_workers,
        )
    elif norm_cache_type == "persistent":
        cdir = Path(cache_dir or ".cache/monai")
        cdir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Building MONAI PersistentDataset (Disk cache dir='{cdir}')...")
        return PersistentDataset(
            data=data_list,
            transform=transforms,
            cache_dir=str(cdir),
        )
    else:
        logger.info("Building standard MONAI Dataset (no caching)...")
        return Dataset(data=data_list, transform=transforms)
