"""
Module: utils.reproducibility

Purpose:
Production Reproducibility Engine for FedMed v2.0.
Captures complete system, hardware, git, python, framework, seed, and configuration
metadata required to guarantee 100% scientific reproducibility across experiment runs.
"""

import os
import sys
import platform
import subprocess
import time
import json
import logging
from typing import Any, Dict, Optional

import torch
import monai
import flwr

logger = logging.getLogger("reproducibility")


def get_git_metadata() -> Dict[str, str]:
    """Retrieves current Git branch, commit SHA, and dirty state."""
    branch = "unknown"
    commit = "unknown"
    is_dirty = False

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
    except Exception:
        pass

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
    except Exception:
        pass

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        is_dirty = len(status) > 0
    except Exception:
        pass

    return {
        "git_branch": branch,
        "git_commit": commit,
        "git_dirty": str(is_dirty),
    }


def get_hardware_metadata() -> Dict[str, Any]:
    """Captures CPU, RAM, CUDA, and GPU hardware environment information."""
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    gpu_count = torch.cuda.device_count() if cuda_available else 0

    # System RAM (using psutil or fallback os sysconf)
    ram_gb = 0.0
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except ImportError:
        try:
            pages = os.sysconf('SC_PHYS_PAGES')
            page_size = os.sysconf('SC_PAGE_SIZE')
            ram_gb = round((pages * page_size) / (1024 ** 3), 2)
        except Exception:
            ram_gb = 0.0

    # CPU cores
    cpu_count = os.cpu_count() or 1

    return {
        "os_platform": platform.platform(),
        "os_system": platform.system(),
        "os_release": platform.release(),
        "cpu_count": cpu_count,
        "cpu_arch": platform.machine(),
        "ram_gb": ram_gb,
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
    }


def get_dependency_versions() -> Dict[str, str]:
    """Retrieves framework and dependency versions."""
    versions = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "monai_version": monai.__version__,
        "flower_version": flwr.__version__,
    }

    optional_pkgs = ["mlflow", "tensorboard", "tenseal", "opacus", "fastapi", "sqlalchemy"]
    for pkg in optional_pkgs:
        try:
            mod = __import__(pkg)
            versions[f"{pkg}_version"] = getattr(mod, "__version__", "installed")
        except ImportError:
            versions[f"{pkg}_version"] = "not_installed"

    return versions


def collect_reproducibility_metadata(
    seed: int = 42,
    config: Optional[Dict[str, Any]] = None,
    extra_tags: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Consolidates full reproducibility context into a single structured dictionary.
    """
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "random_seed": seed,
        "git": get_git_metadata(),
        "hardware": get_hardware_metadata(),
        "dependencies": get_dependency_versions(),
        "config": config or {},
    }

    if extra_tags:
        metadata["extra"] = extra_tags

    return metadata


def save_reproducibility_report(filepath: str, seed: int = 42, config: Optional[Dict[str, Any]] = None) -> str:
    """Saves reproducibility metadata snapshot to a JSON file."""
    metadata = collect_reproducibility_metadata(seed=seed, config=config)
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved reproducibility metadata snapshot to {filepath}")
    return filepath
