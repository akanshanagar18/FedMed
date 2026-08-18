"""
Pytest configuration for FedMed tests.

Ensures that the project root is available on
Python's import path when pytest collects tests.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )