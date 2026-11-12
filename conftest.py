"""Pytest configuration: ensure the project root is on ``sys.path``.

This allows test modules to import from the ``src`` package regardless of
how pytest is invoked.
"""

import sys
from pathlib import Path

_root = str(Path(__file__).parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
