from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Importing training_portal.main constructs the module-level Portal; point it at
# a throwaway directory so tests never touch the project data folder.
os.environ.setdefault(
    "PORTAL_DATA_DIR", tempfile.mkdtemp(prefix="portal-import-")
)
