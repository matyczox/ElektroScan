"""Small shared helpers for legend extraction modules."""

from __future__ import annotations

import re
from pathlib import Path

def _next_template_index(output_path: Path) -> int:
    """Return next numeric template prefix so repeated legend crops append."""
    max_index = 0
    if output_path.exists():
        for existing in output_path.glob("*.png"):
            match = re.match(r"^(\d+)_", existing.stem)
            if match:
                max_index = max(max_index, int(match.group(1)))
    return max_index + 1
