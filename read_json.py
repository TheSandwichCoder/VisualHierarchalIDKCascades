from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GROUPS_PATH = ROOT / "imagenet_visual_groups.json"
PROCESSED_GROUPS_PATH = ROOT / "data" / "processed_data" / "imagenet_visual_groups.json"


def load_groups(path: Path = GROUPS_PATH) -> dict:
    """Load the generated ImageNet visual grouping artifact."""
    if not path.exists() and path == GROUPS_PATH:
        path = PROCESSED_GROUPS_PATH
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
