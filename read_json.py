from __future__ import annotations

import json
from pathlib import Path

GROUPS_PATH = Path(__file__).with_name("imagenet_visual_groups.json")


def load_groups(path: Path = GROUPS_PATH) -> dict:
    """Load the generated ImageNet visual grouping artifact."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
