"""Build visually similar groups for the 1,000 ImageNet-1k classes.

The generated artifact is `imagenet_visual_groups.json`. It assigns every canonical
ImageNet class index to exactly one visual-similarity group.
"""

from __future__ import annotations

import json
from pathlib import Path

GROUPS_PATH = Path(__file__).with_name("imagenet_visual_groups.json")


def load_groups(path: Path = GROUPS_PATH) -> dict:
    """Load the generated ImageNet visual grouping artifact."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    data = load_groups()
    print(f"{data['total_classes']} ImageNet classes segmented into {data['total_groups']} visual groups")
    for group in data["groups"]:
        examples = ", ".join(item["label"] for item in group["classes"][:5])
        print(f"- {group['name']} ({group['count']}): {examples}")


if __name__ == "__main__":
    main()
