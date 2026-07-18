from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


PROTOCOL_VERSION = "1.0"


def save_protocol(
    path: str | Path,
    protocol: Dict,
) -> Path:
    path = Path(path)

    if path.suffix.lower() != ".dbp":
        path = path.with_suffix(".dbp")

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        **protocol,
    }

    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    return path


def load_protocol(
    path: str | Path,
) -> Dict:
    path = Path(path)

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    return payload
