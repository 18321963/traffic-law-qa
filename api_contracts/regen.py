from __future__ import annotations

import json
from pathlib import Path

from rag_service.api.app import app

SPEC_PATH = Path(__file__).resolve().parent / "openapi.json"


def dump() -> str:
    return json.dumps(app.openapi(), ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    SPEC_PATH.write_text(dump(), encoding="utf-8", newline="\n")
    print(f"[openapi] 已写出 {SPEC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
