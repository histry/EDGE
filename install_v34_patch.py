#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install EDGE V34 boundary-safe Contact-INR with timestamped backups."""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_root", required=True)
    parser.add_argument(
        "--patch_root", default=str(Path(__file__).resolve().parent)
    )
    args = parser.parse_args()

    edge = Path(args.edge_root).resolve()
    patch = Path(args.patch_root).resolve()
    if not (edge / "tools").is_dir() or not (edge / "scripts").is_dir():
        raise RuntimeError(f"Not an EDGE repository: {edge}")

    replacements = [
        line.strip()
        for line in (patch / "REPLACEMENT_FILES.txt")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    backup = edge / (
        "backup_v34_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    for relative in replacements:
        source = patch / relative
        destination = edge / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.exists():
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, saved)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"[INSTALLED] {relative}")

    for relative in replacements:
        if relative.endswith(".sh"):
            path = edge / relative
            path.chmod(path.stat().st_mode | 0o111)

    print(f"[BACKUP] {backup}")
    print("[DONE] V34 regularised C3 boundary patch installed")


if __name__ == "__main__":
    main()
